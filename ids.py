import os
import csv
from datetime import datetime

import numpy as np
import joblib
from scapy.all import PcapReader, IP, TCP, UDP

MODEL_PATH = "JN/ids_xgb_scapy.pkl"
FEATURES_PATH = "JN/ids_features_scapy.pkl"

PCAP_FILE = "JN/1.pcap"
MAX_PACKETS_TO_PROCESS = 300000   # Safety limit for big PCAPs
FLOW_TIMEOUT = 2.0                # Seconds of inactivity before classifying a flow

# ✅ CHANGED FILENAME (as requested)
OUTPUT_CSV = "dashboard/data/ids_live_feed.csv"

# Human-readable protocol map
PROTO_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}

# =========================
# LOAD MODEL + FEATURE LIST
# =========================
print(f"Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading feature list: {FEATURES_PATH}")
FEATURES = joblib.load(FEATURES_PATH)

# You said your model expects 13 features, so we enforce that here
if len(FEATURES) != 13:
    raise ValueError(f"Expected 13 features, got {len(FEATURES)}: {FEATURES}")

print("Feature order:", FEATURES)

flows = {}


def get_l4_info(pkt):
    """
    Return (proto_num, sport, dport, l4_name) for TCP/UDP packets.
    If the packet isn't TCP/UDP, return None.
    """
    if TCP in pkt:
        return 6, int(pkt[TCP].sport), int(pkt[TCP].dport), "TCP"
    if UDP in pkt:
        return 17, int(pkt[UDP].sport), int(pkt[UDP].dport), "UDP"
    return None


def canonical_flow_key(ip_src, sport, ip_dst, dport, proto):
    """
    Build a bidirectional key so both directions map to the same flow.
    This makes A->B and B->A packets update the same flow object.
    """
    a = (ip_src, sport)
    b = (ip_dst, dport)
    if a <= b:
        return (a[0], a[1], b[0], b[1], proto)
    else:
        return (b[0], b[1], a[0], a[1], proto)


def is_forward_direction(flow_key, ip_src, sport):
    """
    In a canonical flow_key, A is the "source side".
    If the packet matches A, it's forward direction (s* stats).
    Otherwise it's reverse direction (d* stats).
    """
    a_ip, a_port, b_ip, b_port, proto = flow_key
    return (ip_src, sport) == (a_ip, a_port)


def init_flow(now, flow_key):
    """
    Create a new flow stats object.
    """
    return {
        "start": now,
        "end": now,
        "proto": flow_key[4],
        "spkts": 0,
        "dpkts": 0,
        "sbytes": 0,
        "dbytes": 0,
        "sttl": None,
        "dttl": None,
    }


def update_flow(pkt):
    """
    Update flow stats using a single packet.
    """
    if IP not in pkt:
        return

    l4 = get_l4_info(pkt)
    if l4 is None:
        return

    ip = pkt[IP]
    proto, sport, dport, _ = l4

    # Use packet timestamp for correct durations/rates
    now = float(pkt.time)

    key = canonical_flow_key(ip.src, sport, ip.dst, dport, proto)
    if key not in flows:
        flows[key] = init_flow(now, key)

    flow = flows[key]
    flow["end"] = now

    pkt_len = int(len(pkt))
    fwd = is_forward_direction(key, ip.src, sport)

    if fwd:
        flow["spkts"] += 1
        flow["sbytes"] += pkt_len
        if flow["sttl"] is None and hasattr(ip, "ttl"):
            flow["sttl"] = int(ip.ttl)
    else:
        flow["dpkts"] += 1
        flow["dbytes"] += pkt_len
        if flow["dttl"] is None and hasattr(ip, "ttl"):
            flow["dttl"] = int(ip.ttl)


def flow_to_feature_dict(flow):
    """
    Compute the 13 features exactly as your training expected.

    IMPORTANT:
    - We return a dict with keys that match FEATURES names.
    - Then we build the vector using FEATURES order.
    """
    dur = float(flow["end"] - flow["start"])
    if dur <= 0:
        dur = 1e-6  # avoid divide-by-zero

    spkts = int(flow["spkts"])
    dpkts = int(flow["dpkts"])
    sbytes = int(flow["sbytes"])
    dbytes = int(flow["dbytes"])

    rate = (spkts + dpkts) / dur
    sload = sbytes / dur
    dload = dbytes / dur
    smean = (sbytes / spkts) if spkts > 0 else 0.0
    dmean = (dbytes / dpkts) if dpkts > 0 else 0.0

    feats = {
        "dur": dur,
        "proto": int(flow["proto"]),
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "sttl": int(flow["sttl"]) if flow["sttl"] is not None else 0,
        "dttl": int(flow["dttl"]) if flow["dttl"] is not None else 0,
        "rate": rate,
        "sload": sload,
        "dload": dload,
        "smean": smean,
        "dmean": dmean,
    }

    # Ensure all expected keys exist (safety)
    for f in FEATURES:
        if f not in feats:
            raise KeyError(f"Missing feature '{f}' in computed features. Check your FEATURES list.")

    return feats


def feature_dict_to_vector(feat_dict):
    """
    Convert feature dict to numpy vector in the exact training order.
    """
    vec = np.array([feat_dict[f] for f in FEATURES], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec

# =========================
# CSV OUTPUT HELPERS
# =========================
def append_prediction_row(row: dict, path: str):
    """
    Append a single row to CSV.
    If file doesn't exist, we create it and write headers first.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def classify_and_write(flow_key, flow):
    """
    Convert a flow to features, classify it, and write one event row to the CSV feed.
    """
    feat_dict = flow_to_feature_dict(flow)
    feats_vec = feature_dict_to_vector(feat_dict).reshape(1, -1)

    # Model prediction (0/1)
    pred = int(model.predict(feats_vec)[0])

    # Probability of attack (if model supports it)
    p_attack = None
    if hasattr(model, "predict_proba"):
        p_attack = float(model.predict_proba(feats_vec)[0][1])

    # Severity label (simple SOC-friendly logic)
    if pred == 0:
        severity = "None"
    else:
        if p_attack is None:
            severity = "Medium"
        elif p_attack >= 0.95:
            severity = "High"
        elif p_attack >= 0.85:
            severity = "Medium"
        else:
            severity = "Low"

    # Flow identification
    a_ip, a_port, b_ip, b_port, proto_num = flow_key
    proto_str = PROTO_MAP.get(proto_num, f"OTHER({proto_num})")

    # Timestamp (use flow end time)
    ts = float(flow["end"])
    timestamp_iso = datetime.utcfromtimestamp(ts).isoformat()

    # Build a dashboard-friendly row:
    # - SOC columns
    # - Model outputs
    # - PLUS the 13 features so Streamlit can plot/explain
    row = {
        # SOC / analyst fields
        "timestamp": timestamp_iso,
        "src_ip": a_ip,
        "src_port": int(a_port),
        "dst_ip": b_ip,
        "dst_port": int(b_port),
        "proto_str": proto_str,
        "proto_num": int(proto_num),

        # Model outputs (use consistent names)
        "prediction": int(pred),  # 0 benign, 1 malicious
        "malicious_prob": float(p_attack) if p_attack is not None else (1.0 if pred == 1 else 0.0),
        "severity": severity,

        # Include the 13 features (same names as training)
        **feat_dict,
    }

    append_prediction_row(row, OUTPUT_CSV)


def sweep_timeouts(now):
    """
    Classify and remove flows inactive for longer than FLOW_TIMEOUT.
    """
    for key, flow in list(flows.items()):
        if (now - flow["end"]) > FLOW_TIMEOUT:
            classify_and_write(key, flow)
            del flows[key]


# =========================
# MAIN
# =========================
print(f"Opening PCAP: {PCAP_FILE}")
count = 0
last_now = None

with PcapReader(PCAP_FILE) as pcap:
    print("Processing packets...")
    for pkt in pcap:
        count += 1
        if count > MAX_PACKETS_TO_PROCESS:
            print(f"Reached MAX_PACKETS_TO_PROCESS={MAX_PACKETS_TO_PROCESS}")
            break

        try:
            # Only track IP + TCP/UDP packets
            if IP in pkt and (TCP in pkt or UDP in pkt):
                update_flow(pkt)
                last_now = float(pkt.time)
                sweep_timeouts(last_now)
        except Exception:
            # Ignore malformed packets / edge errors
            continue

        if count % 5000 == 0:
            print(f"  ...processed {count} packets | active flows={len(flows)}")

print("\n--- Final Flush (classifying remaining flows) ---")
if last_now is None:
    last_now = 0.0

for key, flow in list(flows.items()):
    classify_and_write(key, flow)
    del flows[key]

print(f"Done. IDS feed written to: {OUTPUT_CSV}")