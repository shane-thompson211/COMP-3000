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