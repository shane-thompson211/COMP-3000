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