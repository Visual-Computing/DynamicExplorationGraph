from typing import Dict, Any
from deglib.distances import Metric

# Presets for dynamic data benchmark (mirrors bench_dynamic_data.cpp DynamicConfig).
# All datasets use StreamingData optimization target.
# No exploration test parameters needed.
DYNAMIC_DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "sift1m": {
        "metric": Metric.FP32_L2,
        "k": 30,
        "extend_k": 60,
        "build_eps": 0.1,
        "optimization_target": "StreamingData",
        "anns_k": 100,
        "anns_repeat": 1,
        "search_eps_list": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
    },
    "audio": {
        "metric": Metric.FP32_L2,
        "k": 20,
        "extend_k": 40,
        "build_eps": 0.1,
        "optimization_target": "StreamingData",
        "anns_k": 100,
        "anns_repeat": 50,
        "search_eps_list": [0.0, 0.03, 0.05, 0.07, 0.09, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.6, 2.0],
    },
    "enron": {
        "metric": Metric.FP32_L2,
        "k": 30,
        "extend_k": 60,
        "build_eps": 0.1,
        "optimization_target": "StreamingData",
        "anns_k": 100,
        "anns_repeat": 50,
        "search_eps_list": [0.0, 0.03, 0.05, 0.07, 0.09, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.6, 2.0],
    },
    "deep1m": {
        "metric": Metric.FP32_L2,
        "k": 30,
        "extend_k": 60,
        "build_eps": 0.1,
        "optimization_target": "StreamingData",
        "anns_k": 100,
        "anns_repeat": 1,
        "search_eps_list": [0.01, 0.02, 0.03, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0],
    },
    "glove": {
        "metric": Metric.FP32_InnerProduct,
        "k": 30,
        "extend_k": 60,
        "build_eps": 0.1,
        "optimization_target": "StreamingData",
        "anns_k": 100,
        "anns_repeat": 1,
        "search_eps_list": [0.001, 0.05, 0.10, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0],
    },
}

def get_preset(dataset_key: str) -> Dict[str, Any]:
    """Returns preset graph build and benchmark parameters for dataset_key."""
    key = dataset_key.lower()
    if key not in DYNAMIC_DATASET_PRESETS:
        raise ValueError(f"No preset for dataset '{dataset_key}'. Standard presets: {list(DYNAMIC_DATASET_PRESETS.keys())}")
    return DYNAMIC_DATASET_PRESETS[key]
