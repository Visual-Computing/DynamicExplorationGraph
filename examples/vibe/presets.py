from pathlib import Path
from typing import Any, Dict, List
import yaml


def load_vibe_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Loads and parses the VIBE config.yml file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Extract default DEG algorithm run_group
    alg_entry = data["float"]["any"][0]
    base_group = alg_entry["run_groups"]["base"]
    args = base_group["args"]
    query_args = base_group["query_args"]

    return {
        "k_list": args.get("k", [30]),
        "extend_k_list": args.get("extend_k", [60]),
        "extend_eps_list": args.get("extend_eps", args.get("build_eps", [0.1])),
        "opt_target_list": args.get("opt_target", ["HighLID"]),
        "improve_k_list": args.get("improve_k", [0]),
        "improve_eps_list": args.get("improve_eps", [0.0]),
        "prune_non_rng_list": args.get("prune_non_rng", [False]),
        "threads_list": args.get("threads", [1]),
        "query_dtype_list": args.get("query_dtype", ["float32"]),
        "search_eps_list": query_args.get("search_eps", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]),
        "rerank_size_factors": query_args.get("rerank_size_factor", [1.0, 1.2, 1.5, 2.0]),
    }


import itertools


def get_config_grid_presets(dataset_key: str) -> List[Dict[str, Any]]:
    """
    Returns the list of all index configurations (Cartesian product of build args)
    defined in config.yml, tailored with the appropriate optimization target for the dataset.
    """
    cfg = load_vibe_config()
    is_l2 = "euclidean" in dataset_key.lower() or "agnews" in dataset_key.lower()
    default_opt = "LowLID" if is_l2 else "HighLID"

    # Resolve opt_targets from config.yml (mapping '' to default_opt)
    raw_opt_targets = cfg.get("opt_target_list", [default_opt])
    resolved_opt_targets = []
    for t in raw_opt_targets:
        target_name = default_opt if t == "" else t
        if target_name not in resolved_opt_targets:
            resolved_opt_targets.append(target_name)

    grid = []
    # Cartesian product: iterate over opt_target first, then all k, improvement, pruning and query_dtype variants
    for opt_target, k, extend_k, extend_eps, imp_k, imp_eps, prune_non_rng, q_dtype in itertools.product(
        resolved_opt_targets,
        cfg["k_list"],
        cfg["extend_k_list"],
        cfg["extend_eps_list"],
        cfg["improve_k_list"],
        cfg["improve_eps_list"],
        cfg["prune_non_rng_list"],
        cfg["query_dtype_list"],
    ):
        grid.append(
            {
                "k": k,
                "extend_k": extend_k,
                "extend_eps": extend_eps,
                "optimization_target": opt_target,
                "improve_k": imp_k,
                "improve_eps": imp_eps,
                "prune_non_rng": prune_non_rng,
                "query_dtype": q_dtype,
                "anns_k": 100,
                "search_eps_list": cfg["search_eps_list"],
                "rerank_size_factors": cfg["rerank_size_factors"],
            }
        )
    return grid


def get_default_config_preset(dataset_key: str) -> Dict[str, Any]:
    """Returns dataset-tailored defaults while reading search grid from config.yml."""
    presets = get_config_grid_presets(dataset_key)
    return (
        presets[0]
        if presets
        else {
            "k": 30,
            "extend_k": 60,
            "extend_eps": 0.1,
            "optimization_target": "LowLID"
            if ("euclidean" in dataset_key.lower() or "agnews" in dataset_key.lower())
            else "HighLID",
            "improve_k": 0,
            "improve_eps": 0.0,
            "prune_non_rng": False,
            "query_dtype": "float32",
            "anns_k": 100,
            "search_eps_list": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            "rerank_size_factors": [1.0, 1.2, 1.5, 2.0],
        }
    )
