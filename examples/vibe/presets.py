import itertools
from pathlib import Path
from typing import Any, Dict, List
import yaml


def load_vibe_config(config_path: Path | None = None) -> List[Dict[str, Any]]:
    """Loads and parses all algorithms from the VIBE config.yml file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    algorithms = []
    for entry in data.get("float", {}).get("any", []):
        if entry.get("disabled", False):
            continue

        alg_name = entry.get("name", "deg")
        constructor = entry.get("constructor", "DEG")
        run_groups = entry.get("run_groups", {})

        for rg_name, rg_data in run_groups.items():
            args = rg_data.get("args", {})
            query_args = rg_data.get("query_args", {})
            raw_search_vals = query_args.get("eps_or_ef", [])

            algorithms.append(
                {
                    "name": alg_name,
                    "constructor": constructor,
                    "run_group": rg_name,
                    "k_list": args.get("k", [30]),
                    "opt_target_list": args.get("opt_target", ["LowLID"]),
                    "prune_non_rng_list": args.get("prune_non_rng", [False]),
                    "eps_or_ef_list": [float(v) for v in raw_search_vals],
                    "rerank_size_factors": query_args.get("rerank_size_factor", [1.0]),
                }
            )

    return algorithms


def get_config_grid_presets(dataset_key: str, algorithm_name: str | None = None) -> List[Dict[str, Any]]:
    """
    Returns the list of all index configurations (Cartesian product of build args)
    defined in config.yml.
    """
    algorithms = load_vibe_config()
    is_l2 = "euclidean" in dataset_key.lower() or "agnews" in dataset_key.lower()
    default_opt = "LowLID" if is_l2 else "HighLID"

    grid = []
    for alg in algorithms:
        if algorithm_name and alg["name"] != algorithm_name:
            continue

        raw_opt_targets = alg.get("opt_target_list", [default_opt])
        resolved_opt_targets = []
        for t in raw_opt_targets:
            target_name = default_opt if t == "" else t
            if target_name not in resolved_opt_targets:
                resolved_opt_targets.append(target_name)

        for opt_target, k, prune_non_rng in itertools.product(
            resolved_opt_targets,
            alg["k_list"],
            alg["prune_non_rng_list"],
        ):
            grid.append(
                {
                    "alg_name": alg["name"],
                    "constructor": alg["constructor"],
                    "k": k,
                    "optimization_target": opt_target,
                    "prune_non_rng": prune_non_rng,
                    "anns_k": 100,
                    "eps_or_ef_list": alg["eps_or_ef_list"],
                    "rerank_size_factors": alg["rerank_size_factors"],
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
            "alg_name": "deg",
            "constructor": "DEG",
            "k": 30,
            "optimization_target": "LowLID"
            if ("euclidean" in dataset_key.lower() or "agnews" in dataset_key.lower())
            else "HighLID",
            "prune_non_rng": False,
            "anns_k": 100,
            "eps_or_ef_list": [0.0, 0.05, 0.1, 0.2, 0.3],
            "rerank_size_factors": [1.0],
        }
    )
