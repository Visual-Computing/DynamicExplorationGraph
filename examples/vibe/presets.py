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

            algorithms.append(
                {
                    "name": alg_name,
                    "constructor": constructor,
                    "run_group": rg_name,
                    "args": {
                        "k": args.get("k", [30]),
                        "opt_target": args.get("opt_target", ["LowLID"]),
                        "quants": args.get("quants", []),
                    },
                    "query_args": {
                        "quants": query_args.get("quants", []),
                        "eps_or_ef": [float(v) for v in query_args.get("eps_or_ef", [])],
                        "rerank_size_factor": query_args.get("rerank_size_factor", [1.0]),
                    },
                }
            )

    return algorithms


def get_config_grid_presets(
    dataset_key: str,
    algorithm_name: str | None = None,
    config_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns the list of all index configurations (Cartesian product of build `args`)
    defined in config.yml, keeping `args` (for `__init__`) and `query_args`
    (for `set_query_arguments`) cleanly separated.
    """
    algorithms = load_vibe_config(config_path)
    is_l2 = "euclidean" in dataset_key.lower() or "agnews" in dataset_key.lower()
    default_opt = "LowLID" if is_l2 else "HighLID"

    grid = []
    for alg in algorithms:
        if algorithm_name and alg["name"] != algorithm_name:
            continue

        alg_args = alg["args"]
        raw_opt_targets = alg_args.get("opt_target", [default_opt])
        resolved_opt_targets = []
        for t in raw_opt_targets:
            target_name = default_opt if t == "" else t
            if target_name not in resolved_opt_targets:
                resolved_opt_targets.append(target_name)

        raw_args_quants = alg_args.get("quants") or []
        if not raw_args_quants:
            args_quants = [None]
        elif isinstance(raw_args_quants[0], list):
            args_quants = raw_args_quants
        else:
            args_quants = [raw_args_quants]

        for opt_target, k, quants_arg in itertools.product(resolved_opt_targets, alg_args["k"], args_quants):
            index_args: Dict[str, Any] = {
                "k": k,
                "opt_target": opt_target,
            }
            if quants_arg is not None:
                index_args["quants"] = quants_arg

            grid.append(
                {
                    "alg_name": alg["name"],
                    "constructor": alg["constructor"],
                    "args": index_args,
                    "query_args": alg["query_args"],
                    "anns_k": 100,
                }
            )
    return grid


def get_default_config_preset(dataset_key: str, config_path: Path | None = None) -> Dict[str, Any]:
    """Returns dataset-tailored defaults while reading search grid from config.yml."""
    presets = get_config_grid_presets(dataset_key, config_path=config_path)
    return (
        presets[0]
        if presets
        else {
            "alg_name": "deg",
            "constructor": "DEG",
            "args": {
                "k": 30,
                "opt_target": "LowLID"
                if ("euclidean" in dataset_key.lower() or "agnews" in dataset_key.lower())
                else "HighLID",
            },
            "query_args": {
                "quants": [],
                "eps_or_ef": [0.0, 0.05, 0.1, 0.2, 0.3],
                "rerank_size_factor": [1.0],
            },
            "anns_k": 100,
        }
    )
