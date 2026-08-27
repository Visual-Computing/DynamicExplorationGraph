import argparse
from pathlib import Path
import re

from dataset_utils import (
    VIBE_DATASETS,
    get_default_cache_dir,
    resolve_dataset_key,
)
from plot_utils import export_interactive_html


def parse_benchmark_log(log_path: Path) -> tuple[str, str, int, list[dict]]:
    """
    Parses a VIBE benchmark log file and extracts all curve data series.
    Returns: (dataset_name, instruction_set, search_k, results_series)
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    dataset_name = ""
    instruction_set = "AVX2"
    search_k = 100
    results_series = []

    # Infer fallback query dtype from folder name (e.g. deg-int8, deg-fp32)
    path_str = str(log_path).lower()
    inferred_query_dtype = (
        "int8" if "int8" in path_str else ("float16" if "fp16" in path_str and "rerank" not in path_str else "float32")
    )
    file_query_dtype = inferred_query_dtype

    current_config = None
    current_rerank_factor = 1.0
    current_recalls = []
    current_qps = []

    re_dataset = re.compile(r"Selected VIBE Dataset:\s*([^\r\n]+)", re.IGNORECASE)
    re_space = re.compile(r"Vector Space Type:\s*FloatSpace\s*\(([^)]+)\)", re.IGNORECASE)
    re_dtypes = re.compile(r"Query Dtype:\s*([a-zA-Z0-9_-]+)", re.IGNORECASE)
    re_header = re.compile(
        r"---\s*\[\d+/\d+\]\s*Fitting\s*/\s*Loading Index:\s*K=(\d+)(?:,\s*ExtendK=\d+)?(?:,\s*(?:ExtendEps|Eps)=([\d.]+))?(?:,\s*Opt=([a-zA-Z0-9_-]+))?(?:,\s*Threads=\d+)?(?:,\s*(?:Dtype|QueryType)=([a-zA-Z0-9_-]+))?",
        re.IGNORECASE,
    )
    re_prune = re.compile(r"Pruning non-RNG edges", re.IGNORECASE)
    re_eval = re.compile(
        r"Evaluating top-(\d+)\s*search(?:\s*\((?:rerank_factor=([\d.]+))?(?:,\s*fetch_k=\d+)?(?:,\s*[^)]*)?\))?",
        re.IGNORECASE,
    )
    re_eval_old = re.compile(r"Evaluating top-(\d+)\s*search for eps:", re.IGNORECASE)
    re_line = re.compile(r"eps\s+([\d.]+)\s+recall@\d+:\s+([\d.]+)\s+[\d.]+\s+us/query\s+([\d.]+)\s+QPS", re.IGNORECASE)

    def save_current_series():
        nonlocal current_config, current_rerank_factor, current_recalls, current_qps
        if current_config is not None and current_recalls and current_qps:
            results_series.append(
                {
                    "opt_target": current_config["opt_target"],
                    "k": current_config["k"],
                    "query_dtype": file_query_dtype,
                    "prune_non_rng": current_config["is_pruned"],
                    "rerank_factor": current_rerank_factor,
                    "recalls": list(current_recalls),
                    "qps": list(current_qps),
                }
            )
        current_recalls = []
        current_qps = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()

            m = re_dataset.search(line_str)
            if m:
                dataset_name = m.group(1).strip()
                continue

            m = re_space.search(line_str)
            if m:
                instruction_set = m.group(1).strip()
                continue

            m = re_dtypes.search(line_str)
            if m:
                file_query_dtype = m.group(1).strip().lower()
                continue

            m = re_header.search(line_str)
            if m:
                save_current_series()
                k = int(m.group(1))
                build_eps = float(m.group(2)) if m.group(2) is not None else 0.2
                opt_target = m.group(3).strip() if m.group(3) is not None else "Quality"
                if m.group(4):
                    file_query_dtype = m.group(4).strip().lower()
                current_config = {
                    "k": k,
                    "build_eps": build_eps,
                    "opt_target": opt_target,
                    "is_pruned": False,
                }
                current_rerank_factor = 1.0
                continue

            if re_prune.search(line_str) and current_config is not None:
                current_config["is_pruned"] = True
                continue

            m = re_eval.search(line_str)
            if m:
                save_current_series()
                search_k = int(m.group(1))
                current_rerank_factor = float(m.group(2)) if m.group(2) is not None else 1.0
                continue

            m = re_eval_old.search(line_str)
            if m:
                save_current_series()
                search_k = int(m.group(1))
                current_rerank_factor = 1.0
                continue

            m = re_line.search(line_str)
            if m and current_config is not None:
                recall = float(m.group(2))
                qps = float(m.group(3))
                current_recalls.append(recall)
                current_qps.append(qps)

    save_current_series()
    return dataset_name, instruction_set, search_k, results_series


def find_benchmark_logs(cache_dir: Path) -> list[Path]:
    """Finds all benchmark *.log files inside valid VIBE dataset directories."""
    if not cache_dir.exists():
        return []

    logs = []
    # Only scan subdirectories corresponding to known VIBE datasets
    for d_key in VIBE_DATASETS.keys():
        dataset_folder = cache_dir / d_key
        if dataset_folder.is_dir():
            # Collect all .log files in dataset folder & its subfolders (deg, deg-fp32, etc.)
            for log_file in dataset_folder.rglob("*.log"):
                if log_file.is_file() and log_file.stat().st_size > 0:
                    logs.append(log_file)

    return sorted(logs)


def open_plot_for_log(log_path: Path, auto_open: bool = True) -> Path:
    """Helper to parse a log file, generate its Plotly HTML, and open it in the browser."""
    if log_path.is_dir():
        logs = list(log_path.glob("*.log"))
        if not logs:
            raise FileNotFoundError(f"No .log files found in directory: {log_path}")
        log_path = logs[0]

    dataset_key = log_path.stem.replace("_benchmark", "").replace("_anns", "")
    print(f"Reading benchmark log: {log_path.resolve()}")
    dataset_name, instruction_set, search_k, results_series = parse_benchmark_log(log_path)

    if not results_series:
        print("No benchmark series data found.")
        return log_path

    if not dataset_name:
        dataset_name = VIBE_DATASETS.get(dataset_key, {}).get("name", dataset_key)

    html_path = log_path.parent / f"{log_path.stem}_anns_benchmark.html"
    export_interactive_html(
        dataset_name=dataset_name,
        instruction_set=instruction_set,
        search_k=search_k,
        results_series=results_series,
        html_path=html_path,
    )

    if auto_open:
        import webbrowser

        webbrowser.open(html_path.as_uri())

    return html_path


def select_log_gui(cache_dir: Path) -> None:
    """
    Shows a clean dark-themed Tkinter Explorer window with the cache folder structure.
    Stays open so multiple benchmark logs can be viewed and compared seamlessly.
    """
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("DEG ANNS Benchmark Log Explorer")
    root.geometry("700x520")
    root.configure(bg="#0f172a")

    # Style
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Treeview",
        background="#1e293b",
        foreground="#f8fafc",
        fieldbackground="#1e293b",
        rowheight=26,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Treeview.Heading",
        background="#334155",
        foreground="#38bdf8",
        font=("Segoe UI", 10, "bold"),
    )
    style.map("Treeview", background=[("selected", "#0284c7")], foreground=[("selected", "#ffffff")])

    # Header
    lbl_title = tk.Label(
        root,
        text="DEG Benchmark Log Explorer",
        font=("Segoe UI", 12, "bold"),
        bg="#0f172a",
        fg="#f8fafc",
        pady=6,
    )
    lbl_title.pack(fill="x")

    lbl_sub = tk.Label(
        root,
        text=f"Location: {cache_dir.resolve()}  •  (Double-click any log to view plot)",
        font=("Segoe UI", 9),
        bg="#0f172a",
        fg="#94a3b8",
        pady=2,
    )
    lbl_sub.pack(fill="x")

    # Treeview Frame
    frame = tk.Frame(root, bg="#0f172a", padx=14, pady=8)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame, columns=("size", "modified"), selectmode="browse")
    tree.heading("#0", text="Folder / Log File", anchor="w")
    tree.heading("size", text="Size", anchor="e")
    tree.heading("modified", text="Last Modified", anchor="w")
    tree.column("#0", width=380, anchor="w")
    tree.column("size", width=80, anchor="e")
    tree.column("modified", width=150, anchor="w")

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    # Status label
    status_var = tk.StringVar(value="Ready. Select a log file to view.")
    lbl_status = tk.Label(
        root,
        textvariable=status_var,
        font=("Segoe UI", 9, "italic"),
        bg="#1e293b",
        fg="#38bdf8",
        pady=4,
        relief="flat",
    )
    lbl_status.pack(fill="x", padx=14, pady=(0, 6))

    # Populate Tree
    logs = find_benchmark_logs(cache_dir)
    nodes = {}
    import datetime

    for log in logs:
        try:
            rel = log.relative_to(cache_dir)
        except ValueError:
            rel = log

        parts = rel.parts
        curr_parent = ""

        for i, part in enumerate(parts[:-1]):
            node_key = "/".join(parts[: i + 1])
            if node_key not in nodes:
                nodes[node_key] = tree.insert(curr_parent, "end", node_key, text=f"📁 {part}", open=True)
            curr_parent = nodes[node_key]

        file_key = str(log.resolve())
        stat = log.stat()
        sz_kb = f"{stat.st_size / 1024:.1f} KB"
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        tree.insert(
            curr_parent,
            "end",
            file_key,
            text=f"📄 {parts[-1]}",
            values=(sz_kb, mtime),
        )

    def trigger_open():
        sel = tree.selection()
        if not sel:
            status_var.set("Please select a log file.")
            return

        item_id = sel[0]
        p = Path(item_id)
        if p.is_file():
            try:
                status_var.set(f"Generating plot for {p.name}...")
                root.update_idletasks()
                html_p = open_plot_for_log(p, auto_open=True)
                status_var.set(f"✓ Opened in browser: {p.name}")
            except Exception as e:
                status_var.set(f"Error opening log: {e}")
        else:
            status_var.set("Selected item is a directory.")

    def on_double_click(event):
        trigger_open()

    tree.bind("<Double-1>", on_double_click)

    # Button Bar
    btn_frame = tk.Frame(root, bg="#0f172a", padx=14, pady=10)
    btn_frame.pack(fill="x")

    btn_close = tk.Button(
        btn_frame,
        text="Close",
        command=root.destroy,
        bg="#334155",
        fg="#cbd5e1",
        relief="flat",
        padx=16,
        pady=6,
        font=("Segoe UI", 9),
        cursor="hand2",
    )
    btn_close.pack(side="right", padx=6)

    btn_open = tk.Button(
        btn_frame,
        text="Open Selected Plot",
        command=trigger_open,
        bg="#0284c7",
        fg="#ffffff",
        relief="flat",
        padx=18,
        pady=6,
        font=("Segoe UI", 9, "bold"),
        cursor="hand2",
    )
    btn_open.pack(side="right")

    # Center window
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plot from existing log file.")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Dataset name (e.g. laion-clip, arxiv-nomic, yahoo-minilm, agnews-mxbai)",
    )
    parser.add_argument(
        "--log",
        "-l",
        type=Path,
        default=None,
        help="Path to specific benchmark.log file (defaults to <cache_dir>/<dataset>/deg/<dataset>_benchmark.log)",
    )
    parser.add_argument(
        "--cache-dir",
        "-c",
        type=Path,
        default=None,
        help="Base cache directory (defaults to standard cache directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output HTML file path (default: <cache_dir>/<dataset>/deg/<dataset>_anns_benchmark.html)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the interactive HTML plot in web browser",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or get_default_cache_dir()

    # If neither --dataset nor --log was provided, open GUI log selector
    if args.log is None and args.dataset is None:
        select_log_gui(cache_dir)
        return
    elif args.log is not None:
        log_path = args.log
        if log_path.is_dir():
            logs = list(log_path.glob("*.log"))
            if not logs:
                raise FileNotFoundError(f"No .log files found in directory: {log_path}")
            log_path = logs[0]
        dataset_key = log_path.stem.replace("_benchmark", "").replace("_anns", "")
    else:
        dataset_key = resolve_dataset_key(args.dataset)
        log_path = cache_dir / dataset_key / "deg" / f"{dataset_key}_benchmark.log"

    print(f"Reading benchmark log: {log_path.resolve()}")
    dataset_name, instruction_set, search_k, results_series = parse_benchmark_log(log_path)

    if not results_series:
        print("No benchmark series data found.")
        return

    if not dataset_name:
        dataset_name = VIBE_DATASETS.get(dataset_key, {}).get("name", dataset_key)

    if args.output is not None:
        html_path = args.output
    else:
        html_path = log_path.parent / f"{log_path.stem}_anns_benchmark.html"

    print(f"Loaded {len(results_series)} benchmark curves for dataset '{dataset_name}'.")
    export_interactive_html(
        dataset_name=dataset_name,
        instruction_set=instruction_set,
        search_k=search_k,
        results_series=results_series,
        html_path=html_path,
    )

    if not args.no_open:
        import webbrowser

        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
