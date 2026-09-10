import argparse
import colorsys
import datetime
import json
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
import webbrowser

from dataset import (
    VIBE_DATASETS,
    get_default_cache_dir,
    resolve_dataset_key,
)


def export_interactive_html(
    dataset_name: str,
    instruction_set: str,
    search_k: int,
    results_series: list[dict],
    html_path: Path,
):
    """
    Exports a standalone, responsive, interactive Plotly HTML chart.
    Features:
      - Custom grouped legend (Opt Targets, Pruning, K levels, Rerank factors)
      - Interactive toggling of curves via legend groups & items
      - Rich hover tooltips
      - Smooth zooming, box zoom, panning, reset
    """
    if not results_series:
        print("No benchmark series data found.")
        return

    OPT_HUES = {"lowlid": 0.60, "streamingdata": 0.33, "streaming": 0.33, "highlid": 0.02}

    all_opt_targets = []
    for entry in results_series:
        t = str(entry.get("opt_target", "")).lower().strip()
        if t and t not in all_opt_targets:
            all_opt_targets.append(t)

    all_k_values = sorted(list({entry.get("k", 30) for entry in results_series}))
    min_k = min(all_k_values) if all_k_values else 16
    max_k = max(all_k_values) if all_k_values else 48

    has_pruned = any(bool(entry.get("prune_non_rng", False)) for entry in results_series)
    has_unpruned = any(not bool(entry.get("prune_non_rng", False)) for entry in results_series)
    all_rerank_factors = sorted(list({round(float(entry.get("rerank_factor", 1.0)), 2) for entry in results_series}))

    # Dynamically assign distinct symbols to any rerank factors found in results_series
    _PLOTLY_MARKERS = ["circle", "square", "triangle-up", "diamond", "star", "cross", "x", "hexagon"]
    _UNICODE_MARKERS = ["●", "■", "▲", "◆", "★", "✚", "✖", "⬢"]
    rerank_symbols = {}
    rerank_unicode = {}
    for idx, rf in enumerate(all_rerank_factors):
        rerank_symbols[rf] = _PLOTLY_MARKERS[idx % len(_PLOTLY_MARKERS)]
        rerank_unicode[rf] = _UNICODE_MARKERS[idx % len(_UNICODE_MARKERS)]

    data_traces = []

    # 1. Main Data Traces (shown in plot)
    for entry in results_series:
        recalls = entry["recalls"]
        qps = entry["qps"]
        if not (recalls and qps):
            continue

        opt_target = str(entry.get("opt_target", "")).lower().strip()
        k = entry.get("k", 30)
        is_pruned = bool(entry.get("prune_non_rng", False))
        r_factor = round(float(entry.get("rerank_factor", 1.0)), 2)

        hue = OPT_HUES.get(opt_target, 0.60)
        k_ratio = (k - min_k) / (max_k - min_k) if max_k > min_k else 0.5
        lightness = 0.68 - 0.34 * k_ratio
        saturation = 0.45 + 0.55 * k_ratio
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

        dash = "dash" if is_pruned else "solid"
        symbol = rerank_symbols.get(r_factor, "circle")
        display_opt = (
            "StreamingData"
            if "stream" in opt_target
            else ("LowLID" if "low" in opt_target else ("HighLID" if "high" in opt_target else opt_target.capitalize()))
        )
        prune_txt = "MRNG Pruned" if is_pruned else "Unpruned"
        query_dt = str(entry.get("query_dtype", "")).upper()
        rerank_txt = f"{r_factor:g}x (FP16)" if r_factor > 1.0 else "None (1.0x)"

        query_info = f"Query Dtype: {query_dt}<br>" if query_dt else ""
        search_params = entry.get("search_params", [])
        hover_texts = []
        for idx, (x, y) in enumerate(zip(recalls, qps)):
            param_str = f"Param: {search_params[idx]}<br>" if idx < len(search_params) and search_params[idx] else ""
            hover_texts.append(
                f"<b>{display_opt} (K={k})</b><br>"
                f"{query_info}"
                f"Status: {prune_txt}<br>"
                f"Rerank: {rerank_txt}<br>"
                f"{param_str}"
                f"<b>Recall@{search_k}:</b> {x:.5f}<br>"
                f"<b>QPS:</b> {y:,.1f}"
            )

        trace_dtype_str = f", {query_dt}" if query_dt else ""
        trace_name = f"{display_opt} K={k} ({prune_txt}, {rerank_txt}{trace_dtype_str})"
        legend_group = f"{display_opt}"

        data_traces.append(
            {
                "x": recalls,
                "y": qps,
                "mode": "lines+markers",
                "name": trace_name,
                "legendgroup": legend_group,
                "hovertext": hover_texts,
                "hoverinfo": "text",
                "line": {"color": hex_color, "dash": dash, "width": 2},
                "marker": {"symbol": symbol, "size": 7, "color": hex_color},
                "showlegend": True,
                "_meta": {
                    "opt_target": opt_target,
                    "k": k,
                    "is_pruned": is_pruned,
                    "rerank_factor": r_factor,
                },
            }
        )

    # Compute initial global data bounds to fix axes and prevent jumpy rescaling
    all_x = [x for entry in results_series for x in entry["recalls"] if x is not None]
    all_y = [y for entry in results_series for y in entry["qps"] if y is not None]
    min_x = min(all_x) if all_x else 0.5
    max_x = max(all_x) if all_x else 1.0
    max_y = max(all_y) if all_y else 10000.0

    pad_x = (max_x - min_x) * 0.04
    x_range = [max(0.0, min_x - pad_x), min(1.005, max_x + pad_x * 0.5)]
    y_range = [0, max_y * 1.05]

    layout = {
        "title": {
            "text": f"DEG ANNS Benchmark: {dataset_name} ({instruction_set})",
            "font": {"size": 16, "family": "Inter, -apple-system, sans-serif", "color": "#f1f5f9"},
            "x": 0.02,
            "y": 0.98,
        },
        "xaxis": {
            "title": {"text": f"Recall@{search_k}", "font": {"size": 13, "color": "#cbd5e1"}},
            "gridcolor": "#334155",
            "zerolinecolor": "#475569",
            "tickfont": {"color": "#94a3b8"},
            "range": x_range,
            "autorange": False,
        },
        "yaxis": {
            "title": {"text": "Queries Per Second (QPS)", "font": {"size": 13, "color": "#cbd5e1"}},
            "gridcolor": "#334155",
            "zerolinecolor": "#475569",
            "tickfont": {"color": "#94a3b8"},
            "range": y_range,
            "autorange": False,
        },
        "uirevision": "dataset_lock",
        "dragmode": "pan",
        "hovermode": "closest",
        "plot_bgcolor": "#1e293b",
        "paper_bgcolor": "#1e293b",
        "font": {"color": "#e2e8f0"},
        "margin": {"l": 65, "r": 25, "t": 45, "b": 55},
    }

    # 2. Build dynamic HTML sections based only on available data
    # Section A: Optimization Targets
    opt_html_rows = ""
    opt_js_obj = {}
    OPT_CONFIG = {
        "lowlid": ("#38bdf8", "LowLID (Blue)"),
        "streamingdata": ("#4ade80", "StreamingData (Green)"),
        "streaming": ("#4ade80", "StreamingData (Green)"),
        "highlid": ("#f87171", "HighLID (Red)"),
    }
    for opt in all_opt_targets:
        color, label = OPT_CONFIG.get(opt, ("#a855f7", opt.capitalize()))
        opt_html_rows += f"""
                <div class="legend-row" id="opt-{opt}" onclick="toggleFilter('opt', '{opt}')">
                    <span class="legend-label">
                        <span class="color-dot" style="background: {color};"></span>
                        {label}
                    </span>
                    <span style="font-size: 11px; color: #64748b;">All K</span>
                </div>"""
        opt_js_obj[opt] = True

    # Section B: Pruning Status (only if mixed)
    prune_html_section = ""
    prune_js_obj = {"false": True, "true": True}
    if has_pruned and has_unpruned:
        prune_html_section = """
            <div>
                <div class="section-title">Pruning Status</div>
                <div class="legend-row" id="prune-false" onclick="toggleFilter('prune', false)">
                    <span class="legend-label">
                        <span class="line-sample" style="border-top: 2.5px solid #e2e8f0;"></span>
                        Unpruned Graph
                    </span>
                </div>
                <div class="legend-row" id="prune-true" onclick="toggleFilter('prune', true)">
                    <span class="legend-label">
                        <span class="line-sample" style="border-top: 2.5px dashed #e2e8f0;"></span>
                        MRNG Pruned Graph
                    </span>
                </div>
            </div>"""

    # Section C: Graph Degrees K
    k_html_rows = ""
    k_js_obj = {}
    for val_k in all_k_values:
        k_ratio = (val_k - min_k) / (max_k - min_k) if max_k > min_k else 0.5
        depth_lbl = "Light" if k_ratio < 0.3 else ("Deep" if k_ratio > 0.7 else "Medium")
        k_html_rows += f"""
                    <div class="legend-row" id="k-{val_k}" onclick="toggleFilter('k', {val_k})">
                        <span class="legend-label">K = {val_k}</span>
                        <span style="font-size: 11px; color: #94a3b8;">{depth_lbl}</span>
                    </div>"""
        k_js_obj[val_k] = True

    # Section D: Rerank Factors (only if multiple rerank factors exist or factor > 1.0)
    rerank_html_section = ""
    rerank_js_obj = {f"{rf:g}": True for rf in all_rerank_factors}
    has_meaningful_rerank = len(all_rerank_factors) > 1 or (
        len(all_rerank_factors) == 1 and all_rerank_factors[0] > 1.0
    )
    if has_meaningful_rerank:
        rerank_rows = ""
        for rf in all_rerank_factors:
            sym_char = rerank_unicode.get(rf, "●")
            lbl = f"{sym_char} {rf:g}x" + (" (None)" if rf == 1.0 else "")
            rf_key = f"{rf:g}"
            rerank_rows += f"""
                    <div class="legend-row" id="rerank-{rf_key}" onclick="toggleFilter('rerank', '{rf_key}')">
                        <span class="legend-label">{lbl}</span>
                    </div>"""
        rerank_html_section = f"""
            <div>
                <div class="section-title">Rerank Factor &bull; Marker</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                    {rerank_rows}
                </div>
            </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>DEG Benchmark: {dataset_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 14px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a;
            color: #f8fafc;
        }}
        .layout-grid {{
            max-width: 1650px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 16px;
            height: calc(100vh - 28px);
        }}
        .plot-card {{
            background: #1e293b;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.4);
            border: 1px solid #334155;
            display: flex;
            flex-direction: column;
        }}
        #plot {{
            width: 100%;
            flex: 1;
            min-height: 500px;
        }}
        .legend-card {{
            background: #1e293b;
            border-radius: 10px;
            padding: 16px;
            border: 1px solid #334155;
            box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            gap: 14px;
            overflow-y: auto;
        }}
        .legend-title {{
            font-size: 16px;
            font-weight: 600;
            color: #f8fafc;
            border-bottom: 1px solid #334155;
            padding-bottom: 8px;
            margin: 0;
        }}
        .section-title {{
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 10px;
        }}
        .legend-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 10px;
            border-radius: 6px;
            background: #0f172a;
            margin-bottom: 6px;
            cursor: pointer;
            transition: background 0.15s, opacity 0.15s;
            user-select: none;
        }}
        .legend-row:hover {{
            background: #273549;
        }}
        .legend-row.inactive {{
            opacity: 0.35;
            text-decoration: line-through;
        }}
        .legend-label {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13.5px;
        }}
        .color-dot {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            display: inline-block;
        }}
        .line-sample {{
            width: 24px;
            height: 0px;
            display: inline-block;
        }}
        .btn-row {{
            display: flex;
            gap: 8px;
            margin-top: 4px;
        }}
        .ctrl-btn {{
            flex: 1;
            background: #334155;
            color: #cbd5e1;
            border: none;
            padding: 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.15s;
        }}
        .ctrl-btn:hover {{
            background: #475569;
            color: #fff;
        }}
    </style>
</head>
<body>
    <div class="layout-grid">
        <div class="plot-card">
            <div id="plot"></div>
        </div>

        <div class="legend-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 class="legend-title" style="border: none; padding: 0;">Interactive Legend</h3>
                <span style="font-size: 11px; color: #64748b;">Click to Filter</span>
            </div>

            <!-- Optimization Targets -->
            <div>
                <div class="section-title">Optimization Target</div>
                {opt_html_rows}
            </div>

            <!-- Pruning Line Styles -->
            {prune_html_section}

            <!-- Graph Degrees K -->
            <div>
                <div class="section-title">Graph Degree (K) &bull; Color Depth</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                    {k_html_rows}
                </div>
            </div>

            <!-- Rerank Size Factors -->
            {rerank_html_section}

            <div class="btn-row">
                <button class="ctrl-btn" onclick="showAllTraces()">Show All</button>
                <button class="ctrl-btn" onclick="hideAllTraces()">Hide All</button>
            </div>
        </div>
    </div>

    <script>
        const rawTraces = {json.dumps(data_traces)};
        const layout = {json.dumps(layout)};
        layout.showlegend = false;

        const config = {{
            responsive: true,
            scrollZoom: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        }};

        // Dynamic active filters matching available data
        const activeFilters = {{
            opt: {json.dumps(opt_js_obj)},
            prune: {json.dumps(prune_js_obj)},
            k: {json.dumps(k_js_obj)},
            rerank: {json.dumps(rerank_js_obj)}
        }};

        Plotly.newPlot('plot', rawTraces, layout, config);

        function updateTraceVisibility() {{
            const update = {{ visible: [] }};
            rawTraces.forEach(t => {{
                const meta = t._meta;
                const optVisible = activeFilters.opt[meta.opt_target] !== false;
                const pruneVisible = activeFilters.prune[meta.is_pruned.toString()] !== false;
                const kVisible = activeFilters.k[meta.k.toString()] !== false;
                const rerankKey = String(parseFloat(Number(meta.rerank_factor).toFixed(4)));
                const rerankVisible = activeFilters.rerank[rerankKey] !== false;

                const isVisible = optVisible && pruneVisible && kVisible && rerankVisible;
                update.visible.push(isVisible ? true : false);
            }});
            Plotly.restyle('plot', update);
        }}

        function toggleFilter(category, val) {{
            const key = (category === 'rerank') ? String(parseFloat(Number(val).toFixed(4))) : val.toString();
            activeFilters[category][key] = !activeFilters[category][key];
            const elemId = category + '-' + key;
            const elem = document.getElementById(elemId);
            if (elem) {{
                elem.classList.toggle('inactive', !activeFilters[category][key]);
            }}
            updateTraceVisibility();
        }}

        function showAllTraces() {{
            for (let cat in activeFilters) {{
                for (let k in activeFilters[cat]) {{
                    activeFilters[cat][k] = true;
                    const elem = document.getElementById(cat + '-' + k);
                    if (elem) elem.classList.remove('inactive');
                }}
            }}
            updateTraceVisibility();
        }}

        function hideAllTraces() {{
            for (let cat in activeFilters) {{
                for (let k in activeFilters[cat]) {{
                    activeFilters[cat][k] = false;
                    const elem = document.getElementById(cat + '-' + k);
                    if (elem) elem.classList.add('inactive');
                }}
            }}
            updateTraceVisibility();
        }}
    </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Plotly Interactive HTML plot saved to: {html_path.resolve()}")


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
    current_params = []

    re_dataset = re.compile(r"Selected VIBE Dataset:\s*([^\r\n]+)", re.IGNORECASE)
    re_space = re.compile(r"Vector Space Type:\s*FloatSpace\s*\(([^)]+)\)", re.IGNORECASE)
    re_dtypes = re.compile(r"Query Dtype:\s*([a-zA-Z0-9_-]+)", re.IGNORECASE)
    re_header_old = re.compile(
        r"---\s*\[\d+/\d+\]\s*Fitting\s*/\s*Loading Index:\s*K=(\d+)(?:,\s*ExtendK=\d+)?(?:,\s*(?:ExtendEps|Eps)=([\d.]+))?(?:,\s*Opt=([a-zA-Z0-9_-]+))?(?:,\s*Threads=\d+)?(?:,\s*(?:Dtype|QueryType)=([a-zA-Z0-9_-]+))?",
        re.IGNORECASE,
    )
    re_index_config = re.compile(
        r"Index Config:\s*K=(\d+),\s*Opt=([a-zA-Z0-9_-]+),\s*Prune=(True|False),\s*QueryType=([a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    )
    re_build_graph = re.compile(
        r"Building DEG graph\s*\(K=(\d+),\s*Opt=([a-zA-Z0-9_-]+)",
        re.IGNORECASE,
    )
    re_cached_graph = re.compile(
        r"Loading cached DEG graph from .*[\\/](\d+)D_[^_]+_K(\d+)_.*?(LowLID|StreamingData|Quality|HighLID)",
        re.IGNORECASE,
    )
    re_prune = re.compile(r"Pruning non-RNG edges", re.IGNORECASE)
    re_eval = re.compile(
        r"Evaluating top-(\d+)\s*search(?:\s*\((?:rerank_factor=([\d.]+))?(?:,\s*fetch_k=\d+)?(?:,\s*[^)]*)?\))?",
        re.IGNORECASE,
    )
    re_eval_old = re.compile(r"Evaluating top-(\d+)\s*search for (?:eps|ef|eps_or_ef):", re.IGNORECASE)
    re_line = re.compile(
        r"(?:eps\s+([\d.]+)|ef\s+(\d+)|(?:eps_or_ef|param)\s+([\d.]+))\s+recall@\d+:\s+([\d.]+)\s+[\d.]+\s+us/query\s+([\d.]+)\s+QPS",
        re.IGNORECASE,
    )

    def save_current_series():
        nonlocal current_config, current_rerank_factor, current_recalls, current_qps, current_params
        if current_config is not None and current_recalls and current_qps:
            results_series.append(
                {
                    "opt_target": current_config["opt_target"],
                    "k": current_config["k"],
                    "query_dtype": current_config.get("query_dtype", file_query_dtype),
                    "prune_non_rng": current_config["is_pruned"],
                    "rerank_factor": current_rerank_factor,
                    "recalls": list(current_recalls),
                    "qps": list(current_qps),
                    "search_params": list(current_params),
                }
            )
        current_recalls = []
        current_qps = []
        current_params = []

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

            m = re_index_config.search(line_str)
            if m:
                save_current_series()
                k = int(m.group(1))
                opt_target = m.group(2).strip()
                is_pruned = (m.group(3).lower() == "true")
                q_dtype = m.group(4).strip().lower()
                current_config = {
                    "k": k,
                    "opt_target": opt_target,
                    "is_pruned": is_pruned,
                    "query_dtype": q_dtype,
                }
                current_rerank_factor = 1.0
                continue

            m = re_header_old.search(line_str)
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
                    "query_dtype": file_query_dtype,
                }
                current_rerank_factor = 1.0
                continue

            m = re_build_graph.search(line_str)
            if m and (current_config is None or current_config.get("k") != int(m.group(1))):
                save_current_series()
                current_config = {
                    "k": int(m.group(1)),
                    "opt_target": m.group(2).strip(),
                    "is_pruned": False,
                    "query_dtype": file_query_dtype,
                }
                current_rerank_factor = 1.0
                continue

            m = re_cached_graph.search(line_str)
            if m and (current_config is None or current_config.get("k") != int(m.group(2))):
                save_current_series()
                current_config = {
                    "k": int(m.group(2)),
                    "opt_target": m.group(3).strip(),
                    "is_pruned": False,
                    "query_dtype": file_query_dtype,
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
                if m.group(1) is not None:
                    param_val = f"eps={float(m.group(1)):g}"
                elif m.group(2) is not None:
                    param_val = f"ef={int(m.group(2))}"
                else:
                    val = float(m.group(3))
                    param_val = f"ef={int(round(val))}" if val > 1.0 else f"eps={val:g}"
                recall = float(m.group(4))
                qps = float(m.group(5))
                current_params.append(param_val)

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
        webbrowser.open(html_path.as_uri())

    return html_path


def select_log_gui(cache_dir: Path) -> None:
    """
    Shows a clean dark-themed Tkinter Explorer window with the cache folder structure.
    Stays open so multiple benchmark logs can be viewed and compared seamlessly.
    """
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
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
