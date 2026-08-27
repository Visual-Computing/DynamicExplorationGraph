import colorsys
import json
from pathlib import Path


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
    RERANK_SYMBOLS = {1.0: "circle", 1.2: "square", 1.5: "triangle-up", 2.0: "diamond"}

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
    all_rerank_factors = sorted(list({float(entry.get("rerank_factor", 1.0)) for entry in results_series}))

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
        r_factor = float(entry.get("rerank_factor", 1.0))

        hue = OPT_HUES.get(opt_target, 0.60)
        k_ratio = (k - min_k) / (max_k - min_k) if max_k > min_k else 0.5
        lightness = 0.68 - 0.34 * k_ratio
        saturation = 0.45 + 0.55 * k_ratio
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

        dash = "dash" if is_pruned else "solid"
        symbol = RERANK_SYMBOLS.get(r_factor, "circle")
        display_opt = (
            "StreamingData"
            if "stream" in opt_target
            else ("LowLID" if "low" in opt_target else ("HighLID" if "high" in opt_target else opt_target.capitalize()))
        )
        prune_txt = "MRNG Pruned" if is_pruned else "Unpruned"
        query_dt = str(entry.get("query_dtype", "")).upper()
        rerank_txt = f"{r_factor:.1f}x (FP16)" if r_factor > 1.0 else "None (1.0x)"

        query_info = f"Query Dtype: {query_dt}<br>" if query_dt else ""
        hover_texts = [
            f"<b>{display_opt} (K={k})</b><br>"
            f"{query_info}"
            f"Status: {prune_txt}<br>"
            f"Rerank: {rerank_txt}<br>"
            f"<b>Recall@{search_k}:</b> {x:.5f}<br>"
            f"<b>QPS:</b> {y:,.1f}"
            for x, y in zip(recalls, qps)
        ]

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
    rerank_js_obj = {f"{rf:.1f}": True for rf in all_rerank_factors}
    has_meaningful_rerank = len(all_rerank_factors) > 1 or (
        len(all_rerank_factors) == 1 and all_rerank_factors[0] > 1.0
    )
    if has_meaningful_rerank:
        rerank_rows = ""
        for rf in all_rerank_factors:
            sym_char = "●" if rf == 1.0 else ("■" if rf == 1.2 else ("▲" if rf == 1.5 else "◆"))
            lbl = f"{sym_char} {rf:.1f}x" + (" (None)" if rf == 1.0 else "")
            rf_key = f"{rf:.1f}"
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
                const rerankKey = parseFloat(meta.rerank_factor).toFixed(1);
                const rerankVisible = activeFilters.rerank[rerankKey] !== false;

                const isVisible = optVisible && pruneVisible && kVisible && rerankVisible;
                update.visible.push(isVisible ? true : false);
            }});
            Plotly.restyle('plot', update);
        }}

        function toggleFilter(category, val) {{
            const key = (category === 'rerank') ? parseFloat(val).toFixed(1) : val.toString();
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
