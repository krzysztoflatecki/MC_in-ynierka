import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


NODE_STYLES = {
    "Spawn":           {"color": "#00E676", "symbol": "star",             "size": 15, "label": "Spawn"},
    "Village":         {"color": "#FFD54F", "symbol": "square",           "size": 10, "label": "Wioska"},
    "Shipwreck":       {"color": "#8D6E63", "symbol": "triangle-up",      "size": 11, "label": "Wrak statku"},
    "Treasure":        {"color": "#FFEB3B", "symbol": "diamond",          "size": 11, "label": "Skarb"},
    "Lava_Pool":       {"color": "#FF5722", "symbol": "circle",           "size": 10, "label": "Jeziorko lawy"},
    "Underwater_Lava": {"color": "#FF7043", "symbol": "circle-open-dot",  "size": 10, "label": "Lawa podwodna"},
    "Stronghold":      {"color": "#AB47BC", "symbol": "hexagon",          "size": 14, "label": "Twierdza (Stronghold)"},
    "Bastion":         {"color": "#FFB300", "symbol": "pentagon",         "size": 13, "label": "Bastion"},
    "Fortress":        {"color": "#EF5350", "symbol": "cross",            "size": 13, "label": "Forteca Netheru"},
}


def _extract_portals_and_split_ow(
    best_run: Dict
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Rozdziela `overworld_waypoints` na etap przed Netherem (entry) oraz po powrocie (exit)
    i wyznacza dokładne współrzędne portali wejściowego i wyjściowego w obu wymiarach.
    """
    seq = best_run["chosen_sequence"]
    ow_pts = best_run["overworld_waypoints"]

    entry_ow = entry_net = exit_net = exit_ow = None

    for i in range(len(seq) - 1):
        u, v = seq[i], seq[i + 1]
        if u["dimension"] == "Overworld" and v["dimension"] == "Nether":
            entry_ow = (int(u["x"]), int(u["z"]))
            entry_net = (int(u["x"]) // 8, int(u["z"]) // 8)
        elif u["dimension"] == "Nether" and v["dimension"] == "Overworld":
            exit_net = (int(u["x"]), int(u["z"]))
            exit_ow = (int(u["x"]) * 8, int(u["z"]) * 8)

    if not ow_pts:
        return [], [], entry_ow, entry_net, exit_net, exit_ow

    split_idx = None
    if entry_ow is not None and exit_ow is not None:
        for i in range(len(ow_pts) - 1):
            if tuple(ow_pts[i]) == entry_ow and tuple(ow_pts[i + 1]) == exit_ow:
                split_idx = i + 1
                break

    if split_idx is not None:
        ow_entry_pts = ow_pts[:split_idx]
        ow_exit_pts = ow_pts[split_idx:]
    else:
        ow_entry_pts = ow_pts
        ow_exit_pts = []

    return ow_entry_pts, ow_exit_pts, entry_ow, entry_net, exit_net, exit_ow


def _build_biome_heatmap(
    opt,
    xs: List[float],
    zs: List[float],
    grid_resolution: int = 85,
    padding: int = 160
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Próbkuje siatkę biomów i ich kosztów dla tła wykresu."""
    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_z, max_z = min(zs) - padding, max(zs) + padding

    x_grid = np.linspace(min_x, max_x, grid_resolution)
    z_grid = np.linspace(min_z, max_z, grid_resolution)

    cost_matrix = np.zeros((grid_resolution, grid_resolution), dtype=float)
    name_matrix = np.empty((grid_resolution, grid_resolution), dtype=object)

    for r, z_val in enumerate(z_grid):
        iz = int(round(z_val))
        for c, x_val in enumerate(x_grid):
            ix = int(round(x_val))
            b_name = opt.biome_sampler(ix, iz)
            cost_matrix[r, c] = opt.cost_map.get(b_name, opt.fallback_cost)
            name_matrix[r, c] = b_name

    return x_grid, z_grid, cost_matrix, name_matrix


def create_interactive_map(
    seed: int,
    master_df: pd.DataFrame,
    best_run: Dict,
    ow_opt=None,
    net_opt=None,
    output_html: Optional[str] = None,
    show_biome_heatmap: bool = True
) -> go.Figure:
    """
    Tworzy interaktywną mapę 2D (Overworld + Nether) z wyznaczoną trasą A* i strukturami.
    Jeśli podano `output_html`, zapisuje gotową mapę jako samodzielny plik HTML.
    """
    ow_entry_pts, ow_exit_pts, entry_ow, entry_net, exit_net, exit_ow = _extract_portals_and_split_ow(best_run)
    net_pts = best_run["nether_waypoints"]

    scenario_str = " → ".join(best_run["scenario"])
    total_time = best_run["total_time_sec"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Overworld (skala 1:1)</b>",
            "<b>Nether (skala 1:8)</b>"
        ),
        horizontal_spacing=0.07
    )

    ow_df = master_df[master_df["dimension"] == "Overworld"]
    net_df = master_df[master_df["dimension"] == "Nether"]

    # 1. TŁO: HEATMAPA KOSZTÓW BIOMÓW
    if show_biome_heatmap and ow_opt is not None:
        all_ow_x = list(ow_df["x"]) + [p[0] for p in best_run["overworld_waypoints"]]
        all_ow_z = list(ow_df["z"]) + [p[1] for p in best_run["overworld_waypoints"]]
        if all_ow_x:
            xg, zg, cm, nm = _build_biome_heatmap(ow_opt, all_ow_x, all_ow_z, grid_resolution=90, padding=200)
            fig.add_trace(
                go.Heatmap(
                    x=xg, y=zg, z=cm, customdata=nm,
                    colorscale="YlGnBu_r",
                    zmin=3.0, zmax=7.5,
                    opacity=0.45,
                    showscale=False,
                    name="Koszt terenu (OW)",
                    hovertemplate="<b>Overworld</b><br>X: %{x:.0f}, Z: %{y:.0f}<br>Biom: <b>%{customdata}</b><br>Koszt: %{z:.2f} s/chunk<extra></extra>"
                ),
                row=1, col=1
            )

    if show_biome_heatmap and net_opt is not None:
        all_net_x = list(net_df["x"]) + [p[0] for p in net_pts] + ([entry_net[0]] if entry_net else [])
        all_net_z = list(net_df["z"]) + [p[1] for p in net_pts] + ([entry_net[1]] if entry_net else [])
        if all_net_x:
            xg, zg, cm, nm = _build_biome_heatmap(net_opt, all_net_x, all_net_z, grid_resolution=75, padding=80)
            fig.add_trace(
                go.Heatmap(
                    x=xg, y=zg, z=cm, customdata=nm,
                    colorscale="Hot_r",
                    zmin=4.0, zmax=9.2,
                    opacity=0.45,
                    colorbar=dict(title="Koszt<br>[s/chunk]", x=1.02, len=0.45, y=0.22),
                    name="Koszt terenu (Nether)",
                    hovertemplate="<b>Nether</b><br>X: %{x:.0f}, Z: %{y:.0f}<br>Biom: <b>%{customdata}</b><br>Koszt: %{z:.2f} s/chunk<extra></extra>"
                ),
                row=1, col=2
            )

    # 2. WSZYSTKIE WĘZŁY GRAFU (TŁO STRUKTUR)
    for dim_name, dim_df, col_idx in [("Overworld", ow_df, 1), ("Nether", net_df, 2)]:
        for ntype, group in dim_df.groupby("node_type"):
            style = NODE_STYLES.get(ntype, {"color": "#B0BEC5", "symbol": "circle", "size": 9, "label": ntype})
            fig.add_trace(
                go.Scatter(
                    x=group["x"],
                    y=group["z"],
                    mode="markers",
                    name=style["label"],
                    legendgroup=ntype,
                    marker=dict(
                        color=style["color"],
                        symbol=style["symbol"],
                        size=style["size"],
                        line=dict(width=1.2, color="#111111")
                    ),
                    customdata=np.stack([group["node_id"], group["weight"]], axis=-1),
                    hovertemplate=(
                        f"<b>{style['label']} ({dim_name})</b><br>"
                        "ID węzła: %{customdata[0]}<br>"
                        "Współrzędne: (%{x}, %{y})<br>"
                        "Waga/Rozmiar: %{customdata[1]}<extra></extra>"
                    )
                ),
                row=1, col=col_idx
            )

    # 3. WYZNACZONE TRAJEKTORIE A*
    if ow_entry_pts:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in ow_entry_pts],
                y=[p[1] for p in ow_entry_pts],
                mode="lines",
                name="Trasa A*: Overworld (Etap 1)",
                line=dict(color="#00E5FF", width=3.5),
                hovertemplate="Etap 1 (Overworld)<br>X: %{x}, Z: %{y}<extra></extra>"
            ),
            row=1, col=1
        )

    if net_pts:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in net_pts],
                y=[p[1] for p in net_pts],
                mode="lines",
                name="Trasa A*: Nether (Etap 2)",
                line=dict(color="#FFEA00", width=3.5),
                hovertemplate="Etap 2 (Nether)<br>X: %{x}, Z: %{y}<extra></extra>"
            ),
            row=1, col=2
        )

    if ow_exit_pts:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in ow_exit_pts],
                y=[p[1] for p in ow_exit_pts],
                mode="lines",
                name="Trasa A*: Overworld → Twierdza (Etap 3)",
                line=dict(color="#76FF03", width=3.5),
                hovertemplate="Etap 3 (Powrót do Twierdzy)<br>X: %{x}, Z: %{y}<extra></extra>"
            ),
            row=1, col=1
        )

    # 4. ZNACZNIKI PORTALI (PRZEJŚCIA 1:8 ORAZ 8:1)
    if entry_ow and entry_net:
        fig.add_trace(
            go.Scatter(
                x=[entry_ow[0]], y=[entry_ow[1]],
                mode="markers",
                name="Portal wejściowy (1:8)",
                legendgroup="portal_in",
                marker=dict(color="#D500F9", symbol="diamond-tall", size=15, line=dict(width=2, color="white")),
                hovertemplate=f"<b>Portal Wejściowy (OW → Nether)</b><br>OW: {entry_ow}<br>Nether: {entry_net}<extra></extra>"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[entry_net[0]], y=[entry_net[1]],
                mode="markers",
                name="Portal wejściowy (1:8)",
                legendgroup="portal_in",
                showlegend=False,
                marker=dict(color="#D500F9", symbol="diamond-tall", size=15, line=dict(width=2, color="white")),
                hovertemplate=f"<b>Wyjście z portalu w Netherze</b><br>Nether: {entry_net}<br>Z OW: {entry_ow}<extra></extra>"
            ),
            row=1, col=2
        )

    if exit_net and exit_ow:
        fig.add_trace(
            go.Scatter(
                x=[exit_net[0]], y=[exit_net[1]],
                mode="markers",
                name="Portal wyjściowy (8:1)",
                legendgroup="portal_out",
                marker=dict(color="#00B0FF", symbol="diamond-wide", size=15, line=dict(width=2, color="white")),
                hovertemplate=f"<b>Portal Wyjściowy (Nether → OW)</b><br>Nether: {exit_net}<br>OW: {exit_ow}<extra></extra>"
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[exit_ow[0]], y=[exit_ow[1]],
                mode="markers",
                name="Portal wyjściowy (8:1)",
                legendgroup="portal_out",
                showlegend=False,
                marker=dict(color="#00B0FF", symbol="diamond-wide", size=15, line=dict(width=2, color="white")),
                hovertemplate=f"<b>Powrót do Overworldu</b><br>OW: {exit_ow}<br>Z Netheru: {exit_net}<extra></extra>"
            ),
            row=1, col=1
        )

    # 5. NUMEROWANE ETYKIETY WYBRANYCH WĘZŁÓW (KOLEJNOŚĆ PRZEJŚCIA)
    for step_idx, node in enumerate(best_run["chosen_sequence"], start=1):
        col_idx = 1 if node["dimension"] == "Overworld" else 2
        fig.add_annotation(
            x=node["x"],
            y=node["z"],
            text=f"<b>{step_idx}. {node['node_type']}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor="white",
            ax=0,
            ay=-28,
            font=dict(size=11, color="white"),
            bgcolor="rgba(20, 20, 28, 0.85)",
            bordercolor="#00E5FF" if col_idx == 1 else "#FFEA00",
            borderwidth=1,
            row=1, col=col_idx
        )

    # 6. UKŁAD I ORIENTACJA MINECRAFTA (OŚ Z ODWRÓCONA, PROPORCJE 1:1)
    fig.update_layout(
        template="plotly_dark",
        height=760,
        title=dict(
            text=(
                f"<b>Optymalizacja trasy A* — Seed: {seed}</b><br>"
                f"<sup>Scenariusz: {scenario_str} | Szacowany czas całkowity: <b>{total_time:.2f} s</b></sup>"
            ),
            x=0.5,
            xanchor="center"
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(20, 20, 28, 0.7)"
        ),
        margin=dict(l=50, r=210, t=95, b=50),
        hovermode="closest"
    )

    fig.update_xaxes(title_text="Współrzędna X (Zachód ← → Wschód)", row=1, col=1, showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Współrzędna Z (Północ ↑ ↓ Południe)", row=1, col=1, autorange="reversed", scaleanchor="x", scaleratio=1, showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    fig.update_xaxes(title_text="Współrzędna X — Nether (Zachód ← → Wschód)", row=1, col=2, showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Współrzędna Z — Nether", row=1, col=2, autorange="reversed", scaleanchor="x2", scaleratio=1, showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    if output_html:
        os.makedirs(os.path.dirname(os.path.abspath(output_html)), exist_ok=True)
        fig.write_html(output_html, include_plotlyjs="cdn")
        print(f"Zapisano interaktywną mapę: {output_html}")

    return fig
