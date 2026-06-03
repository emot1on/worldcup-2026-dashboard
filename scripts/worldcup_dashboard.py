from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DATA_DIR = PROJECT_ROOT / "output" / "dashboard-data"
NORMALIZED_BUNDLE_PATH = DASHBOARD_DATA_DIR / "normalized" / "dashboard_bundle.json"
RAW_FOOTBALL_DATA_DIR = DASHBOARD_DATA_DIR / "raw" / "football_data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "worldcup-dashboard"

BG = "#f6f2ea"
PANEL = "#fffdf8"
TEXT = "#1f1b17"
MUTED = "#6f675d"
GRID = "#ddd5c7"
ACCENT = "#8b5b47"
AGE = "#2c7da0"
HEIGHT = "#cf6d3e"
CARD = "#efe7db"


def read_wrapped_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["data"]


def base_layout(title: str, subtitle: str, *, height: int = 420, legend: bool = False) -> dict:
    return dict(
        title=dict(text=title, x=0, xanchor="left"),
        annotations=[
            dict(
                text=subtitle,
                x=0,
                y=1.1,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=13, color=MUTED),
            )
        ],
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Arial, Helvetica, sans-serif", color=TEXT, size=14),
        margin=dict(l=58, r=40, t=106, b=56),
        height=height,
        hoverlabel=dict(bgcolor=PANEL, font=dict(color=TEXT)),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
        legend=dict(
            orientation="h",
            x=0,
            y=1.01,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED),
        ),
        showlegend=legend,
    )


def figure_html(fig: go.Figure, include_js: bool) -> str:
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "filename": "worldcup-dashboard-chart", "scale": 2},
        },
    )


def make_trend_chart(bundle: dict) -> go.Figure:
    trends = pd.DataFrame(bundle["trends"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trends["tournament_year"],
            y=trends["average_height_cm"],
            mode="lines+markers",
            name="Average height",
            line=dict(color=HEIGHT, width=3),
            marker=dict(size=7),
            hovertemplate="%{x}<br>%{y:.2f} cm<extra>Average height</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trends["tournament_year"],
            y=trends["average_age"],
            mode="lines+markers",
            name="Average age",
            line=dict(color=AGE, width=3),
            marker=dict(size=7),
            yaxis="y2",
            hovertemplate="%{x}<br>%{y:.2f} years<extra>Average age</extra>",
        )
    )
    fig.update_layout(
        **base_layout(
            "The modern World Cup has become taller and older",
            "Dual-axis view from 1990 onward. Height keeps climbing; age rises, then largely plateaus.",
            height=420,
            legend=True,
        )
    )
    fig.update_layout(
        yaxis=dict(title="Height, cm", gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis2=dict(
            title="Age, years",
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color=MUTED),
        ),
    )
    return fig


def make_country_chart(bundle: dict) -> go.Figure:
    countries = pd.DataFrame(bundle["countries_2026"]).sort_values("average_height_cm", ascending=False).head(12)
    fig = go.Figure(
        go.Bar(
            x=countries["average_height_cm"],
            y=countries["country"],
            orientation="h",
            marker=dict(
                color=countries["average_age"],
                colorscale=[
                    [0.0, "#9ad4e5"],
                    [0.5, "#5da4c4"],
                    [1.0, "#1f4f73"],
                ],
                colorbar=dict(title="Age"),
            ),
            customdata=countries[["average_age", "group", "height_rank", "age_rank"]],
            hovertemplate=(
                "<b>%{y}</b><br>Average height: %{x:.2f} cm<br>"
                "Average age: %{customdata[0]:.2f} years<br>"
                "Group: %{customdata[1]}<br>"
                "Height rank: %{customdata[2]}<br>Age rank: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **base_layout(
            "The tallest squads in the 2026 cohort",
            "Top 12 teams by average height. Color shows average age, so size and experience can be read together.",
            height=500,
            legend=False,
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def make_group_scatter(bundle: dict) -> go.Figure:
    groups = pd.DataFrame(bundle["groups_2026"])
    fig = go.Figure(
        go.Scatter(
            x=groups["average_height_cm"],
            y=groups["average_age"],
            mode="markers+text",
            text=groups["group"],
            textposition="top center",
            marker=dict(
                size=groups["player_rows"] / 4,
                color=groups["height_rank"],
                colorscale=[
                    [0.0, "#f3d6c7"],
                    [0.5, "#d9865d"],
                    [1.0, "#8b4e36"],
                ],
                showscale=False,
                line=dict(color="#ffffff", width=1.2),
            ),
            customdata=groups[["tallest_team", "oldest_team", "height_rank", "age_rank"]],
            hovertemplate=(
                "<b>%{text}</b><br>Average height: %{x:.2f} cm<br>Average age: %{y:.2f} years<br>"
                "Tallest team: %{customdata[0]}<br>Oldest team: %{customdata[1]}<br>"
                "Height rank: %{customdata[2]}<br>Age rank: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **base_layout(
            "Some groups skew big; others skew old",
            "Groups in the upper-right corner are both taller and older than the rest.",
            height=440,
        )
    )
    fig.update_layout(
        xaxis=dict(title="Average height, cm", showgrid=True, gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Average age, years", showgrid=True, gridcolor=GRID, tickfont=dict(color=MUTED)),
    )
    return fig


def make_confed_chart(bundle: dict) -> go.Figure:
    confeds = pd.DataFrame(bundle["confederations_2026"]).sort_values("average_height_cm", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=confeds["confederation"],
            y=confeds["average_height_cm"],
            name="Average height",
            marker_color=HEIGHT,
            hovertemplate="%{x}<br>%{y:.2f} cm<extra>Average height</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=confeds["confederation"],
            y=confeds["average_age"],
            name="Average age",
            mode="lines+markers",
            line=dict(color=AGE, width=3),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate="%{x}<br>%{y:.2f} years<extra>Average age</extra>",
        )
    )
    fig.update_layout(
        **base_layout(
            "Confederations do not share the same physical profile",
            "UEFA sits at the tall end, while other confederations compress differently on age and height.",
            height=420,
            legend=True,
        )
    )
    fig.update_layout(
        yaxis=dict(title="Height, cm", gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis2=dict(
            title="Age, years",
            overlaying="y",
            side="right",
            showgrid=False,
            tickfont=dict(color=MUTED),
        ),
    )
    return fig


def make_position_share_chart(bundle: dict) -> go.Figure:
    share = pd.DataFrame(bundle["position_share_trends"]).sort_values("tournament_year")
    colors = {
        "Goalkeeper": "#8e73b6",
        "Defender": "#ef5a4c",
        "Midfielder": "#f39a35",
        "Forward": "#73c6df",
    }
    fig = go.Figure()
    for position in ["Forward", "Midfielder", "Defender", "Goalkeeper"]:
        fig.add_trace(
            go.Scatter(
                x=share["tournament_year"],
                y=share[position],
                mode="lines",
                stackgroup="one",
                name=position,
                line=dict(width=0.7, color=colors[position]),
                hovertemplate=f"%{{x}}<br>{position}: %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        **base_layout(
            "The old forward-heavy World Cup has faded",
            "Position shares since 1990. This is a good tactical shorthand for how squads have changed.",
            height=430,
            legend=True,
        )
    )
    fig.update_layout(yaxis=dict(title="Share of player rows, %", gridcolor=GRID, tickfont=dict(color=MUTED)))
    return fig


def load_live_snapshot() -> dict:
    competition_path = RAW_FOOTBALL_DATA_DIR / "competition_wc.json"
    matches_path = RAW_FOOTBALL_DATA_DIR / "matches_wc.json"
    standings_path = RAW_FOOTBALL_DATA_DIR / "standings_wc.json"
    if not (competition_path.exists() and matches_path.exists() and standings_path.exists()):
        return {
            "available": False,
            "summary": "Live fixtures and standings will appear here once football-data.org pulls are enabled.",
            "next_matches": [],
            "standings_groups": [],
        }

    matches_payload = read_wrapped_json(matches_path)
    standings_payload = read_wrapped_json(standings_path)
    competition_payload = read_wrapped_json(competition_path)
    matches = sorted(matches_payload.get("matches", []), key=lambda match: match.get("utcDate", ""))
    next_matches = [
        {
            "utcDate": match.get("utcDate"),
            "stage": match.get("stage"),
            "group": match.get("group"),
            "matchday": match.get("matchday"),
            "home": match.get("homeTeam", {}).get("name"),
            "away": match.get("awayTeam", {}).get("name"),
            "status": match.get("status"),
            "venue": match.get("venue"),
            "city": match.get("city"),
        }
        for match in matches
    ]
    standings_groups = []
    for standing in standings_payload.get("standings", []):
        if standing.get("type") != "TOTAL":
            continue
        standings_groups.append(
            {
                "group": standing.get("group"),
                "table": [
                    {
                        "team": entry.get("team", {}).get("name"),
                        "points": entry.get("points"),
                        "playedGames": entry.get("playedGames"),
                    }
                    for entry in standing.get("table", [])[:4]
                ],
            }
        )
    return {
        "available": True,
        "summary": f"Live competition data loaded for {competition_payload.get('name', 'the World Cup')}.",
        "next_matches": next_matches,
        "standings_groups": standings_groups,
    }


def build_html(bundle: dict, charts: list[str], live_snapshot: dict) -> str:
    highlights = bundle["highlights"]
    countries = bundle["countries_2026"]
    live_json = json.dumps(live_snapshot)
    country_json = json.dumps(countries)
    group_members = json.dumps(bundle["group_members_2026"])
    story_json = json.dumps(bundle["story_manifest"])
    confed_history_json = json.dumps(bundle["confederation_history"])
    distribution_json = json.dumps(bundle["player_distribution_pool"])
    distribution_years_json = json.dumps(bundle["metadata"]["distribution_window_years"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup 2026 dashboard</title>
  <style>
    :root {{
      --bg: {BG};
      --panel: {PANEL};
      --text: {TEXT};
      --muted: {MUTED};
      --accent: {ACCENT};
      --age: {AGE};
      --height: {HEIGHT};
      --card: {CARD};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top right, rgba(139, 91, 71, 0.08), transparent 32%),
        linear-gradient(180deg, #f8f4ec 0%, {BG} 28%, #f4efe7 100%);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 38px 24px 90px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 22px;
      align-items: end;
      margin-bottom: 28px;
    }}
    .eyebrow {{
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2.8rem, 5vw, 5.2rem);
      line-height: 0.95;
      letter-spacing: -0.045em;
    }}
    .deck {{
      max-width: 48rem;
      font-size: 1.08rem;
      color: var(--muted);
      margin: 0;
    }}
    .hero-note {{
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 18px;
      padding: 18px;
      color: var(--muted);
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin: 18px 0 34px;
    }}
    .metric-card {{
      background: var(--panel);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 18px;
      padding: 16px;
      min-height: 126px;
    }}
    .metric-kicker {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 2rem;
      line-height: 1;
      letter-spacing: -0.03em;
      margin-bottom: 8px;
    }}
    .metric-copy {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    section {{
      margin-top: 32px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: end;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    h2 {{
      margin: 0;
      font-size: 2rem;
      line-height: 1;
      letter-spacing: -0.03em;
    }}
    .section-copy {{
      max-width: 46rem;
      color: var(--muted);
      margin: 8px 0 0;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .panel {{
      background: rgba(255,255,255,0.6);
      border: 1px solid rgba(0,0,0,0.05);
      border-radius: 20px;
      padding: 16px 18px 10px;
      overflow: hidden;
    }}
    .plotly-graph-div {{
      width: 100% !important;
      max-width: 100% !important;
    }}
    .module-grid {{
      display: grid;
      grid-template-columns: 0.95fr 1.05fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .list-card {{
      background: var(--panel);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 18px;
      padding: 16px;
    }}
    .story-list {{
      display: grid;
      gap: 12px;
      margin-top: 10px;
    }}
    .story-item {{
      background: var(--card);
      border-radius: 14px;
      padding: 14px;
    }}
    .story-item strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .toolbar input, .toolbar select {{
      font: inherit;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(0,0,0,0.12);
      background: #fffdfa;
      color: var(--text);
    }}
    .compare-toolbar {{
      display: grid;
      grid-template-columns: minmax(240px, 340px) 1fr;
      gap: 14px;
      align-items: start;
      margin-top: 12px;
    }}
    .search-shell {{
      position: relative;
    }}
    .search-results {{
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      display: none;
      background: #fffdfa;
      border: 1px solid rgba(0,0,0,0.1);
      border-radius: 14px;
      box-shadow: 0 10px 28px rgba(0,0,0,0.08);
      z-index: 10;
      overflow: hidden;
    }}
    .search-results.visible {{
      display: block;
    }}
    .search-result {{
      width: 100%;
      border: 0;
      background: transparent;
      color: var(--text);
      font: inherit;
      text-align: left;
      padding: 11px 12px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    .search-result:hover {{
      background: var(--card);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      table-layout: fixed;
      background: transparent;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    td.numeric, th.numeric {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .compare-card {{
      background: var(--card);
      border-radius: 18px;
      padding: 16px;
      min-height: 230px;
    }}
    .compare-card.empty {{
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      border: 1px dashed rgba(0,0,0,0.14);
      background: transparent;
    }}
    .compare-card h3 {{
      margin: 0 0 8px;
      font-size: 1.2rem;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .pill {{
      border: 1px solid rgba(0,0,0,0.12);
      background: #fffdfa;
      color: var(--text);
      padding: 9px 12px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .mini {{
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .live-shell {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .toggle-row {{
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .toggle-row button {{
      border: 1px solid rgba(0,0,0,0.12);
      background: #fffdfa;
      color: var(--text);
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .toggle-row button.active {{
      background: var(--card);
      border-color: rgba(0,0,0,0.18);
    }}
    .check-row {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .heatmap-card {{
      margin-top: 18px;
    }}
    .heatmap-scroll {{
      overflow-x: auto;
      padding-bottom: 6px;
      margin-top: 14px;
    }}
    .heatmap-stage {{
      min-width: max-content;
    }}
    .heatmap-row {{
      display: grid;
      align-items: center;
      gap: 2px;
      margin-top: 2px;
    }}
    .heatmap-header {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: linear-gradient(180deg, rgba(255,253,248,0.98), rgba(255,253,248,0.9));
      padding-bottom: 6px;
      margin-bottom: 4px;
    }}
    .heatmap-team {{
      position: sticky;
      left: 0;
      z-index: 1;
      background: rgba(255,253,248,0.96);
      padding-right: 12px;
      font-size: 0.92rem;
      white-space: nowrap;
    }}
    .heatmap-bin {{
      font-size: 0.7rem;
      color: var(--muted);
      text-align: center;
      width: 18px;
    }}
    .heatmap-cell {{
      width: 18px;
      height: 18px;
      border-radius: 2px;
      border: 1px solid rgba(255,255,255,0.4);
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.03);
    }}
    .heatmap-legend {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.84rem;
    }}
    .legend-ramp {{
      display: inline-flex;
      gap: 4px;
      align-items: center;
    }}
    .legend-chip {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(0,0,0,0.05);
    }}
    .live-list {{
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }}
    .live-tools {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .tz-toggle {{
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .live-filter {{
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .tz-toggle button {{
      border: 1px solid rgba(0,0,0,0.12);
      background: #fffdfa;
      color: var(--text);
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .tz-toggle button.active {{
      background: var(--card);
      border-color: rgba(0,0,0,0.18);
    }}
    .live-filter button {{
      border: 1px solid rgba(0,0,0,0.12);
      background: #fffdfa;
      color: var(--text);
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .live-filter button.active {{
      background: var(--card);
      border-color: rgba(0,0,0,0.18);
    }}
    .fixture {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      background: var(--card);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .fixture-meta {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.86rem;
    }}
    .fixture-day {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .group-box {{
      background: var(--card);
      border-radius: 14px;
      padding: 12px 14px;
      margin-top: 10px;
    }}
    .group-box strong {{
      display: block;
      margin-bottom: 6px;
    }}
    .standings-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    @media (max-width: 980px) {{
      .hero, .grid-2, .module-grid, .live-shell {{
        grid-template-columns: 1fr;
      }}
      .compare-toolbar {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <div>
        <div class="eyebrow">World Cup 2026 Dashboard</div>
        <h1>Built for the weeks before kickoff, and ready for the tournament itself.</h1>
        <p class="deck">A World Cup 2026 dashboard designed for fans and journalists: rankings, group comparisons, country comparisons, fast chart reads, and a live-updatable tournament shell. The current editorial focus is on height and age, because those are the clearest physical stories in the squad data.</p>
      </div>
      <div class="hero-note">
        <strong>Data note</strong><br>
        {bundle["metadata"]["dataset_note"]}<br><br>
        The historical trend window here focuses on 1990 onward, while country-vs-history cards look back roughly 30 to 40 years when the data exists.
      </div>
    </div>

    <div class="card-grid">
      <div class="metric-card">
        <div class="metric-kicker">2026 height record</div>
        <div class="metric-value">{highlights["latest_average_height_cm"]:.2f} cm</div>
        <div class="metric-copy">Average listed height in the 2026 cohort, up {highlights["height_gain_cm_since_1930"]:.2f} cm since 1930.</div>
      </div>
      <div class="metric-card">
        <div class="metric-kicker">2026 age level</div>
        <div class="metric-value">{highlights["latest_average_age"]:.2f} years</div>
        <div class="metric-copy">Older than the completed 2022 tournament ({highlights["previous_completed_average_age"]:.2f}) and effectively tied with 2018 when rounded.</div>
      </div>
      <div class="metric-card">
        <div class="metric-kicker">Tallest team</div>
        <div class="metric-value">{highlights["tallest_team_2026"]}</div>
        <div class="metric-copy">The highest average height in the 2026 cohort.</div>
      </div>
      <div class="metric-card">
        <div class="metric-kicker">Oldest team</div>
        <div class="metric-value">{highlights["oldest_team_2026"]}</div>
        <div class="metric-copy">The oldest average squad in the 2026 cohort.</div>
      </div>
      <div class="metric-card">
        <div class="metric-kicker">Tallest group</div>
        <div class="metric-value">{highlights["tallest_group_2026"]}</div>
        <div class="metric-copy">The most physically imposing group on average.</div>
      </div>
      <div class="metric-card">
        <div class="metric-kicker">Oldest group</div>
        <div class="metric-value">{highlights["oldest_group_2026"]}</div>
        <div class="metric-copy">The group that leans most heavily toward experience.</div>
      </div>
    </div>

    <section>
      <div class="section-head">
        <div>
          <h2>Core trends</h2>
          <p class="section-copy">The modern World Cup story is not subtle. Players have grown taller, squads have grown older, and the age curve has mostly flattened since the 1990s.</p>
        </div>
      </div>
      <div class="grid-2">
        <div class="panel">{charts[0]}</div>
        <div class="panel">{charts[1]}</div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Group and confederation pressure points</h2>
          <p class="section-copy">This is where scenario framing gets useful before kickoff. Some groups are big, some are old, and UEFA clearly sits at the tall end of the 2026 field. Use the year selector below to see whether that confederation gap holds in earlier tournaments too.</p>
        </div>
      </div>
      <div class="grid-2">
        <div class="panel">{charts[2]}</div>
        <div class="panel">{charts[3]}</div>
      </div>
      <div class="list-card" style="margin-top: 18px;">
        <div class="toolbar">
          <div class="mini">Confederation comparison</div>
          <select id="confed-year-a-select"></select>
          <select id="confed-year-b-select"></select>
        </div>
        <p class="section-copy" id="confed-copy" style="margin-top: 10px;"></p>
        <table>
          <thead>
            <tr>
              <th>Confederation</th>
              <th class="numeric">Teams A</th>
              <th class="numeric">Height A</th>
              <th class="numeric">Age A</th>
              <th class="numeric">Teams B</th>
              <th class="numeric">Height B</th>
              <th class="numeric">Age B</th>
              <th class="numeric">Height Δ</th>
              <th class="numeric">Age Δ</th>
            </tr>
          </thead>
          <tbody id="confed-table-body"></tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Country and story desk</h2>
          <p class="section-copy">This block is designed for newsroom use: pull quick story angles, filter teams, compare teams, and see who has changed the most from roughly 30 to 40 years ago.</p>
        </div>
      </div>
      <div class="module-grid">
        <div class="list-card">
          <div class="mini">Story prompts</div>
          <div id="story-list" class="story-list"></div>
        </div>
        <div class="list-card">
          <div class="mini">Country explorer</div>
          <div class="toolbar">
            <input id="country-search" type="text" placeholder="Search team country...">
            <select id="country-sort">
              <option value="average_height_cm-desc">Sort: tallest first</option>
              <option value="average_height_cm-asc">Sort: smallest first</option>
              <option value="average_age-desc">Sort: oldest first</option>
              <option value="average_age-asc">Sort: youngest first</option>
            </select>
          </div>
          <table>
            <thead>
              <tr>
                <th>Country</th>
                <th>Group</th>
                <th class="numeric">Height</th>
                <th class="numeric">Age</th>
                <th class="numeric">Hist.</th>
              </tr>
            </thead>
            <tbody id="country-table-body"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Quick comparison</h2>
          <p class="section-copy">Search for teams directly or click groups. Country cards compare the 2026 cohort with `1994` first, then `1986` where `1994` is unavailable.</p>
        </div>
      </div>
      <div class="compare-toolbar">
        <div class="search-shell">
          <input id="compare-search" type="text" placeholder="Search country to compare...">
          <div id="compare-search-results" class="search-results"></div>
        </div>
        <div>
          <div class="pill-row" id="group-pill-row"></div>
          <div class="pill-row" id="team-pill-row"></div>
        </div>
      </div>
      <div class="compare-grid" id="compare-grid"></div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Squad distribution heatmap</h2>
          <p class="section-copy">A GitHub-style tile view works here because it shows distribution, not just averages. Read each row as one team, each column as an age or height bucket, and each tile as how many players land there.</p>
        </div>
      </div>
      <div class="list-card heatmap-card">
        <div class="toolbar">
          <select id="distribution-year"></select>
          <div class="toggle-row">
            <button id="distribution-age" class="active" type="button">Age</button>
            <button id="distribution-height" type="button">Height</button>
          </div>
          <label class="check-row">
            <input id="distribution-exclude-gk" type="checkbox">
            Exclude goalkeepers
          </label>
        </div>
        <p class="section-copy" id="distribution-copy" style="margin-top: 10px;"></p>
        <div class="heatmap-legend" id="distribution-legend"></div>
        <div class="heatmap-scroll">
          <div class="heatmap-stage" id="distribution-stage"></div>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Tactical shape</h2>
          <p class="section-copy">Position shares are not a perfect tactical model, but they are a very good shorthand. The old forward-heavy World Cup is gone.</p>
        </div>
      </div>
      <div class="panel">{charts[4]}</div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Live center readiness</h2>
          <p class="section-copy">This is the section intended to switch from static pre-tournament context to updating tournament service once competition APIs are connected.</p>
        </div>
      </div>
      <div class="live-shell">
        <div class="list-card">
          <div class="mini" id="live-summary"></div>
          <div class="live-tools">
            <div class="mini" id="timezone-label">Times shown in UTC.</div>
            <div class="tz-toggle">
              <button id="tz-utc" class="active" type="button">Show UTC</button>
              <button id="tz-local" type="button">Show local time</button>
            </div>
          </div>
          <div class="live-tools">
            <div class="mini" id="filter-label">Showing all scheduled fixtures in the feed.</div>
            <div class="live-filter">
              <button id="filter-today" type="button">Today</button>
              <button id="filter-24h" type="button">Next 24h</button>
              <button id="filter-3d" type="button">Next 3 days</button>
              <button id="filter-all" class="active" type="button">All</button>
            </div>
          </div>
          <div id="fixture-list" class="live-list"></div>
        </div>
        <div class="list-card">
          <div class="mini">Group table snapshot</div>
          <div id="standings-list" class="standings-grid"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const countries = {country_json};
    const groupMembers = {group_members};
    const liveSnapshot = {live_json};
    const stories = {story_json};
    const confederationHistory = {confed_history_json};
    const playerDistributionPool = {distribution_json};
    const distributionYears = {distribution_years_json};

    (() => {{
      const storyList = document.getElementById("story-list");
      storyList.innerHTML = stories.map((story) => `
        <div class="story-item">
          <strong>${{story.headline}}</strong>
          <div class="mini">${{story.summary}}</div>
        </div>
      `).join("");
    }})();

    (() => {{
      const yearASelect = document.getElementById("confed-year-a-select");
      const yearBSelect = document.getElementById("confed-year-b-select");
      const tableBody = document.getElementById("confed-table-body");
      const copy = document.getElementById("confed-copy");
      const years = [...new Set(confederationHistory.map((row) => row.tournament_year))].sort((a, b) => b - a);
      const options = years.map((year) => `<option value="${{year}}">${{year}}</option>`).join("");
      yearASelect.innerHTML = options;
      yearBSelect.innerHTML = options;
      yearASelect.value = "2026";
      yearBSelect.value = years.includes(1994) ? "1994" : String(years[1] || years[0]);

      function formatDelta(value, unit) {{
        if (value == null || Number.isNaN(value)) return "n/a";
        const prefix = value > 0 ? "+" : "";
        return `${{prefix}}${{value.toFixed(2)}}${{unit}}`;
      }}

      function formatMetric(row, key, unit) {{
        if (!row) return "n/a";
        return `${{row[key].toFixed(2)}}${{unit}}`;
      }}

      function renderConfederations() {{
        const yearA = Number(yearASelect.value);
        const yearB = Number(yearBSelect.value);
        const rowsA = confederationHistory.filter((row) => row.tournament_year === yearA);
        const rowsB = confederationHistory.filter((row) => row.tournament_year === yearB);
        const mapA = new Map(rowsA.map((row) => [row.confederation, row]));
        const mapB = new Map(rowsB.map((row) => [row.confederation, row]));
        const confeds = [...new Set([...mapA.keys(), ...mapB.keys()])].sort((left, right) => {{
          const leftHeight = mapA.get(left)?.average_height_cm ?? -Infinity;
          const rightHeight = mapA.get(right)?.average_height_cm ?? -Infinity;
          if (leftHeight !== rightHeight) return rightHeight - leftHeight;
          return left.localeCompare(right);
        }});
        const tallestA = [...rowsA].sort((a, b) => b.average_height_cm - a.average_height_cm)[0];
        const tallestB = [...rowsB].sort((a, b) => b.average_height_cm - a.average_height_cm)[0];
        copy.textContent = `${{yearA}} vs ${{yearB}}. In ${{yearA}}, ${{tallestA.confederation}} is tallest at ${{tallestA.average_height_cm.toFixed(2)}} cm. In ${{yearB}}, ${{tallestB.confederation}} leads at ${{tallestB.average_height_cm.toFixed(2)}} cm.`;
        tableBody.innerHTML = confeds.map((confed) => {{
          const rowA = mapA.get(confed);
          const rowB = mapB.get(confed);
          const heightDelta = rowA && rowB ? rowA.average_height_cm - rowB.average_height_cm : null;
          const ageDelta = rowA && rowB ? rowA.average_age - rowB.average_age : null;
          return `
          <tr>
            <td>${{confed}}</td>
            <td class="numeric">${{rowA ? rowA.teams : "n/a"}}</td>
            <td class="numeric">${{formatMetric(rowA, "average_height_cm", " cm")}}</td>
            <td class="numeric">${{formatMetric(rowA, "average_age", " years")}}</td>
            <td class="numeric">${{rowB ? rowB.teams : "n/a"}}</td>
            <td class="numeric">${{formatMetric(rowB, "average_height_cm", " cm")}}</td>
            <td class="numeric">${{formatMetric(rowB, "average_age", " years")}}</td>
            <td class="numeric">${{formatDelta(heightDelta, " cm")}}</td>
            <td class="numeric">${{formatDelta(ageDelta, " years")}}</td>
          </tr>
        `;
        }}).join("");
      }}

      yearASelect.addEventListener("change", renderConfederations);
      yearBSelect.addEventListener("change", renderConfederations);
      renderConfederations();
    }})();

    (() => {{
      const search = document.getElementById("country-search");
      const sort = document.getElementById("country-sort");
      const body = document.getElementById("country-table-body");

      function render() {{
        const [key, direction] = sort.value.split("-");
        const query = search.value.trim().toLowerCase();
        const rows = [...countries]
          .filter((row) => row.country.toLowerCase().includes(query))
          .sort((a, b) => {{
            const dir = direction === "asc" ? 1 : -1;
            if (a[key] < b[key]) return -1 * dir;
            if (a[key] > b[key]) return 1 * dir;
            return a.country.localeCompare(b.country);
          }})
          .slice(0, 18);
        body.innerHTML = rows.map((row) => `
          <tr>
            <td>${{row.country}}</td>
            <td>${{row.group}}</td>
            <td class="numeric">${{row.average_height_cm.toFixed(2)}} cm</td>
            <td class="numeric">${{row.average_age.toFixed(2)}} years</td>
            <td class="numeric">${{row.baseline_tournament_year ? row.baseline_tournament_year : "n/a"}}</td>
          </tr>
        `).join("");
      }}

      search.addEventListener("input", render);
      sort.addEventListener("change", render);
      render();
    }})();

    (() => {{
      const compareGrid = document.getElementById("compare-grid");
      const groupPillRow = document.getElementById("group-pill-row");
      const teamPillRow = document.getElementById("team-pill-row");
      const compareSearch = document.getElementById("compare-search");
      const compareSearchResults = document.getElementById("compare-search-results");
      const selected = [];
      const maxCards = 4;

      function addCard(item) {{
        if (selected.find((entry) => entry.id === item.id) || selected.length >= maxCards) return;
        selected.push(item);
        renderCards();
      }}

      function removeCard(id) {{
        const index = selected.findIndex((entry) => entry.id === id);
        if (index >= 0) {{
          selected.splice(index, 1);
          renderCards();
        }}
      }}

      function renderCards() {{
        const cards = [...selected];
        while (cards.length < maxCards) cards.push(null);
        compareGrid.innerHTML = cards.map((item) => {{
          if (!item) return `<div class="compare-card empty">Add a team or a group</div>`;
          if (item.kind === "group") {{
            return `
              <div class="compare-card">
                <h3>${{item.group}}</h3>
                <div class="mini">Average height: ${{item.average_height_cm.toFixed(2)}} cm</div>
                <div class="mini">Average age: ${{item.average_age.toFixed(2)}} years</div>
                <div class="mini">Tallest team: ${{item.tallest_team}}</div>
                <div class="mini">Oldest team: ${{item.oldest_team}}</div>
                <table>
                  <thead>
                    <tr><th>Team</th><th class="numeric">Height</th><th class="numeric">Age</th></tr>
                  </thead>
                  <tbody>
                    ${{item.teams.map((team) => `
                      <tr>
                        <td>${{team.country}}</td>
                        <td class="numeric">${{team.average_height_cm.toFixed(2)}}</td>
                        <td class="numeric">${{team.average_age.toFixed(2)}}</td>
                      </tr>
                    `).join("")}}
                  </tbody>
                </table>
                <div class="pill-row"><button class="pill remove-pill" data-id="${{item.id}}">Remove</button></div>
              </div>
            `;
          }}
          const deltaHeight = item.height_delta_cm == null ? "n/a" : `${{item.height_delta_cm > 0 ? "+" : ""}}${{item.height_delta_cm.toFixed(2)}} cm`;
          const deltaAge = item.age_delta_years == null ? "n/a" : `${{item.age_delta_years > 0 ? "+" : ""}}${{item.age_delta_years.toFixed(2)}} years`;
          return `
            <div class="compare-card">
              <h3>${{item.country}}</h3>
              <div class="mini">Group: ${{item.group}} • ${{item.confederation}}</div>
              <div class="mini">Average height: ${{item.average_height_cm.toFixed(2)}} cm</div>
              <div class="mini">Average age: ${{item.average_age.toFixed(2)}} years</div>
              <div class="mini">Height rank: #${{item.height_rank}} • Age rank: #${{item.age_rank}}</div>
              <div class="mini">Baseline: ${{item.baseline_tournament_year || "n/a"}}</div>
              <div class="mini">Height change: ${{deltaHeight}}</div>
              <div class="mini">Age change: ${{deltaAge}}</div>
              <div class="pill-row"><button class="pill remove-pill" data-id="${{item.id}}">Remove</button></div>
            </div>
          `;
        }}).join("");
        compareGrid.querySelectorAll(".remove-pill").forEach((button) => {{
          button.addEventListener("click", () => removeCard(button.dataset.id));
        }});
      }}

      const groupsById = groupMembers.map((group) => {{
        const rows = countries.filter((country) => country.group === group.group);
        return {{
          id: `group:${{group.group}}`,
          kind: "group",
          group: group.group,
          teams: group.teams,
          average_height_cm: rows.reduce((sum, row) => sum + row.average_height_cm, 0) / rows.length,
          average_age: rows.reduce((sum, row) => sum + row.average_age, 0) / rows.length,
          tallest_team: [...rows].sort((a, b) => b.average_height_cm - a.average_height_cm)[0].country,
          oldest_team: [...rows].sort((a, b) => b.average_age - a.average_age)[0].country,
        }};
      }});

      function renderSearchResults(items) {{
        if (!items.length) {{
          compareSearchResults.classList.remove("visible");
          compareSearchResults.innerHTML = "";
          return;
        }}
        compareSearchResults.innerHTML = items.map((item) => `
          <button class="search-result" type="button" data-country="${{item.country}}">
            <span>${{item.country}}</span>
            <span class="mini">${{item.group}} • ${{item.average_height_cm.toFixed(2)}} cm • ${{item.average_age.toFixed(2)}} years</span>
          </button>
        `).join("");
        compareSearchResults.classList.add("visible");
        compareSearchResults.querySelectorAll(".search-result").forEach((button) => {{
          button.addEventListener("click", () => {{
            const item = countries.find((country) => country.country === button.dataset.country);
            if (item) addCard({{ ...item, id: `country:${{item.country}}`, kind: "country" }});
            compareSearch.value = "";
            renderSearchResults([]);
          }});
        }});
      }}

      groupPillRow.innerHTML = groupsById.map((group) => `
        <button class="pill" data-id="${{group.id}}">+ ${{group.group}}</button>
      `).join("");
      teamPillRow.innerHTML = [...countries]
        .sort((a, b) => a.height_rank - b.height_rank)
        .slice(0, 16)
        .map((country) => `<button class="pill" data-id="country:${{country.country}}">+ ${{country.country}}</button>`)
        .join("");

      document.querySelectorAll("[data-id^='group:']").forEach((button) => {{
        button.addEventListener("click", () => {{
          const item = groupsById.find((group) => group.id === button.dataset.id);
          if (item) addCard(item);
        }});
      }});
      document.querySelectorAll("[data-id^='country:']").forEach((button) => {{
        button.addEventListener("click", () => {{
          const name = button.dataset.id.replace("country:", "");
          const item = countries.find((country) => country.country === name);
          if (item) addCard({{ ...item, id: button.dataset.id, kind: "country" }});
        }});
      }});

      compareSearch.addEventListener("input", () => {{
        const query = compareSearch.value.trim().toLowerCase();
        if (!query) {{
          renderSearchResults([]);
          return;
        }}
        const matches = countries
          .filter((country) => !selected.find((entry) => entry.id === `country:${{country.country}}`))
          .filter((country) => country.country.toLowerCase().includes(query))
          .slice(0, 8);
        renderSearchResults(matches);
      }});

      compareSearch.addEventListener("focus", () => {{
        const query = compareSearch.value.trim().toLowerCase();
        if (!query) return;
        const matches = countries
          .filter((country) => !selected.find((entry) => entry.id === `country:${{country.country}}`))
          .filter((country) => country.country.toLowerCase().includes(query))
          .slice(0, 8);
        renderSearchResults(matches);
      }});

      document.addEventListener("click", (event) => {{
        if (!compareSearchResults.contains(event.target) && event.target !== compareSearch) {{
          renderSearchResults([]);
        }}
      }});

      renderCards();
    }})();

    (() => {{
      const yearSelect = document.getElementById("distribution-year");
      const ageButton = document.getElementById("distribution-age");
      const heightButton = document.getElementById("distribution-height");
      const excludeGk = document.getElementById("distribution-exclude-gk");
      const copy = document.getElementById("distribution-copy");
      const legend = document.getElementById("distribution-legend");
      const stage = document.getElementById("distribution-stage");
      let metric = "age";

      yearSelect.innerHTML = distributionYears
        .slice()
        .sort((a, b) => b - a)
        .map((year) => `<option value="${{year}}">${{year}}</option>`)
        .join("");
      yearSelect.value = "2026";

      function colorForRatio(ratio) {{
        if (ratio <= 0) return "#f7f3c4";
        if (ratio < 0.2) return "#d6efc2";
        if (ratio < 0.4) return "#99d8c9";
        if (ratio < 0.6) return "#55b7d7";
        if (ratio < 0.8) return "#2f7ec7";
        return "#21409a";
      }}

      function buildBins(rows) {{
        if (metric === "age") {{
          const values = rows
            .map((row) => row.age_at_tournament_years)
            .filter((value) => value != null)
            .map((value) => Math.round(value));
          const min = Math.min(...values);
          const max = Math.max(...values);
          return Array.from({{ length: max - min + 1 }}, (_, index) => min + index).map((value) => ({{
            key: String(value),
            label: String(value),
            test: (row) => row.age_at_tournament_years != null && Math.round(row.age_at_tournament_years) === value,
          }}));
        }}
        const values = rows
          .map((row) => row.height_cm)
          .filter((value) => value != null);
        const min = Math.floor(Math.min(...values) / 2) * 2;
        const max = Math.ceil(Math.max(...values) / 2) * 2;
        return Array.from({{ length: Math.floor((max - min) / 2) + 1 }}, (_, index) => min + index * 2).map((value) => ({{
          key: String(value),
          label: `${{value}}`,
          test: (row) => row.height_cm != null && Math.floor(row.height_cm / 2) * 2 === value,
        }}));
      }}

      function metricValue(row) {{
        return metric === "age" ? row.age_at_tournament_years : row.height_cm;
      }}

      function renderLegend(maxCount) {{
        const chips = [0, 0.25, 0.5, 0.75, 1].map((ratio) => `
          <span class="legend-chip" style="background:${{colorForRatio(ratio)}}"></span>
        `).join("");
        legend.innerHTML = `
          <span>Less</span>
          <span class="legend-ramp">${{chips}}</span>
          <span>More</span>
          <span>Cells show player counts per team and ${{metric === "age" ? "rounded age" : "2 cm height bin"}}. Max in this view: ${{maxCount}}.</span>
        `;
      }}

      function render() {{
        const selectedYear = Number(yearSelect.value);
        const filtered = playerDistributionPool
          .filter((row) => row.tournament_year === selectedYear)
          .filter((row) => !excludeGk.checked || row.position !== "Goalkeeper")
          .filter((row) => metricValue(row) != null);
        if (!filtered.length) {{
          stage.innerHTML = `<div class="mini">No player rows match this filter state.</div>`;
          copy.textContent = "No player rows match this filter state.";
          legend.innerHTML = "";
          return;
        }}

        const bins = buildBins(filtered);
        const teams = [...new Set(filtered.map((row) => row.country))].map((country) => {{
          const rows = filtered.filter((row) => row.country === country);
          const average = rows.reduce((sum, row) => sum + metricValue(row), 0) / rows.length;
          const counts = bins.map((bin) => rows.filter((row) => bin.test(row)).length);
          return {{ country, rows, average, counts }};
        }}).sort((a, b) => b.average - a.average || a.country.localeCompare(b.country));

        const maxCount = Math.max(...teams.flatMap((team) => team.counts), 1);
        const columns = `180px repeat(${{bins.length}}, 18px)`;
        const metricLabel = metric === "age" ? "age" : "height";
        const filterLabel = excludeGk.checked ? "excluding goalkeepers" : "including goalkeepers";
        const topTeam = teams[0];
        const lowTeam = teams[teams.length - 1];
        copy.textContent = `${{selectedYear}} distribution view, ${{filterLabel}}. Teams are sorted by average ${{metricLabel}} from high to low. ${{topTeam.country}} sits at the high end, while ${{lowTeam.country}} sits at the low end in this view.`;
        renderLegend(maxCount);

        const header = `
          <div class="heatmap-row heatmap-header" style="grid-template-columns:${{columns}};">
            <div class="heatmap-team mini">${{metric === "age" ? "Team / age" : "Team / height"}}</div>
            ${{bins.map((bin) => `<div class="heatmap-bin">${{bin.label}}</div>`).join("")}}
          </div>
        `;
        const rowsHtml = teams.map((team) => `
          <div class="heatmap-row" style="grid-template-columns:${{columns}};">
            <div class="heatmap-team">${{team.country}}</div>
            ${{team.counts.map((count, index) => {{
              const ratio = count / maxCount;
              const bg = colorForRatio(ratio);
              const label = bins[index].label;
              const unit = metric === "age" ? "years" : "cm bucket";
              return `<div class="heatmap-cell" style="background:${{bg}}" title="${{team.country}} • ${{label}} ${{unit}} • ${{count}} player${{count === 1 ? "" : "s"}}"></div>`;
            }}).join("")}}
          </div>
        `).join("");
        stage.innerHTML = header + rowsHtml;
      }}

      yearSelect.addEventListener("change", render);
      ageButton.addEventListener("click", () => {{
        metric = "age";
        ageButton.classList.add("active");
        heightButton.classList.remove("active");
        render();
      }});
      heightButton.addEventListener("click", () => {{
        metric = "height";
        heightButton.classList.add("active");
        ageButton.classList.remove("active");
        render();
      }});
      excludeGk.addEventListener("change", render);
      render();
    }})();

    (() => {{
      const liveSummary = document.getElementById("live-summary");
      const fixtureList = document.getElementById("fixture-list");
      const standingsList = document.getElementById("standings-list");
      const timezoneLabel = document.getElementById("timezone-label");
      const filterLabel = document.getElementById("filter-label");
      const utcButton = document.getElementById("tz-utc");
      const localButton = document.getElementById("tz-local");
      const filterToday = document.getElementById("filter-today");
      const filter24h = document.getElementById("filter-24h");
      const filter3d = document.getElementById("filter-3d");
      const filterAll = document.getElementById("filter-all");
      let timeMode = "utc";
      let fixtureFilter = "all";

      function localTimeZone() {{
        try {{
          return Intl.DateTimeFormat().resolvedOptions().timeZone || "local time";
        }} catch (error) {{
          return "local time";
        }}
      }}

      function formatKickoff(isoText) {{
        if (!isoText) return "";
        const date = new Date(isoText);
        if (Number.isNaN(date.getTime())) return isoText;
        if (timeMode === "local") {{
          return new Intl.DateTimeFormat(undefined, {{
            weekday: "short",
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
            timeZoneName: "short",
          }}).format(date);
        }}
        return new Intl.DateTimeFormat("en-GB", {{
          weekday: "short",
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          timeZone: "UTC",
          timeZoneName: "short",
        }}).format(date);
      }}

      function formatDayHeading(isoText) {{
        const date = new Date(isoText);
        if (Number.isNaN(date.getTime())) return isoText;
        const options = timeMode === "local"
          ? {{ weekday: "long", month: "short", day: "numeric" }}
          : {{ weekday: "long", month: "short", day: "numeric", timeZone: "UTC" }};
        return new Intl.DateTimeFormat(undefined, options).format(date);
      }}

      function groupLabel(rawGroup) {{
        if (!rawGroup) return "";
        return rawGroup.replace("GROUP_", "Group ");
      }}

      function stageLabel(match) {{
        const bits = [];
        if (match.group) bits.push(groupLabel(match.group));
        if (match.matchday) bits.push(`Matchday ${{match.matchday}}`);
        if (!bits.length && match.stage) bits.push(match.stage.replaceAll("_", " "));
        return bits.join(" · ");
      }}

      function inFilter(match) {{
        if (fixtureFilter === "all") return true;
        const date = new Date(match.utcDate);
        if (Number.isNaN(date.getTime())) return false;
        const now = new Date();
        const diffMs = date.getTime() - now.getTime();
        if (fixtureFilter === "24h") return diffMs >= 0 && diffMs <= 24 * 60 * 60 * 1000;
        if (fixtureFilter === "3d") return diffMs >= 0 && diffMs <= 3 * 24 * 60 * 60 * 1000;
        if (fixtureFilter === "today") {{
          const currentFormatter = timeMode === "local"
            ? new Intl.DateTimeFormat(undefined, {{ year: "numeric", month: "2-digit", day: "2-digit" }})
            : new Intl.DateTimeFormat("en-CA", {{ year: "numeric", month: "2-digit", day: "2-digit", timeZone: "UTC" }});
          return currentFormatter.format(date) === currentFormatter.format(now);
        }}
        return true;
      }}

      function dayKey(isoText) {{
        const date = new Date(isoText);
        if (Number.isNaN(date.getTime())) return isoText;
        if (timeMode === "local") {{
          return new Intl.DateTimeFormat("sv-SE", {{
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
          }}).format(date);
        }}
        return new Intl.DateTimeFormat("en-CA", {{
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          timeZone: "UTC",
        }}).format(date);
      }}

      function renderLive() {{
        timezoneLabel.textContent = timeMode === "local"
          ? `Times shown in your computer timezone: ${{localTimeZone()}}.`
          : "Times shown in UTC.";
        utcButton.classList.toggle("active", timeMode === "utc");
        localButton.classList.toggle("active", timeMode === "local");
        filterToday.classList.toggle("active", fixtureFilter === "today");
        filter24h.classList.toggle("active", fixtureFilter === "24h");
        filter3d.classList.toggle("active", fixtureFilter === "3d");
        filterAll.classList.toggle("active", fixtureFilter === "all");

        const filtered = liveSnapshot.next_matches.filter(inFilter);
        filterLabel.textContent = {{
          today: "Showing fixtures that fall on today in the selected timezone.",
          "24h": "Showing fixtures in the next 24 hours from now.",
          "3d": "Showing fixtures in the next 3 days from now.",
          all: "Showing all scheduled fixtures in the feed.",
        }}[fixtureFilter];

        if (!filtered.length) {{
          fixtureList.innerHTML = `<div class="fixture"><div>No fixtures match this time filter yet.</div><div class="mini">Try another window.</div></div>`;
        }} else {{
          const grouped = new Map();
          filtered.forEach((match) => {{
            const key = dayKey(match.utcDate);
            const bucket = grouped.get(key) || [];
            bucket.push(match);
            grouped.set(key, bucket);
          }});
          fixtureList.innerHTML = Array.from(grouped.entries()).map(([key, matches]) => `
            <div>
              <div class="fixture-day">${{formatDayHeading(matches[0].utcDate)}}</div>
              ${{matches.map((match) => `
                <div class="fixture">
                  <div>
                    <strong>${{match.home}} vs ${{match.away}}</strong>
                    <div class="fixture-meta">${{stageLabel(match)}}${{match.status ? ` • ${{match.status}}` : ""}}</div>
                    ${{match.venue || match.city ? `<div class="fixture-meta">${{[match.venue, match.city].filter(Boolean).join(" • ")}}</div>` : ""}}
                  </div>
                  <div class="mini">${{formatKickoff(match.utcDate)}}</div>
                </div>
              `).join("")}}
            </div>
          `).join("");
        }}

        standingsList.innerHTML = liveSnapshot.standings_groups.map((group) => `
          <div class="group-box">
            <strong>${{group.group || "Group"}}</strong>
            ${{group.table.map((row) => `<div class="mini">${{row.team}} • ${{row.points}} pts • ${{row.playedGames}} played</div>`).join("")}}
          </div>
        `).join("");
      }}

      liveSummary.textContent = liveSnapshot.summary;
      if (!liveSnapshot.available) {{
        fixtureList.innerHTML = `<div class="fixture"><div>No API data loaded yet.</div><div class="mini">Enable football-data.org pulls to populate this area.</div></div>`;
        standingsList.innerHTML = `<div class="group-box">Standings cards will appear once live data is connected.</div>`;
        return;
      }}
      utcButton.addEventListener("click", () => {{
        timeMode = "utc";
        renderLive();
      }});
      localButton.addEventListener("click", () => {{
        timeMode = "local";
        renderLive();
      }});
      filterToday.addEventListener("click", () => {{
        fixtureFilter = "today";
        renderLive();
      }});
      filter24h.addEventListener("click", () => {{
        fixtureFilter = "24h";
        renderLive();
      }});
      filter3d.addEventListener("click", () => {{
        fixtureFilter = "3d";
        renderLive();
      }});
      filterAll.addEventListener("click", () => {{
        fixtureFilter = "all";
        renderLive();
      }});
      renderLive();
    }})();
  </script>
</body>
</html>"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = read_wrapped_json(NORMALIZED_BUNDLE_PATH)
    live_snapshot = load_live_snapshot()

    charts = [
        figure_html(make_trend_chart(bundle), include_js=True),
        figure_html(make_country_chart(bundle), include_js=False),
        figure_html(make_group_scatter(bundle), include_js=False),
        figure_html(make_confed_chart(bundle), include_js=False),
        figure_html(make_position_share_chart(bundle), include_js=False),
    ]

    html = build_html(bundle, charts, live_snapshot)
    output_paths = [
        OUTPUT_DIR / "worldcup-dashboard.html",
        OUTPUT_DIR / "worldcup-dashboard-v2.html",
    ]
    for output_path in output_paths:
        output_path.write_text(html, encoding="utf-8")
        print(f"Saved dashboard to {output_path}")


if __name__ == "__main__":
    main()
