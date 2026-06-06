from __future__ import annotations

import json
from pathlib import Path
import unicodedata
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DATA_DIR = PROJECT_ROOT / "output" / "dashboard-data"
NORMALIZED_BUNDLE_PATH = DASHBOARD_DATA_DIR / "normalized" / "dashboard_bundle.json"
RAW_FOOTBALL_DATA_DIR = DASHBOARD_DATA_DIR / "raw" / "football_data"
TRANSFERMARKT_PLAYERS_PATH = DASHBOARD_DATA_DIR / "curated" / "transfermarkt_players_2026.csv"
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
FAVICON_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <rect width='64' height='64' rx='14' fill='#1f1b17'/>
  <rect x='4' y='4' width='56' height='56' rx='11' fill='#8b5b47' opacity='0.18'/>
  <text x='32' y='30' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='18' font-weight='700' fill='#fffdf8'>WC</text>
  <text x='32' y='48' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='18' font-weight='700' fill='#cf6d3e'>26</text>
</svg>
""".strip()
FAVICON_DATA_URL = f"data:image/svg+xml;utf8,{quote(FAVICON_SVG)}"
ECB_USD_PER_EUR = 1.1614
ECB_RATE_DATE = "3 June 2026"
ECB_EUR_PER_USD = 1 / ECB_USD_PER_EUR


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


def format_eur_millions_from_usd(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"€{(float(value) * ECB_EUR_PER_USD) / 1_000_000:.2f}m"


COUNTRY_NAME_ALIASES = {
    "turkiye": "turkey",
    "korea, south": "south korea",
    "usa": "united states",
    "u.s.a.": "united states",
    "cote d'ivoire": "ivory coast",
    "czechia": "czech republic",
    "iran, islamic republic of": "iran",
    "congo dr": "dr congo",
    "curacao": "curaçao",
    "cabo verde": "cape verde",
}


def normalize_country_name(value: str | None) -> str:
    if value is None:
        return ""
    compact = " ".join(str(value).strip().split()).casefold()
    compact = "".join(
        character
        for character in unicodedata.normalize("NFKD", compact)
        if not unicodedata.combining(character)
    )
    return COUNTRY_NAME_ALIASES.get(compact, compact)


def build_club_pathway_data() -> tuple[list[dict], list[dict]]:
    if not TRANSFERMARKT_PLAYERS_PATH.exists():
        return [], []
    players = pd.read_csv(TRANSFERMARKT_PLAYERS_PATH)
    needed = ["country", "player", "club", "club_country", "competition_name", "position"]
    if any(column not in players.columns for column in needed):
        return [], []
    players = players.dropna(subset=["country", "player", "club", "club_country"]).copy()
    if players.empty:
        return [], []
    players["competition_name"] = players["competition_name"].fillna("Unknown tier")
    players["country_norm"] = players["country"].map(normalize_country_name)
    players["club_country_norm"] = players["club_country"].map(normalize_country_name)
    players["is_domestic"] = players["country_norm"] == players["club_country_norm"]
    players["league_bucket"] = players["club_country"] + " • " + players["competition_name"]

    rows = [
        {
            "country": row["country"],
            "player": row["player"],
            "club": row["club"],
            "club_country": row["club_country"],
            "competition_name": row["competition_name"],
            "league_bucket": row["league_bucket"],
            "position": row["position"] if pd.notna(row["position"]) else None,
            "is_domestic": bool(row["is_domestic"]),
        }
        for row in players.sort_values(["country", "is_domestic", "league_bucket", "club", "player"]).to_dict("records")
    ]

    country_summary = (
        players.groupby("country", as_index=False)
        .agg(players=("player", "count"), domestic=("is_domestic", "sum"))
        .sort_values(["domestic", "players", "country"], ascending=[False, False, True])
    )
    country_summary["overseas"] = country_summary["players"] - country_summary["domestic"]
    country_summary["domestic_pct"] = country_summary["domestic"] / country_summary["players"] * 100

    top_domestic = country_summary.sort_values(
        ["domestic_pct", "players", "country"], ascending=[False, False, True]
    ).iloc[0]
    zero_domestic = country_summary[country_summary["domestic"] == 0].sort_values("country")
    top_bucket = (
        players.groupby("league_bucket", as_index=False)
        .agg(players=("player", "count"), countries=("country", "nunique"))
        .sort_values(["players", "countries", "league_bucket"], ascending=[False, False, True])
        .iloc[0]
    )
    stories = [
        {
            "slug": "domestic-core-leader-2026",
            "headline": f"{top_domestic['country']} keeps the strongest home-league core",
            "metric": "domestic_pct",
            "summary": (
                f"{top_domestic['domestic']} of {top_domestic['players']} players are domestic-based "
                f"({top_domestic['domestic_pct']:.1f}%)."
            ),
        },
        {
            "slug": "overseas-only-squads-2026",
            "headline": f"{len(zero_domestic)} squads have no domestic-based players at all",
            "metric": "domestic_zero_count",
            "summary": (
                f"{', '.join(zero_domestic['country'].head(6).tolist())}"
                + (" and others are fully overseas-based." if len(zero_domestic) > 6 else " are fully overseas-based.")
            ),
        },
        {
            "slug": "league-bucket-leader-2026",
            "headline": f"{top_bucket['league_bucket']} is the biggest club-location pipeline",
            "metric": "league_bucket_players",
            "summary": (
                f"It supplies {int(top_bucket['players'])} World Cup players across "
                f"{int(top_bucket['countries'])} national teams."
            ),
        },
    ]
    return rows, stories


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
    min_height = float(confeds["average_height_cm"].min()) - 2.0
    max_height = float(confeds["average_height_cm"].max()) + 2.0
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
            "Confederation height and age profiles diverge",
            "UEFA sits at the tall end; age and height do not move in lockstep across confederations.",
            height=420,
            legend=True,
        )
    )
    fig.update_layout(
        margin=dict(l=58, r=64, t=126, b=56),
        legend=dict(y=1.06),
        yaxis=dict(title="Height, cm", gridcolor=GRID, tickfont=dict(color=MUTED), range=[min_height, max_height]),
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
    sources = bundle.get("sources", [])
    club_pathway_rows, club_pathway_stories = build_club_pathway_data()
    club_benefit_country_rows = bundle.get("club_benefits_countries_2026", [])
    club_benefit_club_rows = bundle.get("club_benefits_clubs_2026", [])
    top_benefit_country = club_benefit_country_rows[0] if club_benefit_country_rows else {}
    top_benefit_club = club_benefit_club_rows[0] if club_benefit_club_rows else {}
    top_benefit_country_ceiling_eur = format_eur_millions_from_usd(top_benefit_country.get("estimated_ceiling_usd"))
    top_benefit_club_ceiling_eur = format_eur_millions_from_usd(top_benefit_club.get("estimated_ceiling_usd"))
    all_stories = []
    for story in bundle["story_manifest"]:
        adjusted = dict(story)
        if story.get("slug") == "club-benefits-country-leader-2026" and top_benefit_country:
            adjusted["summary"] = (
                f"Using FIFA's club-benefits framework and the current 2026 schedule, clubs based in "
                f"{top_benefit_country['club_country']} have an estimated ceiling of "
                f"{top_benefit_country_ceiling_eur} across {int(top_benefit_country['player_count'])} players."
            )
        elif story.get("slug") == "club-benefits-club-leader-2026" and top_benefit_club:
            adjusted["summary"] = (
                f"The same estimate puts {top_benefit_club['club']} on a ceiling of "
                f"{top_benefit_club_ceiling_eur} from {int(top_benefit_club['player_count'])} released players."
            )
        all_stories.append(adjusted)
    all_stories.extend(club_pathway_stories)
    transfermarkt_note = bundle.get("metadata", {}).get("transfermarkt_note")
    live_json = json.dumps(live_snapshot)
    country_json = json.dumps(countries)
    group_members = json.dumps(bundle["group_members_2026"])
    story_json = json.dumps(all_stories)
    confed_history_json = json.dumps(bundle["confederation_history"])
    distribution_json = json.dumps(bundle["player_distribution_pool"])
    distribution_years_json = json.dumps(bundle["metadata"]["distribution_window_years"])
    source_json = json.dumps(sources)
    market_value_json = json.dumps(bundle.get("market_value_players_2026", []))
    least_value_json = json.dumps(bundle.get("least_valuable_players_2026", []))
    player_stats_json = json.dumps(bundle.get("player_season_stats_2026", []))
    squad_value_json = json.dumps(bundle.get("squad_market_values_2026", []))
    global_club_json = json.dumps(bundle.get("global_club_representation_2026", []))
    country_club_json = json.dumps(bundle.get("country_club_representation_2026", []))
    club_pathway_json = json.dumps(club_pathway_rows)
    club_benefits_json = json.dumps(club_benefit_country_rows)
    club_benefits_clubs_json = json.dumps(club_benefit_club_rows)
    coaches_json = json.dumps(bundle.get("coaches_2026", []))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup 2026 dashboard</title>
  <link rel="icon" href="{FAVICON_DATA_URL}" type="image/svg+xml">
  <link rel="shortcut icon" href="{FAVICON_DATA_URL}" type="image/svg+xml">
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
      align-items: start;
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
      padding: 16px;
      color: var(--muted);
    }}
    .source-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .source-item {{
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(0,0,0,0.05);
      font-size: 0.85rem;
      line-height: 1.2;
    }}
    .source-item strong {{
      color: var(--text);
      display: inline;
      margin-bottom: 0;
    }}
    .hero-note-copy {{
      margin-top: 8px;
      font-size: 0.95rem;
      line-height: 1.42;
    }}
    .source-kicker {{
      margin-top: 12px;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .byline {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .byline a, .text-link {{
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid rgba(139, 91, 71, 0.28);
    }}
    .byline a:hover, .text-link:hover {{
      border-bottom-color: rgba(139, 91, 71, 0.7);
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin: 18px 0 34px;
    }}
    .page-shell {{
      position: relative;
    }}
    .content-shell {{
      min-width: 0;
    }}
    .section-guide {{
      position: fixed;
      top: 26px;
      right: -246px;
      width: 248px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(0,0,0,0.07);
      border-radius: 22px;
      padding: 18px 18px 16px;
      z-index: 40;
      box-shadow: 0 16px 44px rgba(0,0,0,0.08);
      backdrop-filter: blur(8px);
      opacity: 0.9;
      transition: right 180ms ease, opacity 180ms ease, box-shadow 180ms ease;
    }}
    .mobile-guide-toggle {{
      display: none;
    }}
    .mobile-guide-sheet {{
      display: none;
    }}
    .mobile-guide-backdrop {{
      display: none;
    }}
    .section-guide::before {{
      content: "Guide";
      position: absolute;
      left: -36px;
      top: 28px;
      padding: 10px 8px;
      border-radius: 14px 0 0 14px;
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(0,0,0,0.07);
      border-right: 0;
      color: var(--muted);
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      writing-mode: vertical-rl;
      transform: rotate(180deg);
    }}
    .section-guide:hover,
    .section-guide:focus-within {{
      right: 10px;
      opacity: 1;
      box-shadow: 0 20px 52px rgba(0,0,0,0.12);
    }}
    .section-guide h3 {{
      margin: 0 0 10px;
      font-size: 1.1rem;
      color: var(--text);
    }}
    .section-guide-list {{
      display: grid;
      gap: 10px;
    }}
    .section-guide-list a {{
      color: var(--muted);
      text-decoration: none;
      line-height: 1.25;
    }}
    .section-guide-list a:hover {{
      color: var(--accent);
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
    .table-wrap {{
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }}
    .table-wrap table.is-wide {{
      min-width: 820px;
    }}
    .module-grid {{
      display: grid;
      grid-template-columns: 0.95fr 1.05fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .ranking-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
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
    .pathway-shell {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .pathway-bars {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .pathway-row {{
      border: 1px solid rgba(0,0,0,0.08);
      background: #fffdfa;
      border-radius: 14px;
      padding: 10px 12px;
      cursor: pointer;
      text-align: left;
      font: inherit;
      color: inherit;
    }}
    .pathway-row.active {{
      border-color: rgba(44,125,160,0.4);
      box-shadow: inset 0 0 0 1px rgba(44,125,160,0.24);
      background: rgba(44,125,160,0.06);
    }}
    .pathway-row-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .pathway-row-head strong {{
      font-size: 0.98rem;
    }}
    .pathway-bar-track {{
      height: 10px;
      background: rgba(0,0,0,0.06);
      border-radius: 999px;
      overflow: hidden;
    }}
    .pathway-bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #9ad4e5 0%, #2c7da0 100%);
    }}
    .pathway-player-list {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .pathway-player-card {{
      border: 1px solid rgba(0,0,0,0.08);
      background: #fffdfa;
      border-radius: 14px;
      padding: 12px;
    }}
    .pathway-player-card strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .pathway-stat-strip {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .pathway-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 5px 10px;
      background: rgba(0,0,0,0.05);
      color: var(--muted);
      font-size: 0.84rem;
    }}
    .pathway-badge.is-domestic {{
      background: rgba(44,125,160,0.10);
      color: #245d74;
      font-weight: 600;
    }}
    .pathway-badge.is-overseas {{
      background: rgba(207,109,62,0.10);
      color: #8f4f31;
      font-weight: 600;
    }}
    .control-stack {{
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }}
    .inline-tools {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .small-button {{
      border: 1px solid rgba(0,0,0,0.12);
      background: #fffdfa;
      color: var(--text);
      padding: 8px 12px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .small-button[hidden] {{
      display: none;
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
    .mobile-card-list {{
      display: none;
      gap: 10px;
      margin-top: 12px;
    }}
    .desktop-only {{
      display: block;
    }}
    .mobile-data-card {{
      background: var(--card);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .mobile-data-card h3 {{
      margin: 0 0 8px;
      font-size: 1.05rem;
    }}
    .mobile-data-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
    }}
    .mobile-data-item {{
      min-width: 0;
    }}
    .mobile-data-item strong {{
      display: block;
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 2px;
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
    .compare-cohort-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .compare-cohort-block {{
      background: rgba(255,255,255,0.52);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .compare-cohort-label {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .compare-stat {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .compare-stat strong {{
      color: var(--text);
    }}
    .compare-stat.is-height-lead,
    .compare-stat.is-height-lead strong {{
      color: var(--height);
      font-weight: 700;
    }}
    .compare-stat.is-age-lead,
    .compare-stat.is-age-lead strong {{
      color: var(--age);
      font-weight: 700;
    }}
    .compare-delta-strong {{
      font-weight: 700;
      color: var(--text);
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
      cursor: pointer;
    }}
    .heatmap-tooltip {{
      position: fixed;
      z-index: 50;
      max-width: 320px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(25, 21, 18, 0.96);
      color: #fffdfa;
      font-size: 0.84rem;
      line-height: 1.35;
      box-shadow: 0 14px 32px rgba(0,0,0,0.22);
      pointer-events: none;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 120ms ease, transform 120ms ease;
      white-space: normal;
    }}
    .heatmap-tooltip.visible {{
      opacity: 1;
      transform: translateY(0);
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
    .stacked-block {{
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid rgba(0,0,0,0.08);
    }}
    .standings-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    @media (max-width: 980px) {{
      .hero, .grid-2, .module-grid, .live-shell, .ranking-grid, .pathway-shell {{
        grid-template-columns: 1fr;
      }}
      .compare-toolbar {{
        grid-template-columns: 1fr;
      }}
      .section-guide {{
        display: none;
      }}
      .mobile-guide-toggle {{
        display: inline-flex;
        position: fixed;
        right: 14px;
        bottom: 14px;
        z-index: 60;
        border: 1px solid rgba(0,0,0,0.12);
        background: rgba(255,253,248,0.96);
        color: var(--text);
        padding: 10px 14px;
        border-radius: 999px;
        font: inherit;
        box-shadow: 0 12px 28px rgba(0,0,0,0.12);
      }}
      .mobile-guide-backdrop.visible {{
        display: block;
        position: fixed;
        inset: 0;
        background: rgba(19, 16, 13, 0.28);
        z-index: 61;
      }}
      .mobile-guide-sheet {{
        position: fixed;
        left: 12px;
        right: 12px;
        bottom: 12px;
        z-index: 62;
        background: rgba(255,253,248,0.98);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 22px;
        padding: 16px;
        box-shadow: 0 20px 44px rgba(0,0,0,0.18);
        transform: translateY(calc(100% + 20px));
        opacity: 0;
        pointer-events: none;
        transition: transform 180ms ease, opacity 180ms ease;
      }}
      .mobile-guide-sheet.visible {{
        display: block;
        transform: translateY(0);
        opacity: 1;
        pointer-events: auto;
      }}
      .mobile-guide-sheet-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
      }}
      .desktop-only {{
        display: none !important;
      }}
    }}
    @media (max-width: 640px) {{
      main {{
        padding: 24px 14px 72px;
      }}
      h1 {{
        font-size: clamp(2.2rem, 11vw, 3.6rem);
      }}
      h2 {{
        font-size: 1.6rem;
      }}
      .deck, .section-copy {{
        font-size: 0.96rem;
      }}
      .panel {{
        padding: 10px 10px 6px;
      }}
      .list-card {{
        padding: 14px;
      }}
      .toolbar {{
        align-items: stretch;
      }}
      .toolbar input, .toolbar select {{
        width: 100%;
      }}
      .toolbar > .mini {{
        width: 100%;
      }}
      .table-wrap.desktop-only {{
        display: none;
      }}
      .mobile-card-list {{
        display: grid;
      }}
      .country-table-mobile .mobile-data-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .mobile-data-grid {{
        grid-template-columns: 1fr;
      }}
      .js-plotly-plot .modebar {{
        display: none !important;
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
        <p class="deck">A World Cup 2026 dashboard designed for fans and journalists: rankings, group comparisons, country comparisons, fast chart reads, and a live-updatable tournament shell. The current editorial focus is on height, age, market value, club representation, and coaches.</p>
        <p class="byline">A project by emot. <a href="https://joaotome.com" target="_blank" rel="noreferrer">joaotome.com</a></p>
      </div>
      <div class="hero-note">
        <strong>Method note</strong>
        <div class="hero-note-copy">`2026` is treated as a pre-kickoff dataset row, not completed tournament history. Trend charts focus on `1990` onward, while country history cards reach back roughly `30` to `40` years when the data exists.</div>
        <div class="source-kicker">Main sources</div>
        <div id="source-list" class="source-list"></div>
      </div>
    </div>

    <div class="page-shell">
      <div class="content-shell">
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
        <div class="metric-kicker">Top club pipeline</div>
        <div class="metric-value">{highlights["top_player_supply_club"] or "n/a"}</div>
        <div class="metric-copy">{highlights["top_player_supply_club_count"] or "n/a"} players in the 2026 squad pool come from the most represented club.</div>
      </div>
      <div class="metric-card">
        <div class="metric-kicker">Biggest club-country ceiling</div>
        <div class="metric-value">{highlights["top_club_country_benefit"] or "n/a"}</div>
        <div class="metric-copy">{top_benefit_country_ceiling_eur} estimated ceiling across {highlights["top_club_country_benefit_player_count"] or "n/a"} released players, converted from USD using the ECB reference rate from {ECB_RATE_DATE}.</div>
      </div>
      <div class="metric-card">
        <div class="metric-kicker">Biggest club payout ceiling</div>
        <div class="metric-value">{highlights["top_club_benefit_club"] or "n/a"}</div>
        <div class="metric-copy">{top_benefit_club_ceiling_eur} estimated ceiling from {highlights["top_club_benefit_club_player_count"] or "n/a"} released players, converted from USD using the ECB reference rate from {ECB_RATE_DATE}.</div>
      </div>
    </div>

    <section id="core-trends">
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

    <section id="group-confed-points">
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
        <div class="table-wrap desktop-only">
          <table class="is-wide">
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
        <div id="confed-mobile-cards" class="mobile-card-list"></div>
      </div>
    </section>

    <section id="country-story-desk">
      <div class="section-head">
        <div>
          <h2>Country and story desk</h2>
          <p class="section-copy">This block is designed for newsroom use: pull quick story angles, filter teams, compare teams, and see who has changed the most from roughly 30 to 40 years ago.</p>
        </div>
      </div>
      <div class="module-grid">
        <div class="list-card">
          <div class="mini">Story trends</div>
          <div id="story-list" class="story-list"></div>
          <div class="inline-tools">
            <button id="story-more-button" class="small-button" type="button">Show more</button>
          </div>
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
          <table class="desktop-only">
            <thead>
              <tr>
                <th>Country</th>
                <th>Group</th>
                <th class="numeric">Height</th>
                <th class="numeric">Age</th>
              </tr>
            </thead>
            <tbody id="country-table-body"></tbody>
          </table>
          <div id="country-mobile-cards" class="mobile-card-list country-table-mobile"></div>
          <div class="inline-tools">
            <button id="country-more-button" class="small-button" type="button">Show more</button>
          </div>
        </div>
      </div>
    </section>

    <section id="quick-comparison">
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

    <section id="club-pathways">
      <div class="section-head">
        <div>
          <h2>Domestic and overseas club pathways</h2>
          <p class="section-copy">This view tracks where each 2026 squad plays its club football. It uses current Transfermarkt club pages and groups players by club country plus league tier. That is strong enough to show domestic cores, export-heavy squads, and the main club pathways into each national team, even if it is not yet a full exact-league feed.</p>
        </div>
      </div>
      <div class="pathway-shell">
        <div class="list-card">
          <div class="toolbar">
            <div class="mini">League pathways by team</div>
            <select id="pathway-country-select"></select>
            <div class="toggle-row">
              <button id="pathway-all" class="active" type="button">All</button>
              <button id="pathway-domestic" type="button">Domestic only</button>
              <button id="pathway-overseas" type="button">Overseas only</button>
            </div>
          </div>
          <p class="section-copy" id="pathway-copy" style="margin-top: 10px;"></p>
          <div class="pathway-stat-strip" id="pathway-stat-strip"></div>
          <div id="pathway-bars" class="pathway-bars"></div>
        </div>
        <div class="list-card">
          <div class="toolbar">
            <div class="mini">Selected pathway details</div>
            <select id="pathway-club-select"></select>
          </div>
          <p class="section-copy" id="pathway-detail-copy" style="margin-top: 10px;"></p>
          <div id="pathway-player-list" class="pathway-player-list"></div>
        </div>
      </div>
    </section>

    <section id="value-clubs-coaches">
      <div class="section-head">
        <div>
          <h2>Value, clubs and coaches</h2>
          <p class="section-copy">This layer comes from current 2026 Transfermarkt pages rather than the historical Luis Batalha dataset. It is the right place for present-tense rankings: listed player values, club pipelines into the tournament, and the coach field.</p>
          {f'<p class="section-copy">{transfermarkt_note}</p>' if transfermarkt_note else ''}
        </div>
      </div>
      <div class="ranking-grid">
        <div class="list-card">
          <div class="mini">Most valuable players</div>
          <table>
            <thead>
              <tr>
                <th>Player</th>
                <th>Country</th>
                <th class="numeric">Value</th>
              </tr>
            </thead>
            <tbody id="market-value-body"></tbody>
          </table>
          <div class="inline-tools">
            <button id="market-more-button" class="small-button" type="button">Show more</button>
          </div>
          <div class="stacked-block">
            <div class="mini">Most valuable squads</div>
            <div class="control-stack">
              <input id="squad-value-search" type="text" placeholder="Search squad...">
            </div>
            <table>
              <thead>
                <tr>
                  <th>Team</th>
                  <th class="numeric">Squad value</th>
                </tr>
              </thead>
              <tbody id="squad-value-body"></tbody>
            </table>
            <div class="inline-tools">
              <button id="squad-more-button" class="small-button" type="button">Show more</button>
            </div>
          </div>
          <div class="stacked-block">
            <div class="mini">Lowest listed values</div>
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Country</th>
                  <th class="numeric">Value</th>
                </tr>
              </thead>
              <tbody id="least-value-body"></tbody>
            </table>
          </div>
        </div>
        <div class="list-card">
          <div class="mini">Clubs sending the most players</div>
          <div class="control-stack">
            <input id="global-club-search" type="text" placeholder="Search club...">
          </div>
          <table>
            <thead>
              <tr>
                <th>Club</th>
                <th class="numeric">Players</th>
                <th class="numeric">Teams</th>
              </tr>
            </thead>
            <tbody id="global-club-body"></tbody>
          </table>
          <div class="inline-tools">
            <button id="global-club-more-button" class="small-button" type="button">Show more</button>
          </div>
          <div class="stacked-block">
            <div class="mini">Club-benefit estimate by club country</div>
            <p class="section-copy" id="club-benefits-copy" style="margin-top: 8px;"></p>
            <div class="toggle-row" style="margin-top: 8px;">
              <button id="club-benefits-country-view" class="active" type="button">By club country</button>
              <button id="club-benefits-club-view" type="button">By club</button>
            </div>
            <div class="control-stack">
              <input id="club-benefits-search" type="text" placeholder="Search club country...">
            </div>
            <table>
              <thead>
                <tr>
                  <th id="club-benefits-head">Club country</th>
                  <th class="numeric">Players</th>
                  <th class="numeric">Floor</th>
                  <th class="numeric">Ceiling</th>
                </tr>
              </thead>
              <tbody id="club-benefits-body"></tbody>
            </table>
            <div class="inline-tools">
              <button id="club-benefits-more-button" class="small-button" type="button">Show more</button>
            </div>
          </div>
          <div class="stacked-block">
            <div class="toolbar">
              <div class="mini">All clubs in selected squad</div>
              <input id="country-club-search" type="text" placeholder="Search club in squad...">
              <select id="club-country-select"></select>
            </div>
            <p class="section-copy" style="margin-top: 8px;">This list includes every club represented in the selected 2026 squad, including one-player clubs.</p>
            <table>
              <thead>
                <tr>
                  <th>Club</th>
                  <th class="numeric">Players</th>
                </tr>
              </thead>
              <tbody id="country-club-body"></tbody>
            </table>
          </div>
        </div>
        <div class="list-card">
          <div class="toolbar">
            <div class="mini">Attack leaders</div>
            <input id="attack-search" type="text" placeholder="Search player or team...">
            <select id="attack-view-select">
              <option value="goals">Top scorers</option>
              <option value="assists">Top assisters</option>
              <option value="contributions">Top goal contributions</option>
              <option value="youngest-scorers">Youngest scorers</option>
              <option value="oldest-assisters">Oldest assisters</option>
            </select>
          </div>
          <p class="section-copy" id="attack-copy" style="margin-top: 10px;"></p>
          <table>
            <thead>
              <tr>
                <th>Player</th>
                <th>Team</th>
                <th class="numeric" id="attack-metric-head">Goals</th>
                <th class="numeric" id="attack-support-head">Assists</th>
              </tr>
            </thead>
            <tbody id="attack-body"></tbody>
          </table>
          <div class="inline-tools">
            <button id="attack-more-button" class="small-button" type="button">Show more</button>
          </div>
        </div>
        <div class="list-card">
          <div class="toolbar">
            <div class="mini">Coach desk</div>
            <select id="coach-view-select">
              <option value="decorated">Most decorated</option>
              <option value="oldest">Oldest coaches</option>
              <option value="youngest">Youngest coaches</option>
              <option value="tenure">Longest tenure</option>
              <option value="foreign">Foreign coaches</option>
              <option value="native">Native coaches</option>
              <option value="former-player">Former players</option>
            </select>
          </div>
          <p class="section-copy" id="coach-copy" style="margin-top: 10px;"></p>
          <table>
            <thead>
              <tr>
                <th>Coach</th>
                <th>Team</th>
                <th class="numeric">Age</th>
                <th class="numeric">Titles</th>
                <th>Tenure</th>
              </tr>
            </thead>
            <tbody id="coach-body"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="squad-distribution-heatmap">
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

    <section id="tactical-shape">
      <div class="section-head">
        <div>
          <h2>Tactical shape</h2>
          <p class="section-copy">Position shares are not a perfect tactical model, but they are a very good shorthand. The old forward-heavy World Cup is gone.</p>
        </div>
      </div>
      <div class="panel">{charts[4]}</div>
    </section>

    <section id="live-center-readiness">
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
              <button id="tz-local" type="button">Show my local time</button>
              <button id="tz-host" type="button">Show host reference time</button>
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
      </div>
      <aside class="section-guide" aria-label="Section guide">
        <h3>Dashboard guide</h3>
        <div class="section-guide-list">
          <a href="#core-trends">Core trends</a>
          <a href="#group-confed-points">Group and confederation pressure points</a>
          <a href="#country-story-desk">Country and story desk</a>
          <a href="#quick-comparison">Quick comparison</a>
          <a href="#club-pathways">Domestic and overseas pathways</a>
          <a href="#value-clubs-coaches">Value, clubs and coaches</a>
          <a href="#squad-distribution-heatmap">Squad distribution heatmap</a>
          <a href="#tactical-shape">Tactical shape</a>
          <a href="#live-center-readiness">Live center readiness</a>
        </div>
      </aside>
    </div>
  </main>
  <button id="mobile-guide-toggle" class="mobile-guide-toggle" type="button" aria-expanded="false" aria-controls="mobile-guide-sheet">Guide</button>
  <div id="mobile-guide-backdrop" class="mobile-guide-backdrop"></div>
  <div id="mobile-guide-sheet" class="mobile-guide-sheet" aria-label="Mobile dashboard guide">
    <div class="mobile-guide-sheet-head">
      <strong>Dashboard guide</strong>
      <button id="mobile-guide-close" class="small-button" type="button">Close</button>
    </div>
    <div class="section-guide-list">
      <a href="#core-trends">Core trends</a>
      <a href="#group-confed-points">Group and confederation pressure points</a>
      <a href="#country-story-desk">Country and story desk</a>
      <a href="#quick-comparison">Quick comparison</a>
      <a href="#club-pathways">Domestic and overseas pathways</a>
      <a href="#value-clubs-coaches">Value, clubs and coaches</a>
      <a href="#squad-distribution-heatmap">Squad distribution heatmap</a>
      <a href="#tactical-shape">Tactical shape</a>
      <a href="#live-center-readiness">Live center readiness</a>
    </div>
  </div>
  <div id="heatmap-tooltip" class="heatmap-tooltip" role="tooltip"></div>

  <script>
    const countries = {country_json};
    const groupMembers = {group_members};
    const liveSnapshot = {live_json};
    const stories = {story_json};
    const confederationHistory = {confed_history_json};
    const playerDistributionPool = {distribution_json};
    const distributionYears = {distribution_years_json};
    const sourceCatalog = {source_json};
    const marketValuePlayers = {market_value_json};
    const leastValuePlayers = {least_value_json};
    const playerSeasonStats = {player_stats_json};
    const squadValues = {squad_value_json};
    const globalClubCounts = {global_club_json};
    const countryClubCounts = {country_club_json};
    const clubPathwayPlayers = {club_pathway_json};
    const clubBenefitsCountries = {club_benefits_json};
    const clubBenefitsClubs = {club_benefits_clubs_json};
    const coaches = {coaches_json};
    const usdPerEur = {ECB_USD_PER_EUR};
    const eurPerUsd = 1 / usdPerEur;
    const ecbRateDate = "{ECB_RATE_DATE}";

    (() => {{
      function applyMobilePlotlyLayout() {{
        if (!window.Plotly) return;
        const isMobile = window.innerWidth <= 640;
        document.querySelectorAll(".js-plotly-plot").forEach((plot) => {{
          const layout = plot.layout || {{}};
          const update = isMobile
            ? {{
                "title.font.size": 16,
                "margin.l": 42,
                "margin.r": 28,
                "margin.t": 88,
                "margin.b": layout.showlegend ? 92 : 54,
              }}
            : {{
                "title.font.size": 24,
                "margin.l": 58,
                "margin.r": 40,
                "margin.t": 106,
                "margin.b": 56,
              }};
          if (Array.isArray(layout.annotations) && layout.annotations.length) {{
            update["annotations[0].font.size"] = isMobile ? 10 : 13;
            update["annotations[0].y"] = isMobile ? 1.03 : 1.1;
          }}
          if (layout.showlegend) {{
            update["legend.font.size"] = isMobile ? 11 : 14;
            update["legend.y"] = isMobile ? -0.24 : 1.01;
            update["legend.yanchor"] = isMobile ? "top" : "bottom";
            update["legend.x"] = 0;
            update["legend.xanchor"] = "left";
            update["legend.orientation"] = "h";
          }}
          window.Plotly.relayout(plot, update);
        }});
      }}

      let plotlyResizeTimer = null;
      window.addEventListener("load", () => {{
        window.setTimeout(applyMobilePlotlyLayout, 250);
      }});
      window.addEventListener("resize", () => {{
        window.clearTimeout(plotlyResizeTimer);
        plotlyResizeTimer = window.setTimeout(applyMobilePlotlyLayout, 150);
      }});
    }})();

    (() => {{
      const sourceList = document.getElementById("source-list");
      sourceList.innerHTML = sourceCatalog.map((source) => `
        <div class="source-item" title="${{source.scope}}">
          <strong>${{source.label}}</strong>
        </div>
      `).join("");
    }})();

    (() => {{
      const storyList = document.getElementById("story-list");
      const storyMoreButton = document.getElementById("story-more-button");
      let storyLimit = 5;

      function renderStories() {{
        const rows = stories.slice(0, storyLimit);
        storyList.innerHTML = rows.map((story) => `
          <div class="story-item">
            <strong>${{story.headline}}</strong>
            <div class="mini">${{story.summary}}</div>
          </div>
        `).join("");
        storyMoreButton.hidden = storyLimit >= stories.length;
      }}

      storyMoreButton.addEventListener("click", () => {{
        storyLimit = Math.min(storyLimit + 5, stories.length);
        renderStories();
      }});
      renderStories();
    }})();

    (() => {{
      const mobileGuideToggle = document.getElementById("mobile-guide-toggle");
      const mobileGuideSheet = document.getElementById("mobile-guide-sheet");
      const mobileGuideBackdrop = document.getElementById("mobile-guide-backdrop");
      const mobileGuideClose = document.getElementById("mobile-guide-close");

      function setMobileGuide(open) {{
        mobileGuideToggle.setAttribute("aria-expanded", open ? "true" : "false");
        mobileGuideSheet.classList.toggle("visible", open);
        mobileGuideBackdrop.classList.toggle("visible", open);
      }}

      mobileGuideToggle.addEventListener("click", () => setMobileGuide(!mobileGuideSheet.classList.contains("visible")));
      mobileGuideClose.addEventListener("click", () => setMobileGuide(false));
      mobileGuideBackdrop.addEventListener("click", () => setMobileGuide(false));
      mobileGuideSheet.querySelectorAll("a").forEach((anchor) => {{
        anchor.addEventListener("click", () => setMobileGuide(false));
      }});
    }})();

    (() => {{
      const yearASelect = document.getElementById("confed-year-a-select");
      const yearBSelect = document.getElementById("confed-year-b-select");
      const tableBody = document.getElementById("confed-table-body");
      const copy = document.getElementById("confed-copy");
      const mobileCards = document.getElementById("confed-mobile-cards");
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

      function formatUsdEstimateAsEur(value) {{
        if (value == null || Number.isNaN(Number(value))) return "n/a";
        return `€${{((Number(value) * eurPerUsd) / 1_000_000).toFixed(2)}}m`;
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
        mobileCards.innerHTML = confeds.map((confed) => {{
          const rowA = mapA.get(confed);
          const rowB = mapB.get(confed);
          const heightDelta = rowA && rowB ? rowA.average_height_cm - rowB.average_height_cm : null;
          const ageDelta = rowA && rowB ? rowA.average_age - rowB.average_age : null;
          return `
            <article class="mobile-data-card">
              <h3>${{confed}}</h3>
              <div class="mobile-data-grid">
                <div class="mobile-data-item"><strong>${{yearA}} teams</strong>${{rowA ? rowA.teams : "n/a"}}</div>
                <div class="mobile-data-item"><strong>${{yearA}} height</strong>${{formatMetric(rowA, "average_height_cm", " cm")}}</div>
                <div class="mobile-data-item"><strong>${{yearA}} age</strong>${{formatMetric(rowA, "average_age", " years")}}</div>
                <div class="mobile-data-item"><strong>${{yearB}} teams</strong>${{rowB ? rowB.teams : "n/a"}}</div>
                <div class="mobile-data-item"><strong>${{yearB}} height</strong>${{formatMetric(rowB, "average_height_cm", " cm")}}</div>
                <div class="mobile-data-item"><strong>${{yearB}} age</strong>${{formatMetric(rowB, "average_age", " years")}}</div>
                <div class="mobile-data-item"><strong>Height Δ</strong>${{formatDelta(heightDelta, " cm")}}</div>
                <div class="mobile-data-item"><strong>Age Δ</strong>${{formatDelta(ageDelta, " years")}}</div>
              </div>
            </article>
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
      const mobileCards = document.getElementById("country-mobile-cards");
      const moreButton = document.getElementById("country-more-button");
      let rowLimit = 18;

      function render() {{
        const [key, direction] = sort.value.split("-");
        const query = search.value.trim().toLowerCase();
        const allRows = [...countries]
          .filter((row) => row.country.toLowerCase().includes(query))
          .sort((a, b) => {{
            const dir = direction === "asc" ? 1 : -1;
            if (a[key] < b[key]) return -1 * dir;
            if (a[key] > b[key]) return 1 * dir;
            return a.country.localeCompare(b.country);
          }});
        const rows = allRows.slice(0, rowLimit);
        body.innerHTML = rows.map((row) => `
          <tr>
            <td>${{row.country}}</td>
            <td>${{row.group}}</td>
            <td class="numeric">${{row.average_height_cm.toFixed(2)}} cm</td>
            <td class="numeric">${{row.average_age.toFixed(2)}} years</td>
          </tr>
        `).join("");
        mobileCards.innerHTML = rows.map((row) => `
          <article class="mobile-data-card">
            <h3>${{row.country}}</h3>
            <div class="mobile-data-grid">
              <div class="mobile-data-item"><strong>Group</strong>${{row.group}}</div>
              <div class="mobile-data-item"><strong>Height</strong>${{row.average_height_cm.toFixed(2)}} cm</div>
              <div class="mobile-data-item"><strong>Age</strong>${{row.average_age.toFixed(2)}} years</div>
            </div>
          </article>
        `).join("");
        moreButton.hidden = rowLimit >= allRows.length;
      }}

      search.addEventListener("input", () => {{
        rowLimit = 18;
        render();
      }});
      sort.addEventListener("change", () => {{
        rowLimit = 18;
        render();
      }});
      moreButton.addEventListener("click", () => {{
        rowLimit += 18;
        render();
      }});
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
          const baselineLabel = item.baseline_tournament_year || "Baseline n/a";
          const baselineHeight = item.baseline_average_height_cm == null ? null : Number(item.baseline_average_height_cm);
          const baselineAge = item.baseline_average_age == null ? null : Number(item.baseline_average_age);
          const currentHeightLead = baselineHeight == null || item.average_height_cm >= baselineHeight;
          const currentAgeLead = baselineAge == null || item.average_age >= baselineAge;
          const baselineHeightLead = baselineHeight != null && baselineHeight > item.average_height_cm;
          const baselineAgeLead = baselineAge != null && baselineAge > item.average_age;
          const baselineHeightText = baselineHeight == null ? "n/a" : `${{baselineHeight.toFixed(2)}} cm`;
          const baselineAgeText = baselineAge == null ? "n/a" : `${{baselineAge.toFixed(2)}} years`;
          return `
            <div class="compare-card">
              <h3>${{item.country}}</h3>
              <div class="mini">Group: ${{item.group}} • ${{item.confederation}}</div>
              <div class="compare-cohort-grid">
                <div class="compare-cohort-block">
                  <div class="compare-cohort-label">2026 cohort</div>
                  <div class="compare-stat ${{currentHeightLead ? "is-height-lead" : ""}}"><strong>Height</strong> ${{item.average_height_cm.toFixed(2)}} cm</div>
                  <div class="compare-stat ${{currentAgeLead ? "is-age-lead" : ""}}"><strong>Age</strong> ${{item.average_age.toFixed(2)}} years</div>
                </div>
                <div class="compare-cohort-block">
                  <div class="compare-cohort-label">${{baselineLabel}}</div>
                  <div class="compare-stat ${{baselineHeightLead ? "is-height-lead" : ""}}"><strong>Height</strong> ${{baselineHeightText}}</div>
                  <div class="compare-stat ${{baselineAgeLead ? "is-age-lead" : ""}}"><strong>Age</strong> ${{baselineAgeText}}</div>
                </div>
              </div>
              <div class="mini" style="margin-top: 10px;">Height rank: #${{item.height_rank}} • Age rank: #${{item.age_rank}}</div>
              <div class="mini">Height change: <span class="compare-delta-strong">${{deltaHeight}}</span></div>
              <div class="mini">Age change: <span class="compare-delta-strong">${{deltaAge}}</span></div>
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
      const pathwayCountrySelect = document.getElementById("pathway-country-select");
      const pathwayAll = document.getElementById("pathway-all");
      const pathwayDomestic = document.getElementById("pathway-domestic");
      const pathwayOverseas = document.getElementById("pathway-overseas");
      const pathwayCopy = document.getElementById("pathway-copy");
      const pathwayStatStrip = document.getElementById("pathway-stat-strip");
      const pathwayBars = document.getElementById("pathway-bars");
      const pathwayClubSelect = document.getElementById("pathway-club-select");
      const pathwayDetailCopy = document.getElementById("pathway-detail-copy");
      const pathwayPlayerList = document.getElementById("pathway-player-list");

      const pathwayCountries = [...new Set(clubPathwayPlayers.map((row) => row.country))].sort((a, b) => a.localeCompare(b));
      pathwayCountrySelect.innerHTML = pathwayCountries.map((country) => `<option value="${{country}}">${{country}}</option>`).join("");
      if (pathwayCountries.includes("Brazil")) pathwayCountrySelect.value = "Brazil";

      let pathwayScope = "all";
      let selectedBucket = null;
      let selectedClubName = null;

      function scopeMatches(row) {{
        if (pathwayScope === "domestic") return row.is_domestic;
        if (pathwayScope === "overseas") return !row.is_domestic;
        return true;
      }}

      function activeCountryRows() {{
        return clubPathwayPlayers.filter((row) => row.country === pathwayCountrySelect.value);
      }}

      function renderPathways() {{
        const country = pathwayCountrySelect.value;
        const rows = activeCountryRows();
        const filtered = rows.filter(scopeMatches);
        const domesticCount = rows.filter((row) => row.is_domestic).length;
        const overseasCount = rows.length - domesticCount;
        const domesticPct = rows.length ? (domesticCount / rows.length) * 100 : 0;

        const bucketMap = new Map();
        filtered.forEach((row) => {{
          if (!bucketMap.has(row.league_bucket)) {{
            bucketMap.set(row.league_bucket, {{
              league_bucket: row.league_bucket,
              club_country: row.club_country,
              competition_name: row.competition_name,
              is_domestic: row.is_domestic,
              player_count: 0,
              clubs: new Map(),
            }});
          }}
          const bucket = bucketMap.get(row.league_bucket);
          bucket.player_count += 1;
          if (!bucket.clubs.has(row.club)) {{
            bucket.clubs.set(row.club, []);
          }}
          bucket.clubs.get(row.club).push(row);
        }});

        const buckets = [...bucketMap.values()]
          .map((bucket) => ({{
            ...bucket,
            clubs: [...bucket.clubs.entries()]
              .map(([club, players]) => ({{
                club,
                player_count: players.length,
                players: players.sort((left, right) => left.player.localeCompare(right.player)),
              }}))
              .sort((left, right) => right.player_count - left.player_count || left.club.localeCompare(right.club)),
          }}))
          .sort((left, right) => right.player_count - left.player_count || left.league_bucket.localeCompare(right.league_bucket));

        const topBucket = buckets[0];
        const scopeLabel = pathwayScope === "all" ? "all players" : pathwayScope === "domestic" ? "domestic-based players only" : "overseas-based players only";
        pathwayCopy.textContent = !rows.length
          ? "No club-pathway rows are available for this team."
          : !topBucket
            ? `${{country}} has no players in this view.`
            : `${{country}} is ${{domesticPct >= 50 ? "more domestic-based" : "more overseas-based"}} in the current squad snapshot: ${{domesticCount}} domestic and ${{overseasCount}} overseas. In the ${{scopeLabel}} view, the biggest pathway is ${{topBucket.league_bucket}} with ${{topBucket.player_count}} player${{topBucket.player_count === 1 ? "" : "s"}}.`;

        pathwayStatStrip.innerHTML = `
          <span class="pathway-badge is-domestic">Domestic: ${{domesticCount}} / ${{rows.length || 0}} (${{domesticPct.toFixed(1)}}%)</span>
          <span class="pathway-badge is-overseas">Overseas: ${{overseasCount}} / ${{rows.length || 0}} (${{(rows.length ? (overseasCount / rows.length) * 100 : 0).toFixed(1)}}%)</span>
          <span class="pathway-badge">${{topBucket ? `${{buckets.length}} pathway buckets` : "No pathway buckets in this filter"}}</span>
        `;

        const maxCount = Math.max(...buckets.map((bucket) => bucket.player_count), 1);
        pathwayBars.innerHTML = buckets.map((bucket) => `
          <button class="pathway-row ${{selectedBucket === bucket.league_bucket ? "active" : ""}}" type="button" data-bucket="${{bucket.league_bucket}}">
            <div class="pathway-row-head">
              <strong>${{bucket.league_bucket}}</strong>
              <span class="mini">${{bucket.player_count}} player${{bucket.player_count === 1 ? "" : "s"}}</span>
            </div>
            <div class="pathway-bar-track">
              <div class="pathway-bar-fill" style="width:${{(bucket.player_count / maxCount) * 100}}%"></div>
            </div>
          </button>
        `).join("");

        if (!topBucket) {{
          selectedBucket = null;
          selectedClubName = null;
          pathwayClubSelect.innerHTML = "<option value=''>No clubs</option>";
          pathwayDetailCopy.textContent = "This filter leaves no clubs to inspect.";
          pathwayPlayerList.innerHTML = "";
        }} else {{
          if (!selectedBucket || !buckets.find((bucket) => bucket.league_bucket === selectedBucket)) {{
            selectedBucket = topBucket.league_bucket;
          }}
          const selected = buckets.find((bucket) => bucket.league_bucket === selectedBucket) || topBucket;
          pathwayClubSelect.innerHTML = selected.clubs.map((club) => `<option value="${{club.club}}">${{club.club}} (${{club.player_count}})</option>`).join("");
          if (!selected.clubs.find((club) => club.club === selectedClubName)) {{
            selectedClubName = selected.clubs[0]?.club || null;
          }}
          pathwayClubSelect.value = selectedClubName || selected.clubs[0]?.club || "";
          const selectedClub = selected.clubs.find((club) => club.club === pathwayClubSelect.value) || selected.clubs[0];
          pathwayDetailCopy.textContent = `${{selected.league_bucket}} is represented here by ${{selected.clubs.length}} club${{selected.clubs.length === 1 ? "" : "s"}}. ${{selectedClub ? `${{selectedClub.club}} contributes ${{selectedClub.player_count}} player${{selectedClub.player_count === 1 ? "" : "s"}}.` : ""}}`;
          pathwayPlayerList.innerHTML = selectedClub
            ? selectedClub.players.map((row) => `
                <div class="pathway-player-card">
                  <strong>${{row.player}}</strong>
                  <div class="mini">${{row.club}} • ${{row.position || "Position n/a"}}</div>
                  <div class="mini">${{row.is_domestic ? "Domestic-based" : "Overseas-based"}} • ${{row.club_country}} • ${{row.competition_name}}</div>
                </div>
              `).join("")
            : "";
        }}

        pathwayBars.querySelectorAll(".pathway-row").forEach((button) => {{
          button.addEventListener("click", () => {{
            selectedBucket = button.dataset.bucket;
            renderPathways();
          }});
        }});

        pathwayAll.classList.toggle("active", pathwayScope === "all");
        pathwayDomestic.classList.toggle("active", pathwayScope === "domestic");
        pathwayOverseas.classList.toggle("active", pathwayScope === "overseas");
      }}

      pathwayCountrySelect.addEventListener("change", () => {{
        selectedBucket = null;
        selectedClubName = null;
        renderPathways();
      }});
      pathwayClubSelect.addEventListener("change", () => {{ selectedClubName = pathwayClubSelect.value; renderPathways(); }});
      pathwayAll.addEventListener("click", () => {{ pathwayScope = "all"; selectedBucket = null; selectedClubName = null; renderPathways(); }});
      pathwayDomestic.addEventListener("click", () => {{ pathwayScope = "domestic"; selectedBucket = null; selectedClubName = null; renderPathways(); }});
      pathwayOverseas.addEventListener("click", () => {{ pathwayScope = "overseas"; selectedBucket = null; selectedClubName = null; renderPathways(); }});
      renderPathways();
    }})();

    (() => {{
      const marketBody = document.getElementById("market-value-body");
      const marketMoreButton = document.getElementById("market-more-button");
      const squadValueBody = document.getElementById("squad-value-body");
      const squadSearch = document.getElementById("squad-value-search");
      const squadMoreButton = document.getElementById("squad-more-button");
      const leastBody = document.getElementById("least-value-body");
      const globalClubBody = document.getElementById("global-club-body");
      const globalClubSearch = document.getElementById("global-club-search");
      const globalClubMoreButton = document.getElementById("global-club-more-button");
      const clubBenefitsBody = document.getElementById("club-benefits-body");
      const clubBenefitsSearch = document.getElementById("club-benefits-search");
      const clubBenefitsCopy = document.getElementById("club-benefits-copy");
      const clubBenefitsMoreButton = document.getElementById("club-benefits-more-button");
      const clubBenefitsHead = document.getElementById("club-benefits-head");
      const clubBenefitsCountryView = document.getElementById("club-benefits-country-view");
      const clubBenefitsClubView = document.getElementById("club-benefits-club-view");
      const countrySelect = document.getElementById("club-country-select");
      const countryClubSearch = document.getElementById("country-club-search");
      const countryClubBody = document.getElementById("country-club-body");
      const attackSearch = document.getElementById("attack-search");
      const attackViewSelect = document.getElementById("attack-view-select");
      const attackCopy = document.getElementById("attack-copy");
      const attackMetricHead = document.getElementById("attack-metric-head");
      const attackSupportHead = document.getElementById("attack-support-head");
      const attackBody = document.getElementById("attack-body");
      const attackMoreButton = document.getElementById("attack-more-button");
      const coachViewSelect = document.getElementById("coach-view-select");
      const coachBody = document.getElementById("coach-body");
      const coachCopy = document.getElementById("coach-copy");
      let marketValueLimit = 10;
      let squadLimit = 8;
      let globalClubLimit = 12;
      let clubBenefitsLimit = 10;
      let attackLimit = 10;
      let clubBenefitsView = "country";

      function safeBind(element, eventName, handler) {{
        if (!element) return;
        element.addEventListener(eventName, handler);
      }}

      function safeRender(label, renderer) {{
        try {{
          renderer();
        }} catch (error) {{
          console.error(`Value/clubs/coaches renderer failed: ${{label}}`, error);
        }}
      }}

      function renderMarketValues() {{
        const rows = marketValuePlayers.slice(0, marketValueLimit);
        marketBody.innerHTML = rows.map((row) => `
          <tr>
            <td>${{row.player}}<div class="mini">${{row.club || "Club n/a"}}</div></td>
            <td>${{row.country}}</td>
            <td class="numeric">${{row.market_value_text}}</td>
          </tr>
        `).join("");
        marketMoreButton.hidden = marketValueLimit >= marketValuePlayers.length;
      }}

      function renderSquadValues() {{
        const query = squadSearch.value.trim().toLowerCase();
        const rows = squadValues
          .filter((row) => row.country.toLowerCase().includes(query))
          .slice(0, squadLimit);
        squadValueBody.innerHTML = rows.map((row) => `
          <tr>
            <td>${{row.country}}</td>
            <td class="numeric">${{row.squad_market_value_text}}</td>
          </tr>
        `).join("");
        const total = squadValues.filter((row) => row.country.toLowerCase().includes(query)).length;
        squadMoreButton.hidden = squadLimit >= total;
      }}

      leastBody.innerHTML = leastValuePlayers.slice(0, 8).map((row) => `
        <tr>
          <td>${{row.player}}<div class="mini">${{row.club || "Club n/a"}}</div></td>
          <td>${{row.country}}</td>
          <td class="numeric">${{row.market_value_text}}</td>
        </tr>
      `).join("");

      function renderGlobalClubs() {{
        const query = globalClubSearch.value.trim().toLowerCase();
        const filtered = globalClubCounts.filter((row) => (row.club || "").toLowerCase().includes(query));
        const rows = filtered.slice(0, globalClubLimit);
        globalClubBody.innerHTML = rows.map((row) => `
          <tr>
            <td>${{row.club}}<div class="mini">${{row.club_country || "club country n/a"}}</div></td>
            <td class="numeric">${{row.player_count}}</td>
            <td class="numeric">${{row.represented_countries}}</td>
          </tr>
        `).join("");
        globalClubMoreButton.hidden = globalClubLimit >= filtered.length;
      }}

      function renderClubBenefits() {{
        const query = clubBenefitsSearch.value.trim().toLowerCase();
        const sourceRows = clubBenefitsView === "country" ? clubBenefitsCountries : clubBenefitsClubs;
        const filtered = sourceRows.filter((row) => {{
          const left = clubBenefitsView === "country" ? (row.club_country || "") : (row.club || "");
          const right = row.club_country || "";
          return left.toLowerCase().includes(query) || right.toLowerCase().includes(query);
        }});
        const rows = filtered.slice(0, clubBenefitsLimit);
        clubBenefitsHead.textContent = clubBenefitsView === "country" ? "Club country" : "Club";
        clubBenefitsCountryView.classList.toggle("active", clubBenefitsView === "country");
        clubBenefitsClubView.classList.toggle("active", clubBenefitsView === "club");
        clubBenefitsSearch.placeholder = clubBenefitsView === "country" ? "Search club country..." : "Search club...";
        clubBenefitsCopy.textContent = clubBenefitsView === "country"
          ? `Converted to euro using the ECB reference rate from ${{ecbRateDate}}. Original estimate starts from the AP-reported US$5,000-per-player-per-day figure, plus FIFA's 25 May 2026 release date and the current match schedule. Qualifier payments are excluded, and club country is inferred from current Transfermarkt club profile headers.`
          : `Converted to euro using the ECB reference rate from ${{ecbRateDate}}. This ranks individual clubs by the same estimated World Cup release-window compensation range. It is an estimate, not an official FIFA settlement file.`;
        clubBenefitsBody.innerHTML = rows.map((row) => {{
          const primary = clubBenefitsView === "country" ? row.club_country : row.club;
          const meta = clubBenefitsView === "country" ? `${{row.club_count}} clubs` : (row.club_country || "club country n/a");
          return `
            <tr>
              <td>${{primary}}<div class="mini">${{meta}} • ${{row.represented_squads}} squads</div></td>
              <td class="numeric">${{row.player_count}}</td>
              <td class="numeric">${{formatUsdEstimateAsEur(row.estimated_floor_usd)}}</td>
              <td class="numeric">${{formatUsdEstimateAsEur(row.estimated_ceiling_usd)}}</td>
            </tr>
          `;
        }}).join("");
        clubBenefitsMoreButton.hidden = clubBenefitsLimit >= filtered.length;
      }}

      const clubCountries = [...new Set(countryClubCounts.map((row) => row.country))].sort((a, b) => a.localeCompare(b));
      countrySelect.innerHTML = clubCountries.map((country) => `<option value="${{country}}">${{country}}</option>`).join("");

      function renderCountryClubs() {{
        const country = countrySelect.value;
        const rows = countryClubCounts
          .filter((row) => row.country === country)
          .filter((row) => (row.club || "").toLowerCase().includes(countryClubSearch.value.trim().toLowerCase()))
          .sort((a, b) => a.country_rank - b.country_rank || b.player_count - a.player_count || a.club.localeCompare(b.club));
        countryClubBody.innerHTML = rows.map((row) => `
          <tr>
            <td>${{row.club}}<div class="mini">${{row.club_country || "club country n/a"}}</div></td>
            <td class="numeric">${{row.player_count}}</td>
          </tr>
        `).join("");
      }}
      safeBind(countrySelect, "change", renderCountryClubs);
      safeBind(countryClubSearch, "input", renderCountryClubs);
      if (clubCountries.includes("Brazil")) {{
        countrySelect.value = "Brazil";
      }}

      function parseTenureDays(text) {{
        if (!text) return -1;
        const yearMatch = text.match(/(\\d+)\\s+year/);
        const monthMatch = text.match(/(\\d+)\\s+month/);
        const dayMatch = text.match(/(\\d+)\\s+day/);
        const years = yearMatch ? Number(yearMatch[1]) : 0;
        const months = monthMatch ? Number(monthMatch[1]) : 0;
        const days = dayMatch ? Number(dayMatch[1]) : 0;
        return years * 365 + months * 30 + days;
      }}

      function renderAttack() {{
        const view = attackViewSelect.value;
        let rows = [...playerSeasonStats];
        let copy = "Current club-season output from Transfermarkt player performance pages, excluding national-team matches.";
        let metricLabel = "Goals";
        let supportLabel = "Assists";

        if (view === "goals") {{
          rows = rows
            .filter((row) => row.goals > 0)
            .sort((a, b) => b.goals - a.goals || b.assists - a.assists || a.player.localeCompare(b.player))
          copy = "Top scorers among 2026 World Cup players, ranked by goals in each player's latest club season in the fetched Transfermarkt performance data.";
        }} else if (view === "assists") {{
          rows = rows
            .filter((row) => row.assists > 0)
            .sort((a, b) => b.assists - a.assists || b.goals - a.goals || a.player.localeCompare(b.player));
          copy = "Top assisters among 2026 World Cup players, ranked by assists in each player's latest club season in the fetched Transfermarkt performance data.";
          metricLabel = "Assists";
          supportLabel = "Goals";
        }} else if (view === "contributions") {{
          rows = rows
            .filter((row) => row.goal_contributions > 0)
            .sort((a, b) => b.goal_contributions - a.goal_contributions || b.goals - a.goals || a.player.localeCompare(b.player));
          copy = "Goal contributions can be cleaner than raw goals because they catch elite creators and scorers in the same table.";
          metricLabel = "G+A";
          supportLabel = "Season";
        }} else if (view === "youngest-scorers") {{
          rows = rows
            .filter((row) => row.goals > 0 && row.age != null)
            .sort((a, b) => a.age - b.age || b.goals - a.goals || a.player.localeCompare(b.player));
          copy = "This isolates the youngest players who already arrive with real club-season scoring output rather than hype alone.";
          metricLabel = "Age";
          supportLabel = "Goals";
        }} else if (view === "oldest-assisters") {{
          rows = rows
            .filter((row) => row.assists > 0 && row.age != null)
            .sort((a, b) => b.age - a.age || b.assists - a.assists || a.player.localeCompare(b.player));
          copy = "This catches the veteran creators still shaping attacks late in their careers.";
          metricLabel = "Age";
          supportLabel = "Assists";
        }}

        const query = attackSearch.value.trim().toLowerCase();
        rows = rows.filter((row) => {{
          const player = row.player?.toLowerCase() || "";
          const club = row.club?.toLowerCase() || "";
          const country = row.country?.toLowerCase() || "";
          return player.includes(query) || club.includes(query) || country.includes(query);
        }});

        attackCopy.textContent = copy;
        attackMetricHead.textContent = metricLabel;
        attackSupportHead.textContent = supportLabel;
        const visibleRows = rows.slice(0, attackLimit);
        attackBody.innerHTML = visibleRows.map((row) => {{
          let metricValue = row.goals;
          let supportValue = row.assists;
          if (view === "assists") {{
            metricValue = row.assists;
            supportValue = row.goals;
          }} else if (view === "contributions") {{
            metricValue = row.goal_contributions;
            supportValue = row.current_season_id;
          }} else if (view === "youngest-scorers") {{
            metricValue = row.age != null ? `${{row.age.toFixed(0)}}y` : "n/a";
            supportValue = row.goals;
          }} else if (view === "oldest-assisters") {{
            metricValue = row.age != null ? `${{row.age.toFixed(0)}}y` : "n/a";
            supportValue = row.assists;
          }}
          return `
            <tr>
              <td>${{row.player}}<div class="mini">${{row.country}} • ${{row.position || "Position n/a"}}</div></td>
              <td>${{row.club || "Club n/a"}}<div class="mini">season ${{row.current_season_id}}</div></td>
              <td class="numeric">${{metricValue}}</td>
              <td class="numeric">${{supportValue}}</td>
            </tr>
          `;
        }}).join("");
        attackMoreButton.hidden = attackLimit >= rows.length;
      }}

      function renderCoachDesk() {{
        if (!coaches.length) {{
          coachCopy.textContent = "Coach enrichment was not loaded for this build.";
          coachBody.innerHTML = "";
          return;
        }}

        const view = coachViewSelect.value;
        let rows = [...coaches];
        let copy = "";

        if (view === "decorated") {{
          rows = rows
            .sort((a, b) => b.total_titles_won - a.total_titles_won || a.manager.localeCompare(b.manager))
            .slice(0, 10);
          const leader = rows[0];
          copy = `Most decorated coaches by counted Transfermarkt honours. ${{
            leader ? `${{leader.manager}} leads this field with ${{leader.total_titles_won}} titles.` : ""
          }}`;
        }} else if (view === "oldest") {{
          rows = rows
            .filter((row) => row.age != null)
            .sort((a, b) => b.age - a.age || b.total_titles_won - a.total_titles_won || a.manager.localeCompare(b.manager))
            .slice(0, 10);
          const leader = rows[0];
          copy = `Oldest coaches in the 2026 field. ${{
            leader ? `${{leader.manager}} is the oldest at ${{leader.age.toFixed(0)}}.` : ""
          }}`;
        }} else if (view === "youngest") {{
          rows = rows
            .filter((row) => row.age != null)
            .sort((a, b) => a.age - b.age || b.total_titles_won - a.total_titles_won || a.manager.localeCompare(b.manager))
            .slice(0, 10);
          const leader = rows[0];
          copy = `Youngest coaches in the 2026 field. ${{
            leader ? `${{leader.manager}} is the youngest at ${{leader.age.toFixed(0)}}.` : ""
          }}`;
        }} else if (view === "tenure") {{
          rows = rows
            .filter((row) => row.tenure_text)
            .sort((a, b) => parseTenureDays(b.tenure_text) - parseTenureDays(a.tenure_text) || b.total_titles_won - a.total_titles_won || a.manager.localeCompare(b.manager))
            .slice(0, 10);
          const leader = rows[0];
          copy = `Longest-serving coaches in the current World Cup field. ${{
            leader ? `${{leader.manager}} has been in post for ${{leader.tenure_text}}.` : ""
          }}`;
        }} else if (view === "foreign") {{
          rows = rows
            .filter((row) => row.foreign_to_team === true)
            .sort((a, b) => b.total_titles_won - a.total_titles_won || b.age - a.age || a.manager.localeCompare(b.manager))
            .slice(0, 10);
          copy = "Foreign coaches only. This is where the tournament shows which federations imported experience rather than promoting a domestic national figure.";
        }} else if (view === "native") {{
          rows = rows
            .filter((row) => row.foreign_to_team === false)
            .sort((a, b) => b.total_titles_won - a.total_titles_won || b.age - a.age || a.manager.localeCompare(b.manager))
            .slice(0, 10);
          copy = "Native-nationality coaches only. This is the cleaner view if you want to compare homegrown managerial trust.";
        }} else if (view === "former-player") {{
          rows = rows
            .filter((row) => row.former_player === true)
            .sort((a, b) => b.total_titles_won - a.total_titles_won || b.age - a.age || a.manager.localeCompare(b.manager))
            .slice(0, 10);
          copy = "Former-player coaches only. This catches the ex-playing class that moved into the dugout and still dominates the tournament's most decorated bench profile.";
        }}

        coachCopy.textContent = copy;
        coachBody.innerHTML = rows.map((row) => `
          <tr>
            <td>${{row.manager}}<div class="mini">${{row.nationality || "Nationality n/a"}}${{row.foreign_to_team === true ? " • foreign to team" : ""}}</div></td>
            <td>${{row.country}}</td>
            <td class="numeric">${{row.age != null ? row.age.toFixed(0) : "n/a"}}</td>
            <td class="numeric">${{row.total_titles_won}}</td>
            <td>${{row.tenure_text || "n/a"}}</td>
          </tr>
        `).join("");
      }}

      safeBind(squadSearch, "input", () => {{
        squadLimit = 8;
        renderSquadValues();
      }});
      safeBind(marketMoreButton, "click", () => {{
        marketValueLimit += 10;
        renderMarketValues();
      }});
      safeBind(squadMoreButton, "click", () => {{
        squadLimit += 10;
        renderSquadValues();
      }});
      safeBind(globalClubSearch, "input", () => {{
        globalClubLimit = 12;
        renderGlobalClubs();
      }});
      safeBind(globalClubMoreButton, "click", () => {{
        globalClubLimit += 12;
        renderGlobalClubs();
      }});
      safeBind(clubBenefitsSearch, "input", () => {{
        clubBenefitsLimit = 10;
        renderClubBenefits();
      }});
      safeBind(clubBenefitsMoreButton, "click", () => {{
        clubBenefitsLimit += 10;
        renderClubBenefits();
      }});
      safeBind(clubBenefitsCountryView, "click", () => {{
        clubBenefitsView = "country";
        clubBenefitsLimit = 10;
        renderClubBenefits();
      }});
      safeBind(clubBenefitsClubView, "click", () => {{
        clubBenefitsView = "club";
        clubBenefitsLimit = 10;
        renderClubBenefits();
      }});
      safeBind(attackSearch, "input", () => {{
        attackLimit = 10;
        renderAttack();
      }});
      safeBind(attackViewSelect, "change", () => {{
        attackLimit = 10;
        renderAttack();
      }});
      safeBind(attackMoreButton, "click", () => {{
        attackLimit += 10;
        renderAttack();
      }});
      safeBind(coachViewSelect, "change", renderCoachDesk);

      safeRender("market values", renderMarketValues);
      safeRender("squad values", renderSquadValues);
      safeRender("global clubs", renderGlobalClubs);
      safeRender("club benefits", renderClubBenefits);
      safeRender("country clubs", renderCountryClubs);
      safeRender("attack", renderAttack);
      safeRender("coach desk", renderCoachDesk);
    }})();

    (() => {{
      const yearSelect = document.getElementById("distribution-year");
      const ageButton = document.getElementById("distribution-age");
      const heightButton = document.getElementById("distribution-height");
      const excludeGk = document.getElementById("distribution-exclude-gk");
      const copy = document.getElementById("distribution-copy");
      const legend = document.getElementById("distribution-legend");
      const stage = document.getElementById("distribution-stage");
      const tooltip = document.getElementById("heatmap-tooltip");
      let metric = "age";

      function hideTooltip() {{
        tooltip.classList.remove("visible");
      }}

      function positionTooltip(event) {{
        const offset = 14;
        const maxWidth = 320;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const rect = tooltip.getBoundingClientRect();
        let left = event.clientX + offset;
        let top = event.clientY + offset;
        if (left + maxWidth > viewportWidth - 12) {{
          left = Math.max(12, event.clientX - maxWidth - offset);
        }}
        if (top + rect.height > viewportHeight - 12) {{
          top = Math.max(12, event.clientY - rect.height - offset);
        }}
        tooltip.style.left = `${{left}}px`;
        tooltip.style.top = `${{top}}px`;
      }}

      stage.addEventListener("mouseover", (event) => {{
        const cell = event.target.closest(".heatmap-cell");
        if (!cell || !cell.dataset.tooltip) return;
        tooltip.textContent = cell.dataset.tooltip;
        tooltip.classList.add("visible");
        positionTooltip(event);
      }});

      stage.addEventListener("mousemove", (event) => {{
        const cell = event.target.closest(".heatmap-cell");
        if (!cell || !cell.dataset.tooltip) {{
          hideTooltip();
          return;
        }}
        if (!tooltip.classList.contains("visible")) {{
          tooltip.textContent = cell.dataset.tooltip;
          tooltip.classList.add("visible");
        }}
        positionTooltip(event);
      }});

      stage.addEventListener("mouseleave", hideTooltip);
      window.addEventListener("scroll", hideTooltip, {{ passive: true }});
      window.addEventListener("resize", hideTooltip);

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
              const players = team.rows
                .filter((row) => bins[index].test(row))
                .map((row) => row.player)
                .filter(Boolean)
                .join(", ");
              const playerText = players ? ` • Players: ${{players}}` : "";
              const tooltipText = `${{team.country}} • ${{label}} ${{unit}} • ${{count}} player${{count === 1 ? "" : "s"}}${{playerText}}`;
              return `<div class="heatmap-cell" style="background:${{bg}}" data-tooltip="${{tooltipText.replace(/"/g, "&quot;")}}"></div>`;
            }}).join("")}}
          </div>
        `).join("");
        stage.innerHTML = header + rowsHtml;
        hideTooltip();
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
      const hostButton = document.getElementById("tz-host");
      const filterToday = document.getElementById("filter-today");
      const filter24h = document.getElementById("filter-24h");
      const filter3d = document.getElementById("filter-3d");
      const filterAll = document.getElementById("filter-all");
      let timeMode = "utc";
      let fixtureFilter = "all";
      const hostReferenceTimeZone = "America/New_York";

      function localTimeZone() {{
        try {{
          return Intl.DateTimeFormat().resolvedOptions().timeZone || "local time";
        }} catch (error) {{
          return "local time";
        }}
      }}

      function activeTimeZone() {{
        if (timeMode === "local") return undefined;
        if (timeMode === "host") return hostReferenceTimeZone;
        return "UTC";
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
          timeZone: activeTimeZone(),
          timeZoneName: "short",
        }}).format(date);
      }}

      function formatDayHeading(isoText) {{
        const date = new Date(isoText);
        if (Number.isNaN(date.getTime())) return isoText;
        const options = timeMode === "local"
          ? {{ weekday: "long", month: "short", day: "numeric" }}
          : {{ weekday: "long", month: "short", day: "numeric", timeZone: activeTimeZone() }};
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
            : new Intl.DateTimeFormat("en-CA", {{ year: "numeric", month: "2-digit", day: "2-digit", timeZone: activeTimeZone() }});
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
          timeZone: activeTimeZone(),
        }}).format(date);
      }}

      function renderLive() {{
        timezoneLabel.textContent = timeMode === "local"
          ? `Times shown in your computer timezone: ${{localTimeZone()}}.`
          : timeMode === "host"
            ? `Times shown in host reference time: ${{hostReferenceTimeZone}}. This is a single North America reference timezone, not a venue-by-venue local clock.`
            : "Times shown in UTC.";
        utcButton.classList.toggle("active", timeMode === "utc");
        localButton.classList.toggle("active", timeMode === "local");
        hostButton.classList.toggle("active", timeMode === "host");
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
      hostButton.addEventListener("click", () => {{
        timeMode = "host";
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
