from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.io import to_html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(
    "/Volumes/X10 Pro/__ArchiveMacApril2026/Users-emot/opencode-misc/"
    "dataviz-reports-Matplotlib/worldcup-lbatalha/WorldCup_players_all_data.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "worldcup-report-interactive"

FT_BG = "#f3f0ea"
FT_PANEL = "#faf7f2"
FT_GRID = "#d8d2c8"
FT_TEXT = "#2f2a24"
FT_MUTED = "#6b655f"
FT_LINE = "#8c5a4b"
FT_FILL = "#d6b6aa"

POSITION_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
POSITION_COLORS = {
    "Goalkeeper": "#8e73b6",
    "Defender": "#ef5a4c",
    "Midfielder": "#f39a35",
    "Forward": "#73c6df",
}

TEAM_TO_ISO3_2026 = {
    "Algeria": "DZA",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Colombia": "COL",
    "Croatia": "HRV",
    "Curaçao": "CUW",
    "Czech Republic": "CZE",
    "DR Congo": "COD",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "France": "FRA",
    "Germany": "DEU",
    "Ghana": "GHA",
    "Haiti": "HTI",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ivory Coast": "CIV",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PRY",
    "Portugal": "PRT",
    "Qatar": "QAT",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "United States": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
}

SPECIAL_TEAM_MARKERS = {
    "England": {"lat": 52.8, "lon": -1.6, "label": "England"},
    "Scotland": {"lat": 56.5, "lon": -4.1, "label": "Scotland"},
}

TEAM_TO_CONFED_2026 = {
    "Algeria": "CAF",
    "Argentina": "CONMEBOL",
    "Australia": "OFC",
    "Austria": "UEFA",
    "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Brazil": "CONMEBOL",
    "Canada": "CONCACAF",
    "Cape Verde": "CAF",
    "Colombia": "CONMEBOL",
    "Croatia": "UEFA",
    "Curaçao": "CONCACAF",
    "Czech Republic": "UEFA",
    "DR Congo": "CAF",
    "Ecuador": "CONMEBOL",
    "Egypt": "CAF",
    "England": "UEFA",
    "France": "UEFA",
    "Germany": "UEFA",
    "Ghana": "CAF",
    "Haiti": "CONCACAF",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Ivory Coast": "CAF",
    "Japan": "AFC",
    "Jordan": "AFC",
    "Mexico": "CONCACAF",
    "Morocco": "CAF",
    "Netherlands": "UEFA",
    "New Zealand": "OFC",
    "Norway": "UEFA",
    "Panama": "CONCACAF",
    "Paraguay": "CONMEBOL",
    "Portugal": "UEFA",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "Scotland": "UEFA",
    "Senegal": "CAF",
    "South Africa": "CAF",
    "South Korea": "AFC",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Tunisia": "CAF",
    "Turkey": "UEFA",
    "United States": "CONCACAF",
    "Uruguay": "CONMEBOL",
    "Uzbekistan": "AFC",
}

WORLD_CUP_2026_GROUPS = {
    "Group A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "Group B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "Group C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "Group D": ["United States", "Paraguay", "Australia", "Turkey"],
    "Group E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "Group F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "Group G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "Group H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "Group I": ["France", "Senegal", "Norway", "Iraq"],
    "Group J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "Group K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "Group L": ["England", "Croatia", "Panama", "Ghana"],
}

HEIGHT_MAP_SCALE = [
    [0.0, "#fde7cf"],
    [0.35, "#f7a85a"],
    [0.7, "#d66a3d"],
    [1.0, "#8c3b2a"],
]

AGE_MAP_SCALE = [
    [0.0, "#d7ecf4"],
    [0.35, "#7ec2dd"],
    [0.7, "#3f8fb4"],
    [1.0, "#1f4f73"],
]


def load_data() -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(DATASET_PATH)
    df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"], errors="coerce")
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    df["age_at_tournament_years"] = pd.to_numeric(df["age_at_tournament_years"], errors="coerce")
    has_future_tournament = bool((df["tournament_start_date"] > pd.Timestamp(date.today())).any())
    return df, has_future_tournament


def base_layout(title: str, subtitle: str, *, height: int = 640, show_legend: bool = False) -> dict:
    return dict(
        title=dict(text=title, x=0, xanchor="left"),
        annotations=[
            dict(
                text=subtitle,
                x=0,
                y=1.09,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="bottom",
                align="left",
                showarrow=False,
                font=dict(size=14, color=FT_MUTED),
            )
        ],
        paper_bgcolor=FT_BG,
        plot_bgcolor=FT_BG,
        font=dict(family="Arial, Helvetica, sans-serif", color=FT_TEXT, size=15),
        margin=dict(l=70, r=30, t=130, b=110),
        height=height,
        hoverlabel=dict(bgcolor=FT_PANEL, font=dict(color=FT_TEXT)),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(color=FT_MUTED),
        ),
        yaxis=dict(
            gridcolor=FT_GRID,
            zeroline=False,
            showline=False,
            tickfont=dict(color=FT_MUTED),
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=FT_MUTED),
        ),
        showlegend=show_legend,
    )


def make_average_height(df: pd.DataFrame, has_future_tournament: bool) -> tuple[go.Figure, pd.DataFrame]:
    grouped = (
        df.dropna(subset=["height_cm"])
        .groupby("tournament_year", as_index=False)
        .agg(
            average_height_cm=("height_cm", "mean"),
            players_with_height=("height_cm", "size"),
            tournament_start_date=("tournament_start_date", "first"),
        )
        .sort_values("tournament_year")
    )
    grouped["average_height_cm"] = grouped["average_height_cm"].round(2)
    grouped["coverage_note"] = grouped["tournament_start_date"].dt.strftime("%Y-%m-%d").fillna("")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["tournament_year"],
            y=grouped["average_height_cm"],
            mode="lines",
            line=dict(color=FT_LINE, width=4),
            fill="tozeroy",
            fillcolor="rgba(214,182,170,0.28)",
            hovertemplate="<b>%{x}</b><br>Average height: %{y:.2f} cm<br>Players with height: %{customdata[0]}<extra></extra>",
            customdata=np.column_stack([grouped["players_with_height"]]),
            name="Average height",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["tournament_year"],
            y=grouped["average_height_cm"],
            mode="markers",
            marker=dict(size=10, color="#b24c3d", line=dict(color=FT_BG, width=1.5)),
            hovertemplate="<b>%{x}</b><br>Average height: %{y:.2f} cm<br>Players with height: %{customdata[0]}<extra></extra>",
            customdata=np.column_stack([grouped["players_with_height"]]),
            showlegend=False,
        )
    )
    subtitle = "Average listed height of players in the dataset, by tournament year."
    if has_future_tournament:
        subtitle += " The 2026 row is flagged because the source file includes an upcoming cohort."
    fig.update_layout(**base_layout("World Cup players have grown taller", subtitle, show_legend=False))
    fig.update_yaxes(title="Average height (cm)", range=[172, 184.8], tickmode="array", tickvals=[172, 174, 176, 178, 180, 182, 184])
    fig.update_xaxes(tickmode="array", tickvals=grouped["tournament_year"], tickangle=-45)
    return fig, grouped


def make_average_age(df: pd.DataFrame, has_future_tournament: bool) -> tuple[go.Figure, pd.DataFrame]:
    grouped = (
        df.dropna(subset=["age_at_tournament_years"])
        .groupby("tournament_year", as_index=False)
        .agg(average_age=("age_at_tournament_years", "mean"), player_count=("age_at_tournament_years", "size"))
        .sort_values("tournament_year")
    )
    grouped["average_age"] = grouped["average_age"].round(2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["tournament_year"],
            y=grouped["average_age"],
            mode="lines+markers",
            line=dict(color=FT_LINE, width=4),
            marker=dict(size=9, color="#c86b5d"),
            hovertemplate="<b>%{x}</b><br>Average age: %{y:.2f} years<br>Player rows: %{customdata[0]}<extra></extra>",
            customdata=np.column_stack([grouped["player_count"]]),
            name="Average age",
        )
    )
    subtitle = "Average age at the tournament. Age coverage is much more complete than height coverage."
    if has_future_tournament:
        subtitle += " The source file also includes a 2026 dataset row."
    fig.update_layout(**base_layout("World Cup squads have grown older", subtitle, show_legend=False))
    fig.update_yaxes(title="Average age (years)", range=[24.2, 28.6])
    fig.update_xaxes(tickmode="array", tickvals=grouped["tournament_year"], tickangle=-45)
    return fig, grouped


def make_goalkeeper_age(df: pd.DataFrame, has_future_tournament: bool) -> tuple[go.Figure, pd.DataFrame]:
    grouped = (
        df[df["position"] == "Goalkeeper"]
        .dropna(subset=["age_at_tournament_years"])
        .groupby("tournament_year", as_index=False)
        .agg(average_age=("age_at_tournament_years", "mean"), count=("age_at_tournament_years", "size"))
        .sort_values("tournament_year")
    )
    grouped["average_age"] = grouped["average_age"].round(2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["tournament_year"],
            y=grouped["average_age"],
            mode="lines+markers",
            line=dict(color="#7b6a58", width=4),
            marker=dict(size=9, color=POSITION_COLORS["Goalkeeper"]),
            hovertemplate="<b>%{x}</b><br>Average goalkeeper age: %{y:.2f} years<br>Goalkeeper rows: %{customdata[0]}<extra></extra>",
            customdata=np.column_stack([grouped["count"]]),
            name="Goalkeeper age",
        )
    )
    subtitle = "Average goalkeeper age at each tournament. This is one of the cleanest role-specific trends in the file."
    if has_future_tournament:
        subtitle += " The file also includes a 2026 dataset row."
    fig.update_layout(**base_layout("Goalkeepers have aged more than anyone else", subtitle, show_legend=False))
    fig.update_yaxes(title="Average goalkeeper age", range=[25.5, 31.6])
    fig.update_xaxes(tickmode="array", tickvals=grouped["tournament_year"], tickangle=-45)
    return fig, grouped


def make_countries(df: pd.DataFrame, has_future_tournament: bool) -> tuple[go.Figure, pd.DataFrame]:
    grouped = (
        df.groupby("tournament_year", as_index=False)
        .agg(countries=("country", lambda s: s.dropna().nunique()), players=("name", "size"))
        .sort_values("tournament_year")
    )
    colors = ["#d8bfa8"] * len(grouped)
    colors[-1] = "#b78161"
    fig = go.Figure(
        go.Bar(
            x=grouped["tournament_year"].astype(str),
            y=grouped["countries"],
            marker=dict(color=colors),
            hovertemplate="<b>%{x}</b><br>Countries represented: %{y}<br>Player rows: %{customdata[0]}<extra></extra>",
            customdata=np.column_stack([grouped["players"]]),
            name="Countries",
        )
    )
    subtitle = "Distinct countries represented in the player dataset for each World Cup."
    if has_future_tournament:
        subtitle += " The 2026 bar comes from the dataset's upcoming-tournament row."
    fig.update_layout(**base_layout("The tournament has expanded dramatically", subtitle, show_legend=False))
    fig.update_yaxes(title="Countries represented", range=[0, 52])
    return fig, grouped


def make_position_shares(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    counts = (
        df[df["position"].isin(POSITION_ORDER)]
        .groupby(["tournament_year", "position"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=POSITION_ORDER)
        .sort_index()
    )
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for pos in POSITION_ORDER:
        fig.add_trace(
            go.Scatter(
                x=shares.index,
                y=shares[pos],
                stackgroup="one",
                mode="lines",
                line=dict(width=0.8, color=POSITION_COLORS[pos]),
                fillcolor=POSITION_COLORS[pos],
                name=pos,
                hovertemplate=f"<b>%{{x}}</b><br>{pos}: %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        **base_layout(
            "The forward-heavy early World Cup is gone",
            "Share of listed positions in each tournament. The mix shifts away from forwards and toward defenders and midfielders.",
            show_legend=True,
        )
    )
    fig.update_yaxes(title="Share of squad lists (%)", range=[0, 100], ticksuffix="%")
    fig.update_xaxes(tickmode="array", tickvals=shares.index, tickangle=-45)
    return fig, shares.reset_index()


def make_height_by_position(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    grouped = (
        df[df["position"].isin(POSITION_ORDER)]
        .dropna(subset=["height_cm"])
        .groupby(["tournament_year", "position"], as_index=False)
        .agg(average_height=("height_cm", "mean"), count=("height_cm", "size"))
    )
    fig = go.Figure()
    for pos in POSITION_ORDER:
        subset = grouped[grouped["position"] == pos].sort_values("tournament_year")
        fig.add_trace(
            go.Scatter(
                x=subset["tournament_year"],
                y=subset["average_height"],
                mode="lines+markers",
                line=dict(color=POSITION_COLORS[pos], width=3),
                marker=dict(size=7),
                name=pos,
                hovertemplate=f"<b>%{{x}}</b><br>{pos}: %{{y:.2f}} cm<br>Height rows: %{{customdata[0]}}<extra></extra>",
                customdata=np.column_stack([subset["count"]]),
            )
        )
    fig.update_layout(
        **base_layout(
            "Every position has grown taller",
            "Average listed height by position. Early decades rely on thinner height coverage, so the shape matters more than exact decimals.",
            show_legend=True,
        )
    )
    fig.update_yaxes(title="Average height (cm)", range=[170, 191.5])
    fig.update_xaxes(tickmode="array", tickvals=sorted(grouped["tournament_year"].unique()), tickangle=-45)
    return fig, grouped


def make_distribution_2026(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    subset = df[(df["tournament_year"] == 2026) & (df["position"].isin(POSITION_ORDER))].dropna(subset=["height_cm"]).copy()
    subset["position"] = pd.Categorical(subset["position"], categories=POSITION_ORDER, ordered=True)
    subset = subset.sort_values("position")

    fig = go.Figure()
    rng = np.random.default_rng(7)
    for pos in POSITION_ORDER:
        vals = subset.loc[subset["position"] == pos, "height_cm"].to_numpy()
        custom = subset.loc[subset["position"] == pos, ["name", "country"]].to_numpy()
        fig.add_trace(
            go.Box(
                y=vals,
                x=[pos] * len(vals),
                name=pos,
                marker_color=POSITION_COLORS[pos],
                line=dict(color="rgba(110,110,110,0.8)", width=2),
                fillcolor="rgba(255,255,255,0.08)",
                boxpoints="all",
                jitter=0.34,
                pointpos=0,
                marker=dict(size=10, opacity=0.5),
                hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>%{customdata[1]}<br>Height: %{y:.0f} cm<extra></extra>",
                customdata=custom,
            )
        )
    fig.update_layout(
        **base_layout(
            "Height distribution for the different positions in the 2026 World Cup",
            "Boxplots with individual players overlaid. Hover points for player-level detail.",
            height=760,
            show_legend=False,
        )
    )
    fig.update_yaxes(title="Height (cm)", range=[155, 211])
    fig.update_xaxes(title="")
    return fig, subset


def make_height_map_2026(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    subset = (
        df[df["tournament_year"] == 2026]
        .groupby("country", as_index=False)
        .agg(
            average_height=("height_cm", "mean"),
            average_age=("age_at_tournament_years", "mean"),
            players=("name", "size"),
        )
        .sort_values("average_height", ascending=False)
    )
    subset["average_height"] = subset["average_height"].round(2)
    subset["average_age"] = subset["average_age"].round(2)
    subset["iso3"] = subset["country"].map(TEAM_TO_ISO3_2026)

    geo_rows = subset[subset["iso3"].notna()].copy()
    special_rows = subset[subset["country"].isin(SPECIAL_TEAM_MARKERS)].copy()
    min_h = subset["average_height"].min()
    max_h = subset["average_height"].max()

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=geo_rows["iso3"],
            z=geo_rows["average_height"],
            text=geo_rows["country"],
            customdata=np.column_stack([geo_rows["average_age"], geo_rows["players"]]),
            colorscale=HEIGHT_MAP_SCALE,
            marker_line_color=FT_BG,
            marker_line_width=0.7,
            colorbar=dict(
                title=dict(text="Avg height (cm)", font=dict(color=FT_MUTED)),
                tickfont=dict(color=FT_MUTED),
                x=0.98,
                thickness=12,
                outlinewidth=0,
            ),
            hovertemplate="<b>%{text}</b><br>Average height: %{z:.2f} cm<br>Average age: %{customdata[0]:.2f} years<br>Player rows: %{customdata[1]}<extra></extra>",
            locationmode="ISO-3",
            zmin=min_h,
            zmax=max_h,
            showscale=True,
            name="Countries",
        )
    )
    for _, row in special_rows.iterrows():
        meta = SPECIAL_TEAM_MARKERS[row["country"]]
        norm = 0 if max_h == min_h else (row["average_height"] - min_h) / (max_h - min_h)
        marker_color = sample_colorscale(HEIGHT_MAP_SCALE, [norm])[0]
        fig.add_trace(
            go.Scattergeo(
                lon=[meta["lon"]],
                lat=[meta["lat"]],
                text=[row["country"]],
                customdata=[[row["average_height"], row["average_age"], row["players"]]],
                mode="markers+text",
                textposition="top center",
                textfont=dict(size=11, color=FT_MUTED),
                marker=dict(size=10, color=marker_color, line=dict(color=FT_BG, width=1)),
                hovertemplate="<b>%{text}</b><br>Average height: %{customdata[0]:.2f} cm<br>Average age: %{customdata[1]:.2f} years<br>Player rows: %{customdata[2]}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        **base_layout(
            "Where the tallest 2026 squads are",
            "Average listed player height by team country in the 2026 cohort. England and Scotland are shown as markers because a sovereign world basemap cannot split the UK into separate football teams.",
            height=690,
            show_legend=False,
        )
    )
    fig.update_layout(
        geo=dict(
            projection_type="equirectangular",
            showframe=False,
            showcoastlines=False,
            showcountries=True,
            countrycolor="#ffffff",
            showland=True,
            landcolor="#f8f5ef",
            bgcolor=FT_BG,
            lataxis_range=[-58, 84],
        )
    )
    return fig, subset


def make_age_map_2026(df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    subset = (
        df[df["tournament_year"] == 2026]
        .groupby("country", as_index=False)
        .agg(
            average_height=("height_cm", "mean"),
            average_age=("age_at_tournament_years", "mean"),
            players=("name", "size"),
        )
        .sort_values("average_age", ascending=False)
    )
    subset["average_height"] = subset["average_height"].round(2)
    subset["average_age"] = subset["average_age"].round(2)
    subset["iso3"] = subset["country"].map(TEAM_TO_ISO3_2026)

    geo_rows = subset[subset["iso3"].notna()].copy()
    special_rows = subset[subset["country"].isin(SPECIAL_TEAM_MARKERS)].copy()
    min_age = subset["average_age"].min()
    max_age = subset["average_age"].max()

    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=geo_rows["iso3"],
            z=geo_rows["average_age"],
            text=geo_rows["country"],
            customdata=np.column_stack([geo_rows["average_height"], geo_rows["players"]]),
            colorscale=AGE_MAP_SCALE,
            marker_line_color=FT_BG,
            marker_line_width=0.7,
            colorbar=dict(
                title=dict(text="Avg age", font=dict(color=FT_MUTED)),
                tickfont=dict(color=FT_MUTED),
                x=0.98,
                thickness=12,
                outlinewidth=0,
            ),
            hovertemplate="<b>%{text}</b><br>Average age: %{z:.2f} years<br>Average height: %{customdata[0]:.2f} cm<br>Player rows: %{customdata[1]}<extra></extra>",
            locationmode="ISO-3",
            zmin=min_age,
            zmax=max_age,
            showscale=True,
            name="Countries",
        )
    )
    for _, row in special_rows.iterrows():
        meta = SPECIAL_TEAM_MARKERS[row["country"]]
        norm = 0 if max_age == min_age else (row["average_age"] - min_age) / (max_age - min_age)
        marker_color = sample_colorscale(AGE_MAP_SCALE, [norm])[0]
        fig.add_trace(
            go.Scattergeo(
                lon=[meta["lon"]],
                lat=[meta["lat"]],
                text=[row["country"]],
                customdata=[[row["average_age"], row["average_height"], row["players"]]],
                mode="markers+text",
                textposition="top center",
                textfont=dict(size=11, color=FT_MUTED),
                marker=dict(size=10, color=marker_color, line=dict(color=FT_BG, width=1)),
                hovertemplate="<b>%{text}</b><br>Average age: %{customdata[0]:.2f} years<br>Average height: %{customdata[1]:.2f} cm<br>Player rows: %{customdata[2]}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        **base_layout(
            "Where the oldest 2026 squads are",
            "Average squad age by team country in the 2026 cohort. England and Scotland are shown as markers because a sovereign world basemap cannot split the UK into separate football teams.",
            height=690,
            show_legend=False,
        )
    )
    fig.update_layout(
        geo=dict(
            projection_type="equirectangular",
            showframe=False,
            showcoastlines=False,
            showcountries=True,
            countrycolor="#ffffff",
            showland=True,
            landcolor="#f8f5ef",
            bgcolor=FT_BG,
            lataxis_range=[-58, 84],
        )
    )
    return fig, subset


def figure_html(fig: go.Figure, include_js: bool) -> str:
    fig.update_layout(modebar=dict(bgcolor="rgba(0,0,0,0)", color=FT_MUTED, activecolor=FT_TEXT))
    return to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "filename": "worldcup-chart", "scale": 2},
        },
    )


def build_country_history_lookup(df: pd.DataFrame) -> dict[str, dict]:
    country_year = (
        df.groupby(["country", "tournament_year"], as_index=False)
        .agg(
            average_height=("height_cm", "mean"),
            average_age=("age_at_tournament_years", "mean"),
        )
        .sort_values(["country", "tournament_year"])
    )
    lookup: dict[str, dict] = {}
    for country in sorted(country_year["country"].dropna().unique()):
        subset = country_year[country_year["country"] == country].copy()
        current = subset[subset["tournament_year"] == 2026]
        if current.empty:
            continue
        historical = subset[subset["tournament_year"] <= 1994]
        if historical.empty:
            continue
        base = historical.sort_values("tournament_year", ascending=False).iloc[0]
        now = current.iloc[0]
        lookup[country] = {
            "base_year": int(base["tournament_year"]),
            "height_delta": round(float(now["average_height"] - base["average_height"]), 2),
            "age_delta": round(float(now["average_age"] - base["average_age"]), 2),
        }
    return lookup


def build_html(
    has_future_tournament: bool,
    height_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
    countries_summary: pd.DataFrame,
    position_shares: pd.DataFrame,
    position_heights: pd.DataFrame,
    goalkeeper_age: pd.DataFrame,
    heights_2026: pd.DataFrame,
    map_2026: pd.DataFrame,
    age_map_2026: pd.DataFrame,
    history_lookup: dict[str, dict],
    charts: list[str],
) -> Path:
    first_height = height_summary.iloc[0]["average_height_cm"]
    last_height = height_summary.iloc[-1]["average_height_cm"]
    age_1930 = age_summary.loc[age_summary["tournament_year"] == 1930, "average_age"].iloc[0]
    age_2022 = age_summary.loc[age_summary["tournament_year"] == 2022, "average_age"].iloc[0]
    gk_1930 = goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1930, "average_age"].iloc[0]
    gk_2022 = goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2022, "average_age"].iloc[0]
    countries_1930 = int(countries_summary.loc[countries_summary["tournament_year"] == 1930, "countries"].iloc[0])
    countries_2026 = int(countries_summary.loc[countries_summary["tournament_year"] == 2026, "countries"].iloc[0])
    forward_1930 = position_shares.loc[position_shares["tournament_year"] == 1930, "Forward"].iloc[0]
    forward_2022 = position_shares.loc[position_shares["tournament_year"] == 2022, "Forward"].iloc[0]
    pos_1930 = position_heights[position_heights["tournament_year"] == 1930].set_index("position")["average_height"].to_dict()
    pos_2022 = position_heights[position_heights["tournament_year"] == 2022].set_index("position")["average_height"].to_dict()
    tallest_2026 = map_2026.iloc[0]
    coverage = (
        pd.read_csv(DATASET_PATH)
        .assign(height_cm=lambda d: pd.to_numeric(d["height_cm"], errors="coerce"))
        .groupby("tournament_year", as_index=False)
        .agg(total_players=("name", "size"), players_with_height=("height_cm", lambda s: s.notna().sum()))
    )
    coverage["pct"] = (coverage["players_with_height"] / coverage["total_players"] * 100).round(1)
    coverage_1930 = coverage.loc[coverage["tournament_year"] == 1930].iloc[0]
    coverage_2022 = coverage.loc[coverage["tournament_year"] == 2022].iloc[0]
    note = ""
    if has_future_tournament:
        note = (
            "<p class='note'><strong>Dataset note:</strong> the source file already includes a 2026 row "
            "even though the same file says the tournament starts on June 11, 2026. This interactive report "
            "treats 2026 as a dataset cohort, not completed tournament history.</p>"
        )

    map_2026 = map_2026.copy()
    map_2026["confed"] = map_2026["country"].map(TEAM_TO_CONFED_2026)
    country_avg_height_rank = map_2026["average_height"].rank(method="min", ascending=False).astype(int)
    country_avg_age_rank = map_2026["average_age"].rank(method="min", ascending=False).astype(int)
    map_2026["height_rank"] = country_avg_height_rank
    map_2026["age_rank"] = country_avg_age_rank

    group_members = {
        **WORLD_CUP_2026_GROUPS,
        "UEFA": sorted([country for country, confed in TEAM_TO_CONFED_2026.items() if confed == "UEFA"]),
        "CONMEBOL": sorted([country for country, confed in TEAM_TO_CONFED_2026.items() if confed == "CONMEBOL"]),
        "CAF": sorted([country for country, confed in TEAM_TO_CONFED_2026.items() if confed == "CAF"]),
        "AFC": sorted([country for country, confed in TEAM_TO_CONFED_2026.items() if confed == "AFC"]),
        "CONCACAF": sorted([country for country, confed in TEAM_TO_CONFED_2026.items() if confed == "CONCACAF"]),
        "OFC": sorted([country for country, confed in TEAM_TO_CONFED_2026.items() if confed == "OFC"]),
        "Hosts": ["Canada", "Mexico", "United States"],
    }
    group_rows = []
    for label, members in group_members.items():
        subset = map_2026[map_2026["country"].isin(members)].copy()
        if subset.empty:
            continue
        group_rows.append(
            {
                "label": label,
                "members": members,
                "members_detail": (
                    subset.sort_values("average_height", ascending=False)[
                        ["country", "average_height", "average_age", "players"]
                    ]
                    .round({"average_height": 2, "average_age": 2})
                    .to_dict("records")
                ),
                "average_height": round(float(subset["average_height"].mean()), 2),
                "average_age": round(float(subset["average_age"].mean()), 2),
                "teams": int(subset["country"].nunique()),
                "players": int(subset["players"].sum()),
            }
        )
    group_rows.sort(key=lambda row: row["average_height"], reverse=True)
    for index, row in enumerate(group_rows, start=1):
        row["height_rank"] = index
    group_rows_by_age = sorted(group_rows, key=lambda row: row["average_age"], reverse=True)
    for index, row in enumerate(group_rows_by_age, start=1):
        row["age_rank"] = index

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup player trends report - interactive</title>
  <style>
    :root {{
      --bg: {FT_BG};
      --panel: {FT_PANEL};
      --text: {FT_TEXT};
      --muted: {FT_MUTED};
      --accent: {FT_LINE};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.55;
      overflow-x: hidden;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 24px 88px;
      overflow-x: clip;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2.4rem, 5vw, 4.4rem);
      line-height: 0.98;
      letter-spacing: -0.03em;
    }}
    .deck {{
      max-width: 64rem;
      color: var(--muted);
      font-size: 1.15rem;
      margin: 0 0 28px;
    }}
    .note {{
      background: #ebe5dc;
      border-left: 4px solid var(--accent);
      padding: 14px 16px;
      margin: 0 0 28px;
    }}
    .lede-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 28px 0 40px;
    }}
    .card {{
      background: var(--panel);
      border-top: 3px solid var(--accent);
      padding: 14px 16px;
    }}
    .card strong {{
      display: block;
      font-size: 1.6rem;
      line-height: 1.05;
      margin-bottom: 6px;
    }}
    section {{
      margin: 44px 0 54px;
      padding-top: 8px;
      border-top: 1px solid rgba(0,0,0,0.06);
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 2rem;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }}
    p {{
      margin: 0 0 14px;
      max-width: 60rem;
    }}
    .chart-wrap {{
      margin: 18px 0 0;
      background: var(--panel);
      padding: 6px 8px 2px;
      border: 1px solid rgba(0,0,0,0.04);
      overflow: hidden;
    }}
    .plotly-graph-div {{
      width: 100% !important;
      max-width: 100% !important;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 18px 0 0;
      background: var(--panel);
      font-size: 0.98rem;
      table-layout: fixed;
    }}
    th, td {{
      text-align: left;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
    }}
    th {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      background: rgba(0,0,0,0.02);
    }}
    th.sortable {{
      cursor: pointer;
      user-select: none;
    }}
    th.sortable:hover {{
      color: var(--text);
      background: rgba(0,0,0,0.04);
    }}
    th.sortable.active {{
      color: var(--text);
    }}
    th.numeric-head {{
      text-align: right;
    }}
    .sort-indicator {{
      display: inline-block;
      min-width: 1.2em;
      margin-left: 6px;
      color: var(--muted);
    }}
    tbody tr:last-child td {{
      border-bottom: none;
    }}
    td.numeric {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    .table-controls {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin: 18px 0 0;
      flex-wrap: wrap;
    }}
    .table-meta {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .show-more-btn {{
      border: 1px solid rgba(0,0,0,0.12);
      background: var(--panel);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .show-more-btn:hover {{
      border-color: rgba(0,0,0,0.22);
      background: #f7f3ec;
    }}
    .tip {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-top: 10px;
    }}
    .compare-intro {{
      margin-top: 24px;
      color: var(--muted);
      font-size: 0.98rem;
    }}
    .compare-toolbar {{
      display: grid;
      grid-template-columns: minmax(260px, 340px) 1fr;
      gap: 18px;
      align-items: start;
      margin-top: 14px;
    }}
    .search-panel {{
      position: relative;
    }}
    .search-input {{
      width: 100%;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid rgba(0,0,0,0.12);
      background: var(--panel);
      color: var(--text);
      font: inherit;
    }}
    .search-input:focus {{
      outline: none;
      border-color: rgba(0,0,0,0.22);
      box-shadow: 0 0 0 3px rgba(140, 90, 75, 0.08);
    }}
    .search-results {{
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      background: var(--panel);
      border: 1px solid rgba(0,0,0,0.1);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.08);
      overflow: hidden;
      display: none;
      z-index: 20;
    }}
    .search-results.visible {{
      display: block;
    }}
    .search-result {{
      width: 100%;
      border: 0;
      background: transparent;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      text-align: left;
      padding: 12px 14px;
      cursor: pointer;
      font: inherit;
      color: var(--text);
    }}
    .search-result:hover {{
      background: rgba(0,0,0,0.03);
    }}
    .search-result-meta {{
      color: var(--muted);
      font-size: 0.92rem;
      white-space: nowrap;
    }}
    .group-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-content: start;
    }}
    .group-pills-block {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .group-pills-label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .group-pill {{
      border: 1px solid rgba(0,0,0,0.12);
      background: var(--panel);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
      white-space: nowrap;
    }}
    .group-pill:hover {{
      border-color: rgba(0,0,0,0.22);
      background: #f7f3ec;
    }}
    .metric-toggle {{
      display: inline-flex;
      gap: 8px;
      margin: 18px 0 0;
      flex-wrap: wrap;
    }}
    .metric-toggle button {{
      border: 1px solid rgba(0,0,0,0.12);
      background: var(--panel);
      color: var(--text);
      padding: 9px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .metric-toggle button.active {{
      background: #ebe5dc;
      border-color: rgba(0,0,0,0.18);
    }}
    .metric-panel {{
      display: none;
    }}
    .metric-panel.active {{
      display: block;
    }}
    .comparison-shell {{
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 18px;
      padding: 18px;
    }}
    .comparison-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .comparison-title {{
      font-size: 1.15rem;
      font-weight: 700;
    }}
    .comparison-subtitle {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .clear-btn {{
      border: 1px solid rgba(0,0,0,0.12);
      background: transparent;
      color: var(--text);
      padding: 9px 14px;
      border-radius: 999px;
      cursor: pointer;
      font: inherit;
    }}
    .comparison-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .compare-card {{
      position: relative;
      min-height: 240px;
      background: #f7f3ec;
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 18px;
      padding: 18px;
    }}
    .compare-card.empty {{
      border-style: dashed;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      text-align: center;
    }}
    .compare-type {{
      display: inline-block;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .compare-name {{
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 2px;
      max-width: calc(100% - 36px);
    }}
    .compare-rank {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-bottom: 16px;
    }}
    .compare-delta {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid rgba(0,0,0,0.08);
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .compare-delta strong {{
      color: var(--text);
    }}
    .compare-remove {{
      position: absolute;
      top: 14px;
      right: 14px;
      border: 0;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 1.3rem;
      line-height: 1;
    }}
    .compare-remove:hover {{
      color: var(--text);
    }}
    .compare-metric {{
      margin-bottom: 12px;
    }}
    .compare-label {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 2px;
    }}
    .compare-value {{
      font-size: 1.05rem;
      font-weight: 700;
    }}
    .compare-members {{
      margin-top: 14px;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
    }}
    .compare-members-table {{
      width: 100%;
      margin-top: 14px;
      background: transparent;
      border-collapse: collapse;
      table-layout: auto;
      font-size: 0.82rem;
    }}
    .compare-members-table th,
    .compare-members-table td {{
      padding: 6px 0;
      border-bottom: 1px solid rgba(0,0,0,0.06);
      background: transparent;
    }}
    .compare-members-table th {{
      font-size: 0.74rem;
      letter-spacing: 0.05em;
      color: var(--muted);
    }}
    .compare-members-table th:first-child,
    .compare-members-table td:first-child {{
      padding-right: 10px;
    }}
    .compare-members-table th.numeric,
    .compare-members-table td.numeric {{
      text-align: right;
      white-space: nowrap;
    }}
    .head-wrap {{
      display: inline-flex;
      align-items: center;
      justify-content: flex-end;
      width: 100%;
      gap: 6px;
    }}
    @media (max-width: 860px) {{
      .compare-toolbar {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>World Cup player trends, interactive</h1>
    <p class="deck">The sharpest headline is simple: the source file's 2026 cohort is both the tallest on record and, when rounded to one decimal place, tied with 2018 for the oldest average squad age. Average player height rises by 8.9 cm, or 5.1%, from 1930 to the 2026 row, while average age moves from 24.8 to 27.9 years and then mostly plateaus from 1994 onward. This version behaves more like a notebook or Datawrapper export: hover for details, zoom into periods, toggle series in legends, and use the modebar to download snapshots.</p>
    {note}
    <div class="lede-grid">
      <div class="card"><strong>{last_height:.1f} cm in 2026</strong>The tallest tournament row in the dataset, up {last_height - first_height:.1f} cm from 1930.</div>
      <div class="card"><strong>{age_summary.loc[age_summary["tournament_year"] == 2026, "average_age"].iloc[0]:.2f} years in 2026</strong>Oldest average age in the file, effectively tied with 2018 when rounded to one decimal.</div>
      <div class="card"><strong>{gk_1930:.2f} to {gk_2022:.2f} years</strong>Average goalkeeper age from 1930 to the completed 2022 tournament.</div>
      <div class="card"><strong>{countries_1930} to {countries_2026} countries</strong>Representation growth from the first World Cup to the dataset's 2026 row.</div>
    </div>

    <section>
      <h2>1. Players have grown taller, but early height coverage is thin</h2>
      <p>Average listed height rises from {first_height:.1f} cm in 1930 to {last_height:.1f} cm in the source file's 2026 row, a gain of {last_height - first_height:.1f} cm or 5.1%. The catch is coverage: only {int(coverage_1930['players_with_height'])} of {int(coverage_1930['total_players'])} player rows in 1930 include height, or {coverage_1930['pct']:.1f}%. By 2022 that becomes {int(coverage_2022['players_with_height'])} of {int(coverage_2022['total_players'])}, or {coverage_2022['pct']:.1f}%.</p>
      <div class="chart-wrap">{charts[0]}</div>
      <p class="tip">Tip: drag to zoom into specific eras; double-click to reset.</p>
    </section>

    <section>
      <h2>2. Squads are older now</h2>
      <p>Average age climbs from {age_1930:.2f} years in 1930 to {age_2022:.2f} in 2022, with the dataset's 2026 row at {age_summary.iloc[-1]['average_age']:.2f}. The more interesting twist is that the series mostly settles into a narrow band from 1994 onward rather than rising cleanly every cycle.</p>
      <div class="chart-wrap">{charts[1]}</div>
    </section>

    <section>
      <h2>3. Goalkeepers have aged more than anyone else</h2>
      <p>Average goalkeeper age goes from {gk_1930:.2f} years in 1930 to {gk_2022:.2f} in 2022. That is one of the strongest role-specific signals in the whole dataset.</p>
      <div class="chart-wrap">{charts[2]}</div>
      <table>
        <thead>
          <tr>
            <th>Tournament year</th>
            <th>Average goalkeeper age</th>
            <th>Goalkeeper rows</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>1930</td><td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1930, "average_age"].iloc[0]:.2f}</td><td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1930, "count"].iloc[0])}</td></tr>
          <tr><td>1982</td><td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1982, "average_age"].iloc[0]:.2f}</td><td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1982, "count"].iloc[0])}</td></tr>
          <tr><td>2022</td><td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2022, "average_age"].iloc[0]:.2f}</td><td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2022, "count"].iloc[0])}</td></tr>
          <tr><td>2026 dataset row</td><td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2026, "average_age"].iloc[0]:.2f}</td><td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2026, "count"].iloc[0])}</td></tr>
        </tbody>
      </table>
    </section>

    <section>
      <h2>4. The tournament has expanded dramatically</h2>
      <p>The dataset goes from {countries_1930} represented countries in 1930 to {countries_2026} in the 2026 row, with major step changes in the middle. Any raw count story has to be read through that expansion.</p>
      <div class="chart-wrap">{charts[3]}</div>
    </section>

    <section>
      <h2>5. The old forward-heavy squad mix has faded</h2>
      <p>Forwards accounted for {forward_1930:.1f}% of listed positions in 1930. By 2022 that drops to {forward_2022:.1f}%, while defenders and midfielders each sit above 32%.</p>
      <div class="chart-wrap">{charts[4]}</div>
      <p class="tip">Tip: click legend items to isolate one position at a time.</p>
    </section>

    <section>
      <h2>6. Every position has grown taller</h2>
      <p>From 1930 to 2022, listed average height rises from {pos_1930['Goalkeeper']:.2f} to {pos_2022['Goalkeeper']:.2f} cm for goalkeepers, {pos_1930['Defender']:.2f} to {pos_2022['Defender']:.2f} for defenders, {pos_1930['Midfielder']:.2f} to {pos_2022['Midfielder']:.2f} for midfielders, and {pos_1930['Forward']:.2f} to {pos_2022['Forward']:.2f} for forwards.</p>
      <div class="chart-wrap">{charts[5]}</div>
    </section>

    <section>
      <h2>7. The 2026 height spread by position looks exactly like you would expect, only more extreme</h2>
      <p>Goalkeepers are tallest, defenders next, then forwards, then midfielders. In the 2026 row, the averages are {heights_2026.loc[heights_2026['position'] == 'Goalkeeper', 'height_cm'].mean():.2f} cm for goalkeepers, {heights_2026.loc[heights_2026['position'] == 'Defender', 'height_cm'].mean():.2f} for defenders, {heights_2026.loc[heights_2026['position'] == 'Forward', 'height_cm'].mean():.2f} for forwards, and {heights_2026.loc[heights_2026['position'] == 'Midfielder', 'height_cm'].mean():.2f} for midfielders.</p>
      <div class="chart-wrap">{charts[6]}</div>
      <p class="tip">Tip: hover individual points to see player names and countries.</p>
    </section>

    <section>
      <h2>8. The size and age advantage is not evenly distributed</h2>
      <p>This section is about both size and experience. Some squads are clearly taller, some are clearly older, and the interesting part is where those two patterns line up or diverge. Use the map toggle to switch between the two lenses.</p>
      <div class="metric-toggle">
        <button type="button" class="active" data-metric-panel="height">Map: average height</button>
        <button type="button" data-metric-panel="age">Map: average age</button>
      </div>
      <div id="metric-panel-height" class="metric-panel active">
        <div class="chart-wrap">{charts[7]}</div>
      </div>
      <div id="metric-panel-age" class="metric-panel">
        <div class="chart-wrap">{charts[8]}</div>
      </div>
      <div class="table-controls">
        <div class="table-meta" id="height-map-table-meta">Showing the 10 tallest teams first.</div>
        <button class="show-more-btn" id="height-map-show-more" type="button">Show more</button>
      </div>
      <table>
        <thead>
          <tr>
            <th class="numeric-head">Rank</th>
            <th>2026 team country</th>
            <th class="sortable active numeric-head" data-sort-key="average_height" data-sort-type="number"><span class="head-wrap"><span>Average height</span><span class="sort-indicator">▼</span></span></th>
            <th class="sortable numeric-head" data-sort-key="average_age" data-sort-type="number"><span class="head-wrap"><span>Average age</span><span class="sort-indicator"></span></span></th>
            <th class="numeric-head">Player rows</th>
          </tr>
        </thead>
        <tbody id="height-map-table-body"></tbody>
      </table>
      <p class="compare-intro">You can also compare up to four 2026 teams or preset groups here. Search for countries, click official groups, or use broader presets like UEFA and CONMEBOL. Country cards also show a 2026 vs roughly 30-to-40-years-ago change when the older baseline exists in the dataset.</p>
      <div class="compare-toolbar">
        <div class="search-panel">
          <input id="compare-search-input" class="search-input" type="text" placeholder="Search 2026 team country...">
          <div id="compare-search-results" class="search-results"></div>
        </div>
        <div class="group-pills-block">
          <div class="group-pills-label">Official Groups</div>
          <div id="compare-group-pills-groups" class="group-pills"></div>
          <div class="group-pills-label">Other Presets</div>
          <div id="compare-group-pills-presets" class="group-pills"></div>
        </div>
      </div>
      <div class="comparison-shell">
        <div class="comparison-header">
          <div>
            <div class="comparison-title">2026 Comparison</div>
            <div class="comparison-subtitle">Mix countries and groups. Maximum 4 cards.</div>
          </div>
          <button id="compare-clear-all" class="clear-btn" type="button">Clear all</button>
        </div>
        <div id="comparison-grid" class="comparison-grid"></div>
      </div>
    </section>
  </main>
  <script>
    const heightMapRows = {json.dumps([
        {
            "country": row["country"],
            "average_height": round(float(row["average_height"]), 2),
            "average_age": round(float(row["average_age"]), 2),
            "players": int(row["players"]),
            "height_rank": int(row["height_rank"]),
            "age_rank": int(row["age_rank"]),
            "history": history_lookup.get(row["country"]),
        }
        for row in map_2026.to_dict("records")
    ])};
    const heightGroupRows = {json.dumps(group_rows)};

    (() => {{
      const tableBody = document.getElementById("height-map-table-body");
      const meta = document.getElementById("height-map-table-meta");
      const showMoreBtn = document.getElementById("height-map-show-more");
      const sortHeaders = Array.from(document.querySelectorAll("th.sortable"));
      const metricButtons = Array.from(document.querySelectorAll(".metric-toggle button"));
      const metricPanels = {{
        height: document.getElementById("metric-panel-height"),
        age: document.getElementById("metric-panel-age"),
      }};
      const initialVisibleRows = 10;
      let visibleRows = initialVisibleRows;
      let currentSortKey = "average_height";
      let currentSortDirection = "desc";

      function formatValue(key, value) {{
        if (key === "average_height") return `${{value.toFixed(2)}} cm`;
        if (key === "average_age") return `${{value.toFixed(2)}} years`;
        return String(value);
      }}

      function sortedRows() {{
        return [...heightMapRows].sort((a, b) => {{
          const dir = currentSortDirection === "asc" ? 1 : -1;
          if (a[currentSortKey] < b[currentSortKey]) return -1 * dir;
          if (a[currentSortKey] > b[currentSortKey]) return 1 * dir;
          return a.country.localeCompare(b.country);
        }});
      }}

      function updateHeaderState() {{
        sortHeaders.forEach((header) => {{
          const key = header.dataset.sortKey;
          const indicator = header.querySelector(".sort-indicator");
          header.classList.toggle("active", key === currentSortKey);
          indicator.textContent = key === currentSortKey ? (currentSortDirection === "desc" ? "▼" : "▲") : "";
        }});
      }}

      function updateMeta(totalRows) {{
        const sortLabel = currentSortKey === "average_height" ? "average height" : "average age";
        const dirLabel = currentSortDirection === "desc" ? "highest to lowest" : "lowest to highest";
        const shown = Math.min(visibleRows, totalRows);
        meta.textContent = `Showing ${{shown}} of ${{totalRows}} teams, sorted by ${{sortLabel}} (${{dirLabel}}).`;
      }}

      function renderTable() {{
        const rows = sortedRows();
        const shownRows = rows.slice(0, visibleRows);
        tableBody.innerHTML = shownRows.map((row, index) => `
          <tr>
            <td class="numeric">${{index + 1}}</td>
            <td>${{row.country}}</td>
            <td class="numeric">${{formatValue("average_height", row.average_height)}}</td>
            <td class="numeric">${{formatValue("average_age", row.average_age)}}</td>
            <td class="numeric">${{row.players}}</td>
          </tr>
        `).join("");
        updateMeta(rows.length);
        updateHeaderState();
        if (visibleRows >= rows.length) {{
          showMoreBtn.textContent = "Show fewer";
        }} else {{
          showMoreBtn.textContent = "Show more";
        }}
      }}

      sortHeaders.forEach((header) => {{
        header.addEventListener("click", () => {{
          const key = header.dataset.sortKey;
          if (currentSortKey === key) {{
            currentSortDirection = currentSortDirection === "desc" ? "asc" : "desc";
          }} else {{
            currentSortKey = key;
            currentSortDirection = "desc";
          }}
          visibleRows = initialVisibleRows;
          renderTable();
        }});
      }});

      showMoreBtn.addEventListener("click", () => {{
        if (visibleRows >= heightMapRows.length) {{
          visibleRows = initialVisibleRows;
        }} else {{
          visibleRows = heightMapRows.length;
        }}
        renderTable();
      }});

      metricButtons.forEach((button) => {{
        button.addEventListener("click", () => {{
          const panel = button.dataset.metricPanel;
          currentSortKey = panel === "age" ? "average_age" : "average_height";
          currentSortDirection = "desc";
          visibleRows = initialVisibleRows;
          metricButtons.forEach((btn) => btn.classList.toggle("active", btn === button));
          Object.entries(metricPanels).forEach(([key, el]) => {{
            el.classList.toggle("active", key === panel);
          }});
          renderTable();
          setTimeout(() => window.dispatchEvent(new Event("resize")), 60);
        }});
      }});

      renderTable();
    }})();

    (() => {{
      const maxCards = 4;
      const searchInput = document.getElementById("compare-search-input");
      const searchResults = document.getElementById("compare-search-results");
      const groupPillsGroups = document.getElementById("compare-group-pills-groups");
      const groupPillsPresets = document.getElementById("compare-group-pills-presets");
      const comparisonGrid = document.getElementById("comparison-grid");
      const clearAllBtn = document.getElementById("compare-clear-all");
      const selected = [];

      const countryEntries = heightMapRows.map((row) => ({{
        kind: "country",
        id: `country:${{row.country}}`,
        name: row.country,
        average_height: row.average_height,
        average_age: row.average_age,
        players: row.players,
        height_rank: row.height_rank,
        age_rank: row.age_rank,
        history: row.history,
      }}));
      const groupEntries = heightGroupRows.map((row) => ({{
        kind: "group",
        id: `group:${{row.label}}`,
        name: row.label,
        average_height: row.average_height,
        average_age: row.average_age,
        players: row.players,
        teams: row.teams,
        members: row.members,
        members_detail: row.members_detail,
        height_rank: row.height_rank,
        age_rank: row.age_rank,
        isOfficialGroup: row.label.startsWith("Group "),
      }}));

      function isSelected(id) {{
        return selected.some((item) => item.id === id);
      }}

      function addItem(item) {{
        if (isSelected(item.id) || selected.length >= maxCards) return;
        selected.push(item);
        searchInput.value = "";
        renderSearchResults([]);
        renderComparison();
      }}

      function removeItem(id) {{
        const index = selected.findIndex((item) => item.id === id);
        if (index >= 0) {{
          selected.splice(index, 1);
          renderComparison();
        }}
      }}

      function clearAll() {{
        selected.splice(0, selected.length);
        renderComparison();
      }}

      function renderSearchResults(items) {{
        if (!items.length) {{
          searchResults.classList.remove("visible");
          searchResults.innerHTML = "";
          return;
        }}
        searchResults.innerHTML = items.map((item) => `
          <button class="search-result" type="button" data-id="${{item.id}}">
            <span>${{item.name}}</span>
            <span class="search-result-meta">${{item.average_height.toFixed(2)}} cm • ${{item.average_age.toFixed(2)}} yrs</span>
          </button>
        `).join("");
        searchResults.classList.add("visible");
        searchResults.querySelectorAll(".search-result").forEach((btn) => {{
          btn.addEventListener("click", () => {{
            const item = countryEntries.find((entry) => entry.id === btn.dataset.id);
            if (item) addItem(item);
          }});
        }});
      }}

      function renderGroupPills() {{
        const official = groupEntries
          .filter((group) => group.isOfficialGroup)
          .sort((a, b) => a.name.localeCompare(b.name, undefined, {{ numeric: true }}));
        const presetOrder = ["Hosts", "UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"];
        const preset = groupEntries
          .filter((group) => !group.isOfficialGroup)
          .sort((a, b) => presetOrder.indexOf(a.name) - presetOrder.indexOf(b.name));
        groupPillsGroups.innerHTML = official.map((group) => `
          <button class="group-pill" type="button" data-id="${{group.id}}">
            + ${{group.name}}
          </button>
        `).join("");
        groupPillsPresets.innerHTML = preset.map((group) => `
          <button class="group-pill" type="button" data-id="${{group.id}}">
            + ${{group.name}}
          </button>
        `).join("");
        document.querySelectorAll(".group-pill").forEach((btn) => {{
          btn.addEventListener("click", () => {{
            const item = groupEntries.find((entry) => entry.id === btn.dataset.id);
            if (item) addItem(item);
          }});
        }});
      }}

      function renderComparison() {{
        const cards = [...selected];
        while (cards.length < maxCards) {{
          cards.push(null);
        }}
        comparisonGrid.innerHTML = cards.map((item) => {{
          if (!item) {{
            return `<div class="compare-card empty">Add a country or group to compare</div>`;
          }}
          const rankLine = item.kind === "country"
            ? `#${{item.height_rank}} by height • #${{item.age_rank}} by age`
            : `#${{item.height_rank}} among groups by height • #${{item.age_rank}} by age`;
          const typeLabel = item.kind === "country" ? "Country" : "Group";
          const members = item.kind === "group"
            ? `<div class="compare-members"><strong>Members:</strong> ${{item.members.join(", ")}}</div>
               <table class="compare-members-table">
                 <thead>
                   <tr>
                     <th>Team</th>
                     <th class="numeric">Height</th>
                     <th class="numeric">Age</th>
                   </tr>
                 </thead>
                 <tbody>
                   ${{item.members_detail.map((member) => `
                     <tr>
                       <td>${{member.country}}</td>
                       <td class="numeric">${{member.average_height.toFixed(2)}} cm</td>
                       <td class="numeric">${{member.average_age.toFixed(2)}} years</td>
                     </tr>
                   `).join("")}}
                 </tbody>
               </table>`
            : "";
          const delta = item.kind === "country" && item.history
            ? `<div class="compare-delta"><strong>${{item.history.base_year}} vs 2026</strong><br>Height: ${{item.history.height_delta > 0 ? "+" : ""}}${{item.history.height_delta.toFixed(2)}} cm<br>Age: ${{item.history.age_delta > 0 ? "+" : ""}}${{item.history.age_delta.toFixed(2)}} years</div>`
            : "";
          const teamsMetric = item.kind === "group"
            ? `<div class="compare-metric"><div class="compare-label">Teams</div><div class="compare-value">${{item.teams}}</div></div>`
            : "";
          return `
            <div class="compare-card">
              <button class="compare-remove" type="button" data-id="${{item.id}}">×</button>
              <div class="compare-type">${{typeLabel}}</div>
              <div class="compare-name">${{item.name}}</div>
              <div class="compare-rank">${{rankLine}}</div>
              <div class="compare-metric"><div class="compare-label">Average height</div><div class="compare-value">${{item.average_height.toFixed(2)}} cm</div></div>
              <div class="compare-metric"><div class="compare-label">Average age</div><div class="compare-value">${{item.average_age.toFixed(2)}} years</div></div>
              <div class="compare-metric"><div class="compare-label">Player rows</div><div class="compare-value">${{item.players}}</div></div>
              ${{teamsMetric}}
              ${{delta}}
              ${{members}}
            </div>
          `;
        }}).join("");
        comparisonGrid.querySelectorAll(".compare-remove").forEach((btn) => {{
          btn.addEventListener("click", () => removeItem(btn.dataset.id));
        }});
      }}

      searchInput.addEventListener("input", () => {{
        const query = searchInput.value.trim().toLowerCase();
        if (!query) {{
          renderSearchResults([]);
          return;
        }}
        const matches = countryEntries
          .filter((item) => !isSelected(item.id) && item.name.toLowerCase().includes(query))
          .slice(0, 8);
        renderSearchResults(matches);
      }});

      searchInput.addEventListener("focus", () => {{
        const query = searchInput.value.trim().toLowerCase();
        if (!query) return;
        const matches = countryEntries
          .filter((item) => !isSelected(item.id) && item.name.toLowerCase().includes(query))
          .slice(0, 8);
        renderSearchResults(matches);
      }});

      document.addEventListener("click", (event) => {{
        if (!searchResults.contains(event.target) && event.target !== searchInput) {{
          renderSearchResults([]);
        }}
      }});

      clearAllBtn.addEventListener("click", clearAll);
      renderGroupPills();
      renderComparison();
    }})();
  </script>
</body>
</html>
"""
    out = OUTPUT_DIR / "worldcup-trends-report-interactive.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, has_future_tournament = load_data()
    height_fig, height_summary = make_average_height(df, has_future_tournament)
    age_fig, age_summary = make_average_age(df, has_future_tournament)
    gk_fig, goalkeeper_age = make_goalkeeper_age(df, has_future_tournament)
    countries_fig, countries_summary = make_countries(df, has_future_tournament)
    shares_fig, position_shares = make_position_shares(df)
    heights_fig, position_heights = make_height_by_position(df)
    dist_fig, heights_2026 = make_distribution_2026(df)
    map_fig, map_2026 = make_height_map_2026(df)
    age_map_fig, age_map_2026 = make_age_map_2026(df)
    history_lookup = build_country_history_lookup(df)

    charts = [
        figure_html(height_fig, include_js=True),
        figure_html(age_fig, include_js=False),
        figure_html(gk_fig, include_js=False),
        figure_html(countries_fig, include_js=False),
        figure_html(shares_fig, include_js=False),
        figure_html(heights_fig, include_js=False),
        figure_html(dist_fig, include_js=False),
        figure_html(map_fig, include_js=False),
        figure_html(age_map_fig, include_js=False),
    ]
    report_path = build_html(
        has_future_tournament,
        height_summary,
        age_summary,
        countries_summary,
        position_shares,
        position_heights,
        goalkeeper_age,
        heights_2026,
        map_2026,
        age_map_2026,
        history_lookup,
        charts,
    )
    print(f"Saved interactive report to {report_path}")


if __name__ == "__main__":
    main()
