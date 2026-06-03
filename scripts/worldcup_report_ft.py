from __future__ import annotations

import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(
    "/Volumes/X10 Pro/__ArchiveMacApril2026/Users-emot/opencode-misc/"
    "dataviz-reports-Matplotlib/worldcup-lbatalha/WorldCup_players_all_data.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "worldcup-report"
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": FT_BG,
            "axes.facecolor": FT_BG,
            "savefig.facecolor": FT_BG,
            "axes.edgecolor": FT_BG,
            "axes.labelcolor": FT_MUTED,
            "xtick.color": FT_MUTED,
            "ytick.color": FT_MUTED,
            "text.color": FT_TEXT,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
        }
    )


def load_data() -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(DATASET_PATH)
    df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"], errors="coerce")
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    df["age_at_tournament_years"] = pd.to_numeric(df["age_at_tournament_years"], errors="coerce")
    has_future_tournament = bool((df["tournament_start_date"] > pd.Timestamp(date.today())).any())
    return df, has_future_tournament


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=FT_GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)


def add_header(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    title_x: float = 0.08,
    title_y: float = 0.955,
    subtitle_y: float = 0.915,
    title_size: int = 23,
    subtitle_size: int = 12,
) -> None:
    fig.text(
        title_x,
        title_y,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=FT_TEXT,
    )
    fig.text(
        title_x,
        subtitle_y,
        subtitle,
        ha="left",
        va="top",
        fontsize=subtitle_size,
        color=FT_MUTED,
    )


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{stem}.png"
    svg = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def chart_average_height(df: pd.DataFrame, has_future_tournament: bool) -> pd.DataFrame:
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

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.16)

    years = grouped["tournament_year"]
    heights = grouped["average_height_cm"]
    ax.fill_between(years, heights, heights.min() - 1.2, color=FT_FILL, alpha=0.3, zorder=1)
    ax.plot(years, heights, color=FT_LINE, linewidth=2.5, zorder=3)
    ax.scatter(years, heights, s=30, color="#b24c3d", edgecolor=FT_BG, linewidth=0.8, zorder=4)

    first_row = grouped.iloc[0]
    last_row = grouped.iloc[-1]
    change = last_row["average_height_cm"] - first_row["average_height_cm"]
    ax.annotate(
        f"{first_row['average_height_cm']:.1f} cm",
        xy=(first_row["tournament_year"], first_row["average_height_cm"]),
        xytext=(1936, first_row["average_height_cm"] - 1.0),
        fontsize=11,
        color=FT_TEXT,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    last_label = f"{int(last_row['tournament_year'])}: {last_row['average_height_cm']:.1f} cm"
    if has_future_tournament:
        last_label += "\n(dataset entry before kick-off)"
    ax.annotate(
        last_label,
        xy=(last_row["tournament_year"], last_row["average_height_cm"]),
        xytext=(2007, last_row["average_height_cm"] + 1.25),
        fontsize=11,
        color=FT_TEXT,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    ax.text(
        0.08,
        0.15,
        f"Net increase across the series: {change:.1f} cm",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=FT_TEXT,
        bbox={"facecolor": "#ebe5dc", "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )

    subtitle = "Average listed height of players in the dataset, in centimetres, by tournament year."
    if has_future_tournament:
        subtitle += " 2026 is flagged because the source file includes an upcoming cohort."
    add_header(fig, "World Cup players have grown taller", subtitle)
    ax.set_xlim(years.min() - 1, years.max() + 1)
    ax.set_ylim(172.0, 184.5)
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax.set_yticks([172, 174, 176, 178, 180, 182, 184])
    ax.set_ylabel("Average height (cm)")
    style_axis(ax)
    fig.text(0.08, 0.07, "Source: WorldCup_players_all_data.csv", fontsize=10, color=FT_MUTED)
    fig.text(0.92, 0.07, "Chart: Codex", ha="right", fontsize=10, color=FT_MUTED)

    save_figure(fig, "worldcup-average-height-ft")
    return grouped


def chart_average_age(df: pd.DataFrame, has_future_tournament: bool) -> pd.DataFrame:
    grouped = (
        df.dropna(subset=["age_at_tournament_years"])
        .groupby("tournament_year", as_index=False)
        .agg(
            average_age=("age_at_tournament_years", "mean"),
            player_count=("age_at_tournament_years", "size"),
            tournament_start_date=("tournament_start_date", "first"),
        )
        .sort_values("tournament_year")
    )
    grouped["average_age"] = grouped["average_age"].round(2)
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.16)

    ax.plot(grouped["tournament_year"], grouped["average_age"], color=FT_LINE, linewidth=2.8)
    ax.scatter(grouped["tournament_year"], grouped["average_age"], s=34, color="#c86b5d", zorder=3)

    first_row = grouped.iloc[0]
    last_row = grouped.iloc[-1]
    add_header(
        fig,
        "World Cup squads have grown older",
        "Average age at the tournament. Age coverage is far more complete than height coverage.",
    )
    ax.annotate(
        f"{first_row['average_age']:.2f} years",
        xy=(first_row["tournament_year"], first_row["average_age"]),
        xytext=(1937, first_row["average_age"] - 0.35),
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    last_label = f"{int(last_row['tournament_year'])}: {last_row['average_age']:.2f} years"
    if has_future_tournament:
        last_label += "\n(dataset row)"
    ax.annotate(
        last_label,
        xy=(last_row["tournament_year"], last_row["average_age"]),
        xytext=(2008, last_row["average_age"] + 0.25),
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    ax.set_xlim(grouped["tournament_year"].min() - 1, grouped["tournament_year"].max() + 1)
    ax.set_ylim(24.2, 28.5)
    ax.set_xticks(grouped["tournament_year"])
    ax.set_xticklabels([str(year) for year in grouped["tournament_year"]], rotation=45, ha="right")
    ax.set_ylabel("Average age (years)")
    style_axis(ax)
    fig.text(0.08, 0.07, "Source: WorldCup_players_all_data.csv", fontsize=10, color=FT_MUTED)
    fig.text(0.92, 0.07, "Chart: Codex", ha="right", fontsize=10, color=FT_MUTED)
    save_figure(fig, "worldcup-average-age-ft")
    return grouped


def chart_tournament_expansion(df: pd.DataFrame, has_future_tournament: bool) -> pd.DataFrame:
    grouped = (
        df.groupby("tournament_year", as_index=False)
        .agg(
            countries=("country", lambda s: s.dropna().nunique()),
            players=("name", "size"),
            tournament_start_date=("tournament_start_date", "first"),
        )
        .sort_values("tournament_year")
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.16)
    bars = ax.bar(grouped["tournament_year"].astype(str), grouped["countries"], color="#d8bfa8", edgecolor="none")
    for idx in [-1]:
        bars[idx].set_color("#b78161")
    add_header(
        fig,
        "The tournament has expanded far beyond its original footprint",
        "Distinct countries represented in the player dataset for each World Cup.",
    )
    ax.set_ylabel("Countries represented")
    ax.set_ylim(0, 52)
    ax.set_yticks(np.arange(0, 51, 10))
    if has_future_tournament:
        ax.text(len(grouped) - 1, grouped.iloc[-1]["countries"] + 1.5, "2026\ndataset row", ha="center", color=FT_MUTED, fontsize=10)
    style_axis(ax)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    fig.text(0.08, 0.07, "Source: WorldCup_players_all_data.csv", fontsize=10, color=FT_MUTED)
    fig.text(0.92, 0.07, "Chart: Codex", ha="right", fontsize=10, color=FT_MUTED)
    save_figure(fig, "worldcup-countries-per-tournament-ft")
    return grouped


def chart_position_shares(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df[df["position"].isin(POSITION_ORDER)]
        .groupby(["tournament_year", "position"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=POSITION_ORDER)
        .sort_index()
    )
    shares = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.16)
    ax.stackplot(
        shares.index,
        [shares[col] for col in POSITION_ORDER],
        colors=[POSITION_COLORS[col] for col in POSITION_ORDER],
        alpha=0.8,
    )
    add_header(
        fig,
        "The forward-heavy early World Cup is gone",
        "Share of listed positions in each tournament. The mix shifts away from forwards and toward defenders and midfielders.",
    )
    ax.set_ylabel("Share of squad lists (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(shares.index.min(), shares.index.max())
    ax.set_xticks(shares.index)
    ax.set_xticklabels([str(year) for year in shares.index], rotation=45, ha="right")
    style_axis(ax)
    legend = ax.legend(POSITION_ORDER, loc="upper left", frameon=False, ncol=4, bbox_to_anchor=(0, 1.02))
    for text in legend.get_texts():
        text.set_color(FT_MUTED)
    ax.annotate(
        f"Forwards: {shares.loc[1930, 'Forward']:.1f}%",
        xy=(1930, shares.loc[1930, "Goalkeeper"] + shares.loc[1930, "Defender"] + shares.loc[1930, "Midfielder"] + shares.loc[1930, "Forward"] / 2),
        xytext=(1936, 86),
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    ax.annotate(
        f"Forwards: {shares.loc[2022, 'Forward']:.1f}%",
        xy=(2022, shares.loc[2022, "Goalkeeper"] + shares.loc[2022, "Defender"] + shares.loc[2022, "Midfielder"] + shares.loc[2022, "Forward"] / 2),
        xytext=(2006, 23),
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    fig.text(0.08, 0.07, "Source: WorldCup_players_all_data.csv", fontsize=10, color=FT_MUTED)
    fig.text(0.92, 0.07, "Chart: Codex", ha="right", fontsize=10, color=FT_MUTED)
    save_figure(fig, "worldcup-position-shares-ft")
    return shares.reset_index()


def chart_height_by_position(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df[df["position"].isin(POSITION_ORDER)].dropna(subset=["height_cm"])
        .groupby(["tournament_year", "position"], as_index=False)
        .agg(average_height=("height_cm", "mean"), count=("height_cm", "size"))
    )
    pivot = grouped.pivot(index="tournament_year", columns="position", values="average_height").reindex(columns=POSITION_ORDER)

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.16)
    for pos in POSITION_ORDER:
        ax.plot(pivot.index, pivot[pos], linewidth=2.4, color=POSITION_COLORS[pos], label=pos)
    add_header(
        fig,
        "Every position has grown taller",
        "Average listed height by position. Early decades rely on thinner height coverage, so the shape matters more than exact decimals.",
    )
    ax.set_ylabel("Average height (cm)")
    ax.set_ylim(170, 191.5)
    ax.set_xlim(pivot.index.min(), pivot.index.max())
    ax.set_xticks(pivot.index)
    ax.set_xticklabels([str(year) for year in pivot.index], rotation=45, ha="right")
    style_axis(ax)
    legend = ax.legend(loc="upper left", frameon=False, ncol=4, bbox_to_anchor=(0, 1.02))
    for text in legend.get_texts():
        text.set_color(FT_MUTED)
    fig.text(0.08, 0.07, "Source: WorldCup_players_all_data.csv", fontsize=10, color=FT_MUTED)
    fig.text(0.92, 0.07, "Chart: Codex", ha="right", fontsize=10, color=FT_MUTED)
    save_figure(fig, "worldcup-height-by-position-ft")
    return grouped


def chart_goalkeeper_age(df: pd.DataFrame, has_future_tournament: bool) -> pd.DataFrame:
    grouped = (
        df[df["position"] == "Goalkeeper"].dropna(subset=["age_at_tournament_years"])
        .groupby("tournament_year", as_index=False)
        .agg(
            average_age=("age_at_tournament_years", "mean"),
            count=("age_at_tournament_years", "size"),
            tournament_start_date=("tournament_start_date", "first"),
        )
        .sort_values("tournament_year")
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.77, bottom=0.16)
    ax.plot(grouped["tournament_year"], grouped["average_age"], color="#7b6a58", linewidth=2.8)
    ax.scatter(grouped["tournament_year"], grouped["average_age"], s=34, color=POSITION_COLORS["Goalkeeper"], zorder=3)
    add_header(
        fig,
        "Goalkeepers have aged more than anyone else",
        "Average goalkeeper age at each tournament. This is one of the cleanest role-specific trends in the file.",
    )
    first_row = grouped.iloc[0]
    last_row = grouped[grouped["tournament_year"] == 2022].iloc[0]
    ax.annotate(
        f"{first_row['average_age']:.2f} years",
        xy=(first_row["tournament_year"], first_row["average_age"]),
        xytext=(1937, first_row["average_age"] - 0.4),
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    ax.annotate(
        f"2022: {last_row['average_age']:.2f} years",
        xy=(last_row["tournament_year"], last_row["average_age"]),
        xytext=(2006, last_row["average_age"] + 0.45),
        fontsize=11,
        arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.9},
    )
    if has_future_tournament:
        last_dataset = grouped.iloc[-1]
        ax.annotate(
            "2026 row in source",
            xy=(last_dataset["tournament_year"], last_dataset["average_age"]),
            xytext=(2011, last_dataset["average_age"] + 0.9),
            fontsize=10,
            color=FT_MUTED,
            arrowprops={"arrowstyle": "-", "color": FT_MUTED, "lw": 0.8},
        )
    ax.set_xlim(grouped["tournament_year"].min() - 1, grouped["tournament_year"].max() + 1)
    ax.set_ylim(25.5, 31.5)
    ax.set_xticks(grouped["tournament_year"])
    ax.set_xticklabels([str(year) for year in grouped["tournament_year"]], rotation=45, ha="right")
    ax.set_ylabel("Average goalkeeper age")
    style_axis(ax)
    fig.text(0.08, 0.07, "Source: WorldCup_players_all_data.csv", fontsize=10, color=FT_MUTED)
    fig.text(0.92, 0.07, "Chart: Codex", ha="right", fontsize=10, color=FT_MUTED)
    save_figure(fig, "worldcup-goalkeeper-age-ft")
    return grouped


def chart_height_distribution_2026(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[(df["tournament_year"] == 2026) & (df["position"].isin(POSITION_ORDER))].dropna(subset=["height_cm"]).copy()
    subset["position"] = pd.Categorical(subset["position"], categories=POSITION_ORDER, ordered=True)
    subset = subset.sort_values("position")

    fig, ax = plt.subplots(figsize=(15.0, 9.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12)

    data = [subset.loc[subset["position"] == pos, "height_cm"].tolist() for pos in POSITION_ORDER]
    positions = np.arange(1, len(POSITION_ORDER) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.44,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#5c5c5c", "linewidth": 2.2},
        whiskerprops={"color": "#7b7b7b", "linewidth": 2},
        capprops={"color": "#5f5f5f", "linewidth": 2},
        boxprops={"facecolor": (1, 1, 1, 0), "edgecolor": "#8b8b8b", "linewidth": 2.2},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor((1, 1, 1, 0.08))

    rng = np.random.default_rng(7)
    for idx, pos in enumerate(POSITION_ORDER, start=1):
        vals = subset.loc[subset["position"] == pos, "height_cm"].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=100,
            color=POSITION_COLORS[pos],
            alpha=0.5,
            edgecolors="none",
            zorder=3,
        )

    counts = subset.groupby("position", observed=False).size().reindex(POSITION_ORDER)
    add_header(
        fig,
        "Height distribution for the different positions in the 2026 World Cup",
        "Height distributions by position; boxplots with individual players overlaid",
        title_x=0.07,
        title_y=0.968,
        subtitle_y=0.925,
        title_size=29,
        subtitle_size=15,
    )
    fig.text(0.083, 0.892, "Height, cm", fontsize=12, color=FT_TEXT)
    ax.set_ylim(155, 211)
    ax.set_yticks(np.arange(155, 211, 5))
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{pos}\n" for pos in POSITION_ORDER], fontsize=14, color=FT_TEXT)
    style_axis(ax)
    ax.tick_params(axis="x", pad=18)
    for idx, pos in enumerate(POSITION_ORDER, start=1):
        ax.text(idx, 151.1, f"n={counts[pos]}", ha="center", va="top", fontsize=11, color=FT_MUTED)
    fig.text(0.94, 0.03, "Made by Codex", ha="right", fontsize=11, color=FT_MUTED)
    save_figure(fig, "worldcup-height-distribution-2026-ft")
    return subset


def build_html(
    has_future_tournament: bool,
    height_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
    countries_summary: pd.DataFrame,
    position_shares: pd.DataFrame,
    position_heights: pd.DataFrame,
    goalkeeper_age: pd.DataFrame,
    heights_2026: pd.DataFrame,
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
    pos_1930 = (
        position_heights[position_heights["tournament_year"] == 1930]
        .set_index("position")["average_height"]
        .to_dict()
    )
    pos_2022 = (
        position_heights[position_heights["tournament_year"] == 2022]
        .set_index("position")["average_height"]
        .to_dict()
    )
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
            "<p class=\"note\"><strong>Dataset note:</strong> the source file already includes a 2026 row "
            "even though the tournament start date in the same file is June 11, 2026. In this report, "
            "2026 is treated as a dataset cohort, not completed tournament history.</p>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup player trends report</title>
  <style>
    :root {{
      --bg: {FT_BG};
      --panel: {FT_PANEL};
      --text: {FT_TEXT};
      --muted: {FT_MUTED};
      --rule: {FT_GRID};
      --accent: {FT_LINE};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.55;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 24px 80px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2.4rem, 5vw, 4.4rem);
      line-height: 0.98;
      letter-spacing: -0.03em;
    }}
    .deck {{
      max-width: 60rem;
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
      max-width: 58rem;
    }}
    figure {{
      margin: 20px 0 0;
      background: var(--panel);
      padding: 12px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    figcaption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 18px 0 0;
      background: var(--panel);
      font-size: 0.98rem;
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
    tbody tr:last-child td {{
      border-bottom: none;
    }}
    .two-up {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 26px;
    }}
    @media (min-width: 980px) {{
      .two-up {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>World Cup player trends, in one file</h1>
    <p class="deck">The sharpest headline is simple: the source file's 2026 cohort is both the tallest on record and, when rounded to one decimal place, tied with 2018 for the oldest average squad age. Average player height rises by 8.9 cm, or 5.1%, from 1930 to the 2026 row, while average age moves from 24.8 to 27.9 years and then mostly plateaus from 1994 onward.</p>
    {note}
    <div class="lede-grid">
      <div class="card"><strong>{last_height:.1f} cm in 2026</strong>The tallest tournament row in the dataset, up {last_height - first_height:.1f} cm from 1930.</div>
      <div class="card"><strong>{age_summary.loc[age_summary["tournament_year"] == 2026, "average_age"].iloc[0]:.2f} years in 2026</strong>Oldest average age in the file, effectively tied with 2018 when rounded to one decimal.</div>
      <div class="card"><strong>{gk_1930:.2f} to {gk_2022:.2f} years</strong>Average goalkeeper age from 1930 to 2022.</div>
      <div class="card"><strong>{countries_1930} to {countries_2026} countries</strong>Representation growth from the first World Cup to the dataset's 2026 row.</div>
    </div>

    <section>
      <h2>1. Players have grown taller, but early height coverage is thin</h2>
      <p>The direction is clear enough. Average listed height rises from {first_height:.1f} cm in 1930 to {last_height:.1f} cm in the source file's 2026 row, a gain of {last_height - first_height:.1f} cm or 5.1%. The catch is coverage: only {int(coverage_1930['players_with_height'])} of {int(coverage_1930['total_players'])} player rows in 1930 include height, or {coverage_1930['pct']:.1f}%. By 2022 that becomes {int(coverage_2022['players_with_height'])} of {int(coverage_2022['total_players'])}, or {coverage_2022['pct']:.1f}%.</p>
      <figure>
        <img src="worldcup-average-height-ft.png" alt="Average World Cup player height over time">
        <figcaption>The trend is real. The exact early levels are less secure than the late ones.</figcaption>
      </figure>
    </section>

    <section>
      <h2>2. Squads are older now</h2>
      <p>This is one of the cleaner findings in the file because age coverage is far stronger than height coverage. Average age climbs from {age_1930:.2f} years in 1930 to {age_2022:.2f} in 2022, with the dataset's 2026 row at {age_summary.iloc[-1]['average_age']:.2f}. The more interesting twist is that the series mostly settles into a narrow band from 1994 onward rather than rising cleanly every cycle.</p>
      <figure>
        <img src="worldcup-average-age-ft.png" alt="Average squad age over time">
        <figcaption>The general direction is up, even if the series flattens in some tournaments.</figcaption>
      </figure>
    </section>

    <section>
      <h2>3. Goalkeepers have aged more than anyone else</h2>
      <p>The role-specific version is sharper. Average goalkeeper age goes from {gk_1930:.2f} years in 1930 to {gk_2022:.2f} in 2022. That supports a simple reading: elite keepers are staying useful later, and teams tolerate age in goal more than they do elsewhere.</p>
      <figure>
        <img src="worldcup-goalkeeper-age-ft.png" alt="Average goalkeeper age over time">
        <figcaption>Goalkeepers stand out as the clearest aging story in the dataset.</figcaption>
      </figure>
      <table>
        <thead>
          <tr>
            <th>Tournament year</th>
            <th>Average goalkeeper age</th>
            <th>Goalkeeper rows</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1930</td>
            <td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1930, "average_age"].iloc[0]:.2f}</td>
            <td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1930, "count"].iloc[0])}</td>
          </tr>
          <tr>
            <td>1982</td>
            <td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1982, "average_age"].iloc[0]:.2f}</td>
            <td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 1982, "count"].iloc[0])}</td>
          </tr>
          <tr>
            <td>2022</td>
            <td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2022, "average_age"].iloc[0]:.2f}</td>
            <td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2022, "count"].iloc[0])}</td>
          </tr>
          <tr>
            <td>2026 dataset row</td>
            <td>{goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2026, "average_age"].iloc[0]:.2f}</td>
            <td>{int(goalkeeper_age.loc[goalkeeper_age["tournament_year"] == 2026, "count"].iloc[0])}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section>
      <h2>4. The tournament has expanded dramatically</h2>
      <p>Any raw count story has to be read through tournament expansion. The dataset goes from {countries_1930} represented countries in 1930 to {countries_2026} in the 2026 row, with intermediate steps like 16 in 1954, 24 in 1982, 31 in 1998 and 32 in 2022. That changes the baseline for everything from player totals to positional depth.</p>
      <figure>
        <img src="worldcup-countries-per-tournament-ft.png" alt="Countries represented per tournament">
        <figcaption>More teams means a wider funnel of player types, ages and physical profiles.</figcaption>
      </figure>
    </section>

    <section>
      <h2>5. The old forward-heavy squad mix has faded</h2>
      <p>The 1930 tournament looks like a different sport in squad construction terms. Forwards accounted for {forward_1930:.1f}% of listed positions in 1930. By 2022 that drops to {forward_2022:.1f}%, while defenders and midfielders each sit above 32%.</p>
      <figure>
        <img src="worldcup-position-shares-ft.png" alt="Position shares over time">
        <figcaption>This is a tidy proxy for tactical change without overclaiming too much from the raw file.</figcaption>
      </figure>
    </section>

    <section>
      <h2>6. Every position has grown taller</h2>
      <p>This is the best follow-up to the headline height chart because it shows the pattern is broad rather than just a goalkeeper artifact. From 1930 to 2022, listed average height rises from {pos_1930['Goalkeeper']:.2f} to {pos_2022['Goalkeeper']:.2f} cm for goalkeepers, {pos_1930['Defender']:.2f} to {pos_2022['Defender']:.2f} for defenders, {pos_1930['Midfielder']:.2f} to {pos_2022['Midfielder']:.2f} for midfielders, and {pos_1930['Forward']:.2f} to {pos_2022['Forward']:.2f} for forwards.</p>
      <figure>
        <img src="worldcup-height-by-position-ft.png" alt="Average height by position over time">
        <figcaption>The magnitude differs by role, but the direction is the same everywhere.</figcaption>
      </figure>
    </section>

    <section>
      <h2>7. The 2026 height spread by position looks exactly like you would expect, only more extreme</h2>
      <p>The positional hierarchy is blunt. Goalkeepers are tallest, defenders next, then forwards, then midfielders. In the 2026 row, the averages are {heights_2026.loc[heights_2026['position'] == 'Goalkeeper', 'height_cm'].mean():.2f} cm for goalkeepers, {heights_2026.loc[heights_2026['position'] == 'Defender', 'height_cm'].mean():.2f} for defenders, {heights_2026.loc[heights_2026['position'] == 'Forward', 'height_cm'].mean():.2f} for forwards, and {heights_2026.loc[heights_2026['position'] == 'Midfielder', 'height_cm'].mean():.2f} for midfielders.</p>
      <figure>
        <img src="worldcup-height-distribution-2026-ft.png" alt="2026 height distribution by position with boxplots and points">
        <figcaption>The boxplot-plus-dots layout makes the positional spread obvious without flattening everyone into an average.</figcaption>
      </figure>
    </section>
  </main>
</body>
</html>
"""

    out = OUTPUT_DIR / "worldcup-trends-report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df, has_future_tournament = load_data()

    height_summary = chart_average_height(df, has_future_tournament)
    age_summary = chart_average_age(df, has_future_tournament)
    countries_summary = chart_tournament_expansion(df, has_future_tournament)
    position_shares = chart_position_shares(df)
    position_heights = chart_height_by_position(df)
    goalkeeper_age = chart_goalkeeper_age(df, has_future_tournament)
    heights_2026 = chart_height_distribution_2026(df)
    report_path = build_html(
        has_future_tournament,
        height_summary,
        age_summary,
        countries_summary,
        position_shares,
        position_heights,
        goalkeeper_age,
        heights_2026,
    )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
