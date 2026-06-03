from __future__ import annotations

import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(
    "/Volumes/X10 Pro/__ArchiveMacApril2026/Users-emot/opencode-misc/"
    "dataviz-reports-Matplotlib/worldcup-lbatalha/WorldCup_players_all_data.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "output" / "worldcup-height"
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FT_BG = "#f3f0ea"
FT_GRID = "#d8d2c8"
FT_TEXT = "#2f2a24"
FT_MUTED = "#6b655f"
FT_LINE = "#8c5a4b"
FT_FILL = "#d6b6aa"
FT_POINT = "#b24c3d"


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
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "Georgia", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
        }
    )


def load_data() -> tuple[pd.DataFrame, bool]:
    today = pd.Timestamp(date.today())
    df = pd.read_csv(DATASET_PATH)
    df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"], errors="coerce")
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    df["age_at_tournament_years"] = pd.to_numeric(df["age_at_tournament_years"], errors="coerce")

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

    has_future_tournament = bool((grouped["tournament_start_date"] > today).any())
    return grouped, has_future_tournament


def add_titles(fig: plt.Figure, has_future_tournament: bool) -> None:
    fig.text(
        0.08,
        0.94,
        "World Cup players have grown taller",
        ha="left",
        va="top",
        fontsize=23,
        fontweight="bold",
        color=FT_TEXT,
    )
    subtitle = (
        "Average listed height of players in the dataset, in centimetres, by tournament year."
    )
    if has_future_tournament:
        subtitle += " 2026 is flagged because the dataset includes an upcoming tournament cohort."
    fig.text(0.08, 0.895, subtitle, ha="left", va="top", fontsize=12, color=FT_MUTED)


def add_footer(fig: plt.Figure) -> None:
    fig.text(0.08, 0.08, "Source: WorldCup_players_all_data.csv", ha="left", fontsize=10, color=FT_MUTED)
    fig.text(0.92, 0.08, "Chart: Codex", ha="right", fontsize=10, color=FT_MUTED)


def build_chart(grouped: pd.DataFrame, has_future_tournament: bool) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.16)

    years = grouped["tournament_year"]
    heights = grouped["average_height_cm"]

    ax.fill_between(years, heights, heights.min() - 1.2, color=FT_FILL, alpha=0.3, zorder=1)
    ax.plot(years, heights, color=FT_LINE, linewidth=2.5, zorder=3)
    ax.scatter(years, heights, s=30, color=FT_POINT, edgecolor=FT_BG, linewidth=0.8, zorder=4)

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
    if has_future_tournament and pd.notna(last_row["tournament_start_date"]):
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

    ax.set_xlim(years.min() - 1, years.max() + 1)
    ax.set_ylim(172.0, 184.5)
    ax.set_xticks(years)
    ax.set_xticklabels([str(year) for year in years], rotation=45, ha="right")
    ax.set_yticks([172, 174, 176, 178, 180, 182, 184])
    ax.set_ylabel("Average height (cm)", labelpad=10)

    ax.grid(axis="y", color=FT_GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    add_titles(fig, has_future_tournament)
    add_footer(fig)

    png_path = OUTPUT_DIR / "worldcup-average-height-ft.png"
    svg_path = OUTPUT_DIR / "worldcup-average-height-ft.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def build_html(grouped: pd.DataFrame, png_path: Path, has_future_tournament: bool) -> Path:
    first_row = grouped.iloc[0]
    last_row = grouped.iloc[-1]
    change = last_row["average_height_cm"] - first_row["average_height_cm"]
    note = ""
    if has_future_tournament and pd.notna(last_row["tournament_start_date"]):
        start = pd.Timestamp(last_row["tournament_start_date"]).strftime("%B %-d, %Y")
        note = (
            f"<p class=\"note\"><strong>Note:</strong> The {int(last_row['tournament_year'])} "
            f"row is included in the source file, but its tournament start date is {start}, "
            "so it should be treated as a dataset cohort rather than completed tournament history.</p>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup Players Height</title>
  <style>
    :root {{
      --bg: {FT_BG};
      --panel: #faf7f2;
      --text: {FT_TEXT};
      --muted: {FT_MUTED};
      --line: {FT_LINE};
      --rule: {FT_GRID};
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}
    h1 {{
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1.05;
      margin: 0 0 10px;
    }}
    .deck {{
      font-size: 1.1rem;
      color: var(--muted);
      margin: 0 0 28px;
      max-width: 52rem;
    }}
    figure {{
      margin: 0;
      background: var(--panel);
      border: 1px solid rgba(0, 0, 0, 0.04);
      padding: 18px 18px 12px;
    }}
    img {{
      width: 100%;
      display: block;
      height: auto;
    }}
    figcaption {{
      margin-top: 10px;
      font-size: 0.95rem;
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 26px 0;
    }}
    .stat {{
      background: var(--panel);
      border-top: 3px solid var(--line);
      padding: 14px 16px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.5rem;
      margin-bottom: 4px;
    }}
    .note {{
      padding: 14px 16px;
      background: #ebe5dc;
      border-left: 4px solid var(--line);
    }}
  </style>
</head>
<body>
  <main>
    <h1>World Cup players have grown taller</h1>
    <p class="deck">Average listed height of players in the dataset has risen steadily from the early tournaments to the modern era, with the series climbing by {change:.1f} cm between {int(first_row["tournament_year"])} and {int(last_row["tournament_year"])}.</p>
    <figure>
      <img src="{png_path.name}" alt="Financial Times style chart showing average height of World Cup players over time">
      <figcaption>Source: WorldCup_players_all_data.csv. Rendered from a local Matplotlib script for easy export to static sites.</figcaption>
    </figure>
    <section class="stats">
      <div class="stat"><strong>{first_row["average_height_cm"]:.1f} cm</strong>{int(first_row["tournament_year"])} average</div>
      <div class="stat"><strong>{last_row["average_height_cm"]:.1f} cm</strong>{int(last_row["tournament_year"])} average</div>
      <div class="stat"><strong>{int(grouped["players_with_height"].sum()):,}</strong>player rows with height data</div>
    </section>
    {note}
  </main>
</body>
</html>
"""

    html_path = OUTPUT_DIR / "worldcup-average-height-ft.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    grouped, has_future_tournament = load_data()
    png_path, _svg_path = build_chart(grouped, has_future_tournament)
    html_path = build_html(grouped, png_path, has_future_tournament)
    csv_path = OUTPUT_DIR / "worldcup-average-height-summary.csv"
    grouped.to_csv(csv_path, index=False)
    print(f"Saved chart to {png_path}")
    print(f"Saved HTML to {html_path}")
    print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    main()
