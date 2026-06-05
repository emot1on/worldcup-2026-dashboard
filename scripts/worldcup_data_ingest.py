from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from worldcup_transfermarkt import build_transfermarkt_enrichment, fetch_transfermarkt_raw, format_money_short


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
LOCAL_DATASET_PATH = Path(
    "/Volumes/X10 Pro/__ArchiveMacApril2026/Users-emot/opencode-misc/"
    "dataviz-reports-Matplotlib/worldcup-lbatalha/WorldCup_players_all_data.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "output" / "dashboard-data"
RAW_DIR = OUTPUT_ROOT / "raw"
REFERENCE_DIR = OUTPUT_ROOT / "reference"
CURATED_DIR = OUTPUT_ROOT / "curated"
NORMALIZED_DIR = OUTPUT_ROOT / "normalized"

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

FOOTBALL_DATA_NAME_ALIASES = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
}

FIFA_WORLD_CUP_2026_RELEASE_DATE = date(2026, 5, 25)
FIFA_WORLD_CUP_2026_APPROX_PLAYER_DAY_USD = 5000

TEAM_TO_GROUP_2026 = {
    team: group
    for group, teams in WORLD_CUP_2026_GROUPS.items()
    for team in teams
}


def ensure_dirs() -> None:
    for path in (RAW_DIR, REFERENCE_DIR, CURATED_DIR, NORMALIZED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_players_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"], errors="coerce")
    df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
    df["age_at_tournament_years"] = pd.to_numeric(df["age_at_tournament_years"], errors="coerce")
    return df


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = CURATED_DIR / filename
    df.to_csv(path, index=False)
    return path


def round_or_none(value: Any, digits: int = 2) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def build_tournament_trends(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("tournament_year", as_index=False)
        .agg(
            average_height_cm=("height_cm", "mean"),
            average_age=("age_at_tournament_years", "mean"),
            player_rows=("name", "size"),
            countries=("country", "nunique"),
            height_rows=("height_cm", lambda s: int(s.notna().sum())),
            age_rows=("age_at_tournament_years", lambda s: int(s.notna().sum())),
        )
        .sort_values("tournament_year")
    )
    grouped["height_coverage_pct"] = (grouped["height_rows"] / grouped["player_rows"] * 100).round(1)
    grouped["age_coverage_pct"] = (grouped["age_rows"] / grouped["player_rows"] * 100).round(1)
    grouped["average_height_cm"] = grouped["average_height_cm"].round(2)
    grouped["average_age"] = grouped["average_age"].round(2)
    return grouped


def choose_history_baseline(country_history: pd.DataFrame) -> pd.Series | None:
    baseline_1994 = country_history[country_history["tournament_year"] == 1994]
    if not baseline_1994.empty:
        return baseline_1994.iloc[0]
    baseline_1986 = country_history[country_history["tournament_year"] == 1986]
    if not baseline_1986.empty:
        return baseline_1986.iloc[0]
    older = country_history[country_history["tournament_year"] < 2026].sort_values(
        "tournament_year", ascending=False
    )
    if older.empty:
        return None
    return older.iloc[0]


def build_country_snapshot_2026(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    country_year = (
        df.groupby(["country", "tournament_year"], as_index=False)
        .agg(
            average_height_cm=("height_cm", "mean"),
            average_age=("age_at_tournament_years", "mean"),
            player_rows=("name", "size"),
        )
        .sort_values(["country", "tournament_year"])
    )
    current = country_year[country_year["tournament_year"] == 2026].copy()
    current["confederation"] = current["country"].map(TEAM_TO_CONFED_2026)
    current["group"] = current["country"].map(TEAM_TO_GROUP_2026)

    history_rows: list[dict[str, Any]] = []
    for country in current["country"].tolist():
        subset = country_year[country_year["country"] == country].copy()
        current_row = subset[subset["tournament_year"] == 2026]
        if current_row.empty:
            continue
        baseline = choose_history_baseline(subset)
        if baseline is None:
            history_rows.append(
                {
                    "country": country,
                    "baseline_tournament_year": None,
                    "baseline_average_height_cm": None,
                    "baseline_average_age": None,
                    "height_delta_cm": None,
                    "age_delta_years": None,
                }
            )
            continue
        now = current_row.iloc[0]
        history_rows.append(
            {
                "country": country,
                "baseline_tournament_year": int(baseline["tournament_year"]),
                "baseline_average_height_cm": round(float(baseline["average_height_cm"]), 2),
                "baseline_average_age": round(float(baseline["average_age"]), 2),
                "height_delta_cm": round(float(now["average_height_cm"] - baseline["average_height_cm"]), 2),
                "age_delta_years": round(float(now["average_age"] - baseline["average_age"]), 2),
            }
        )

    history = pd.DataFrame(history_rows)
    current = current.merge(history, on="country", how="left")
    current["height_rank"] = current["average_height_cm"].rank(method="min", ascending=False).astype(int)
    current["age_rank"] = current["average_age"].rank(method="min", ascending=False).astype(int)
    current = current.sort_values(["height_rank", "country"]).reset_index(drop=True)
    current["average_height_cm"] = current["average_height_cm"].round(2)
    current["average_age"] = current["average_age"].round(2)
    return current, history.sort_values("country").reset_index(drop=True)


def build_group_snapshot_2026(country_snapshot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, teams in WORLD_CUP_2026_GROUPS.items():
        subset = country_snapshot[country_snapshot["country"].isin(teams)].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "group": group,
                "teams": len(teams),
                "player_rows": int(subset["player_rows"].sum()),
                "average_height_cm": round(float(subset["average_height_cm"].mean()), 2),
                "average_age": round(float(subset["average_age"].mean()), 2),
                "tallest_team": subset.sort_values("average_height_cm", ascending=False).iloc[0]["country"],
                "oldest_team": subset.sort_values("average_age", ascending=False).iloc[0]["country"],
            }
        )
    out = pd.DataFrame(rows)
    out["height_rank"] = out["average_height_cm"].rank(method="min", ascending=False).astype(int)
    out["age_rank"] = out["average_age"].rank(method="min", ascending=False).astype(int)
    return out.sort_values("group").reset_index(drop=True)


def build_position_trends(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    position_trends = (
        df.groupby(["tournament_year", "position"], as_index=False)
        .agg(
            average_height_cm=("height_cm", "mean"),
            player_rows=("name", "size"),
        )
        .sort_values(["tournament_year", "position"])
    )
    position_trends["average_height_cm"] = position_trends["average_height_cm"].round(2)

    share = (
        df.groupby(["tournament_year", "position"])
        .size()
        .rename("player_rows")
        .reset_index()
    )
    totals = share.groupby("tournament_year")["player_rows"].transform("sum")
    share["share_pct"] = (share["player_rows"] / totals * 100).round(2)
    share_pivot = (
        share.pivot(index="tournament_year", columns="position", values="share_pct")
        .reset_index()
        .sort_values("tournament_year")
    )
    share_pivot.columns.name = None
    return position_trends, share_pivot


def build_confederation_history(df: pd.DataFrame) -> pd.DataFrame:
    confed_df = df.copy()
    confed_df["confederation"] = confed_df["country"].map(TEAM_TO_CONFED_2026)
    confed_df = confed_df.dropna(subset=["confederation"])
    history = (
        confed_df.groupby(["tournament_year", "confederation"], as_index=False)
        .agg(
            teams=("country", "nunique"),
            player_rows=("name", "size"),
            average_height_cm=("height_cm", "mean"),
            average_age=("age_at_tournament_years", "mean"),
        )
        .sort_values(["tournament_year", "average_height_cm"], ascending=[True, False])
    )
    history["average_height_cm"] = history["average_height_cm"].round(2)
    history["average_age"] = history["average_age"].round(2)
    return history


def build_player_distribution_pool(df: pd.DataFrame) -> list[dict[str, Any]]:
    selected_years = [2006, 2010, 2014, 2018, 2022, 2026]
    subset = df[df["tournament_year"].isin(selected_years)].copy()
    rows: list[dict[str, Any]] = []
    for row in subset.to_dict("records"):
        age_value = round_or_none(row.get("age_at_tournament_years"))
        height_value = round_or_none(row.get("height_cm"))
        rows.append(
            {
                "tournament_year": int(row["tournament_year"]),
                "country": row["country"],
                "position": row["position"],
                "age_at_tournament_years": age_value,
                "height_cm": height_value,
                "name": row["name"],
            }
        )
    return rows


def build_story_manifest(
    tournament_trends: pd.DataFrame,
    country_snapshot: pd.DataFrame,
    group_snapshot: pd.DataFrame,
    transfermarkt_enrichment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    latest = tournament_trends.iloc[-1]
    first = tournament_trends.iloc[0]
    tallest = country_snapshot.sort_values("average_height_cm", ascending=False).iloc[0]
    oldest = country_snapshot.sort_values("average_age", ascending=False).iloc[0]
    toughest_group_by_height = group_snapshot.sort_values("average_height_cm", ascending=False).iloc[0]
    oldest_group = group_snapshot.sort_values("average_age", ascending=False).iloc[0]
    uefa_height = float(country_snapshot[country_snapshot["confederation"] == "UEFA"]["average_height_cm"].mean())
    conmebol_height = float(
        country_snapshot[country_snapshot["confederation"] == "CONMEBOL"]["average_height_cm"].mean()
    )
    stories = [
        {
            "slug": "world-cup-players-have-grown-taller",
            "headline": "World Cup players have grown taller",
            "metric": "average_height_cm",
            "summary": (
                f"Average listed height rises from {first['average_height_cm']:.2f} cm in "
                f"{int(first['tournament_year'])} to {latest['average_height_cm']:.2f} cm in "
                f"{int(latest['tournament_year'])}."
            ),
        },
        {
            "slug": "world-cup-squads-are-older-now",
            "headline": "World Cup squads are older now",
            "metric": "average_age",
            "summary": (
                f"Average age rises from {first['average_age']:.2f} years in {int(first['tournament_year'])} "
                f"to {latest['average_age']:.2f} in {int(latest['tournament_year'])}."
            ),
        },
        {
            "slug": "tallest-team-2026",
            "headline": f"{tallest['country']} is the tallest 2026 squad in the file",
            "metric": "average_height_cm",
            "summary": f"{tallest['country']} averages {tallest['average_height_cm']:.2f} cm.",
        },
        {
            "slug": "oldest-team-2026",
            "headline": f"{oldest['country']} is the oldest 2026 squad in the file",
            "metric": "average_age",
            "summary": f"{oldest['country']} averages {oldest['average_age']:.2f} years.",
        },
        {
            "slug": "group-physical-profile",
            "headline": f"{toughest_group_by_height['group']} is the tallest group on average",
            "metric": "group_average_height_cm",
            "summary": (
                f"{toughest_group_by_height['group']} averages "
                f"{toughest_group_by_height['average_height_cm']:.2f} cm, while "
                f"{oldest_group['group']} is oldest at {oldest_group['average_age']:.2f} years."
            ),
        },
        {
            "slug": "uefa-height-edge",
            "headline": "UEFA enters 2026 with the tallest confederation profile",
            "metric": "confederation_average_height_cm",
            "summary": (
                f"UEFA teams average {uefa_height:.2f} cm, ahead of CONMEBOL at "
                f"{conmebol_height:.2f} cm and the rest of the field."
            ),
        },
        {
            "slug": "old-vs-big-groups",
            "headline": "The oldest groups are not always the tallest ones",
            "metric": "group_average_age",
            "summary": (
                f"{oldest_group['group']} leads on age at {oldest_group['average_age']:.2f} years, "
                f"while {toughest_group_by_height['group']} leads on height."
            ),
        },
        {
            "slug": "country-body-shifts",
            "headline": "Some 2026 teams changed physically far more than others",
            "metric": "height_delta_cm",
            "summary": (
                "The country comparison module shows which squads are much taller or older than their "
                "nearest 1994 or 1986 benchmark, and which ones barely moved."
            ),
        },
    ]
    if transfermarkt_enrichment:
        players = transfermarkt_enrichment.get("players")
        player_stats = transfermarkt_enrichment.get("player_season_stats")
        squad_values = transfermarkt_enrichment.get("squad_values")
        clubs = transfermarkt_enrichment.get("global_clubs")
        coaches = transfermarkt_enrichment.get("coaches")
        if isinstance(players, pd.DataFrame) and not players.empty:
            most_valuable = players.sort_values(["market_value_eur", "player"], ascending=[False, True]).iloc[0]
            stories.extend(
                [
                    {
                        "slug": "most-valuable-player-2026",
                        "headline": f"{most_valuable['player']} is the most valuable player in this 2026 field",
                        "metric": "market_value_eur",
                        "summary": (
                            f"Transfermarkt lists {most_valuable['player']} at "
                            f"{format_money_short(int(most_valuable['market_value_eur']))} for "
                            f"{most_valuable['country']}."
                        ),
                    },
                ]
            )
            low_players = players[
                players["market_value_eur"].notna() & (players["market_value_eur"] > 0)
            ].sort_values(["market_value_eur", "player"], ascending=[True, True])
            if not low_players.empty:
                least_valuable = low_players.iloc[0]
                stories.append(
                    {
                        "slug": "least-valuable-listed-player-2026",
                        "headline": "The value spread inside this field is enormous",
                        "metric": "market_value_eur",
                        "summary": (
                            f"Listed values run from {format_money_short(int(least_valuable['market_value_eur']))} "
                            f"for {least_valuable['player']} of {least_valuable['country']} up to the elite end."
                        ),
                    }
                )
        if isinstance(squad_values, pd.DataFrame) and not squad_values.empty:
            richest = squad_values.sort_values("squad_market_value_eur", ascending=False).iloc[0]
            stories.append(
                {
                    "slug": "most-valuable-squad-2026",
                    "headline": f"{richest['country']} brings the most valuable squad in the field",
                    "metric": "squad_market_value_eur",
                    "summary": (
                        f"Using Transfermarkt squad pages, {richest['country']} leads at "
                        f"{format_money_short(int(richest['squad_market_value_eur']))} in listed player value."
                    ),
                }
            )
        if isinstance(player_stats, pd.DataFrame) and not player_stats.empty:
            scorers = player_stats.sort_values(["goals", "assists", "player"], ascending=[False, False, True])
            assisters = player_stats.sort_values(["assists", "goals", "player"], ascending=[False, False, True])
            top_scorer = scorers.iloc[0]
            top_assister = assisters.iloc[0]
            young_scorer = scorers[scorers["goals"] > 0].sort_values(["age", "goals"], ascending=[True, False]).iloc[0]
            old_assister = assisters[assisters["assists"] > 0].sort_values(["age", "assists"], ascending=[False, False]).iloc[0]
            stories.extend(
                [
                    {
                        "slug": "top-scorer-current-season-2026",
                        "headline": f"{top_scorer['player']} arrives as the top scorer in this squad pool",
                        "metric": "goals",
                        "summary": (
                            f"Using Transfermarkt's player-performance endpoint, {top_scorer['player']} has "
                            f"{int(top_scorer['goals'])} club goals in his current season."
                        ),
                    },
                    {
                        "slug": "top-assister-current-season-2026",
                        "headline": f"{top_assister['player']} leads this field on current-season assists",
                        "metric": "assists",
                        "summary": (
                            f"{top_assister['player']} has {int(top_assister['assists'])} club assists in the current season."
                        ),
                    },
                    {
                        "slug": "young-scorer-current-season-2026",
                        "headline": f"{young_scorer['player']} is the youngest serious scorer in the 2026 field",
                        "metric": "goals",
                        "summary": (
                            f"At {float(young_scorer['age']):.2f}, {young_scorer['player']} already brings "
                            f"{int(young_scorer['goals'])} current-season club goals."
                        ),
                    },
                    {
                        "slug": "old-assister-current-season-2026",
                        "headline": f"{old_assister['player']} is the oldest productive creator in the field",
                        "metric": "assists",
                        "summary": (
                            f"At {float(old_assister['age']):.2f}, {old_assister['player']} still has "
                            f"{int(old_assister['assists'])} current-season club assists."
                        ),
                    },
                ]
            )
        if isinstance(clubs, pd.DataFrame) and not clubs.empty:
            top_club = clubs.sort_values(["player_count", "club"], ascending=[False, True]).iloc[0]
            stories.append(
                {
                    "slug": "club-most-represented-2026",
                    "headline": f"{top_club['club']} sends the most players to this World Cup",
                    "metric": "club_player_count",
                    "summary": (
                        f"{top_club['club']} contributes {int(top_club['player_count'])} players across "
                        f"{int(top_club['represented_countries'])} national teams."
                    ),
                }
            )
        if isinstance(coaches, pd.DataFrame) and not coaches.empty:
            decorated = coaches.sort_values(["total_titles_won", "manager"], ascending=[False, True]).iloc[0]
            stories.append(
                {
                    "slug": "most-decorated-coach-2026",
                    "headline": f"{decorated['manager']} arrives with the deepest trophy cabinet",
                    "metric": "coach_total_titles_won",
                    "summary": (
                        f"Transfermarkt achievement pages credit {decorated['manager']} with "
                        f"{int(decorated['total_titles_won'])} titled honours across "
                        f"{int(decorated['title_types'])} competition types."
                    ),
                }
            )
    return stories


def build_normalized_bundle(
    tournament_trends: pd.DataFrame,
    country_snapshot: pd.DataFrame,
    country_history: pd.DataFrame,
    group_snapshot: pd.DataFrame,
    position_trends: pd.DataFrame,
    position_share_trends: pd.DataFrame,
    confederation_history: pd.DataFrame,
    story_manifest: list[dict[str, Any]],
    player_distribution_pool: list[dict[str, Any]],
    transfermarkt_enrichment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trend_start = tournament_trends[tournament_trends["tournament_year"] >= 1990].copy()
    latest = tournament_trends.iloc[-1]
    first = tournament_trends.iloc[0]
    previous_completed = tournament_trends[tournament_trends["tournament_year"] == 2022].iloc[0]
    oldest_team = country_snapshot.sort_values("average_age", ascending=False).iloc[0]
    tallest_team = country_snapshot.sort_values("average_height_cm", ascending=False).iloc[0]
    oldest_group = group_snapshot.sort_values("average_age", ascending=False).iloc[0]
    tallest_group = group_snapshot.sort_values("average_height_cm", ascending=False).iloc[0]

    country_records = []
    for row in country_snapshot.to_dict("records"):
        country_records.append(
            {
                "country": row["country"],
                "tournament_year": int(row["tournament_year"]),
                "group": row["group"],
                "confederation": row["confederation"],
                "average_height_cm": round(float(row["average_height_cm"]), 2),
                "average_age": round(float(row["average_age"]), 2),
                "player_rows": int(row["player_rows"]),
                "height_rank": int(row["height_rank"]),
                "age_rank": int(row["age_rank"]),
                "baseline_tournament_year": (
                    int(row["baseline_tournament_year"])
                    if not pd.isna(row["baseline_tournament_year"])
                    else None
                ),
                "baseline_average_height_cm": round_or_none(row["baseline_average_height_cm"]),
                "baseline_average_age": round_or_none(row["baseline_average_age"]),
                "height_delta_cm": round_or_none(row["height_delta_cm"]),
                "age_delta_years": round_or_none(row["age_delta_years"]),
            }
        )

    group_members = []
    for group, teams in WORLD_CUP_2026_GROUPS.items():
        members = country_snapshot[country_snapshot["country"].isin(teams)].copy()
        group_members.append(
            {
                "group": group,
                "teams": [
                    {
                        "country": row["country"],
                        "average_height_cm": round(float(row["average_height_cm"]), 2),
                        "average_age": round(float(row["average_age"]), 2),
                        "height_rank": int(row["height_rank"]),
                        "age_rank": int(row["age_rank"]),
                    }
                    for row in members.sort_values("average_height_cm", ascending=False).to_dict("records")
                ],
            }
        )

    confed_summary = (
        country_snapshot.groupby("confederation", as_index=False)
        .agg(
            teams=("country", "nunique"),
            average_height_cm=("average_height_cm", "mean"),
            average_age=("average_age", "mean"),
        )
        .sort_values("average_height_cm", ascending=False)
    )

    source_catalog = [
        {
            "key": "luis_batalha_worldcup_players",
            "label": "Luis Batalha World Cup players dataset",
            "scope": "historical player rows, age, height, positions, and tournament-year comparisons",
            "url": str(LOCAL_DATASET_PATH),
        },
        {
            "key": "football_data_org_worldcup",
            "label": "football-data.org World Cup competition feed",
            "scope": "fixtures, standings, and live-center tournament service",
            "url": "https://docs.football-data.org/general/v4/index.html",
        },
        {
            "key": "fifa_club_benefits_programme_2026",
            "label": "FIFA Club Benefits Programme",
            "scope": "USD 355m overall fund, 25 May 2026 release date, and tournament compensation framework for clubs",
            "url": "https://inside.fifa.com/organisation/media-releases/club-benefits-programme-reward-record-number-of-clubs",
        },
    ]
    if transfermarkt_enrichment and transfermarkt_enrichment.get("source_rows"):
        source_catalog.extend(transfermarkt_enrichment["source_rows"])

    top_market_players = []
    least_market_players = []
    squad_market_values = []
    global_club_counts = []
    country_club_counts = []
    club_benefits_rows = []
    club_benefits_club_rows = []
    coach_rows = []
    coach_title_rows = []
    player_season_rows = []
    transfermarkt_note = None
    if transfermarkt_enrichment:
        players_df = transfermarkt_enrichment.get("players")
        player_stats_df = transfermarkt_enrichment.get("player_season_stats")
        squad_values_df = transfermarkt_enrichment.get("squad_values")
        clubs_df = transfermarkt_enrichment.get("global_clubs")
        country_clubs_df = transfermarkt_enrichment.get("country_clubs")
        club_benefits_df = transfermarkt_enrichment.get("club_benefits")
        club_benefits_clubs_df = transfermarkt_enrichment.get("club_benefits_clubs")
        coaches_df = transfermarkt_enrichment.get("coaches")
        coach_titles_df = transfermarkt_enrichment.get("coach_titles")

        if isinstance(players_df, pd.DataFrame) and not players_df.empty:
            player_count = int(players_df.shape[0])
            country_counts = players_df.groupby("country").size()
            short_teams = [country for country, total in country_counts.items() if int(total) != 26]
            if short_teams:
                transfermarkt_note = (
                    f"Transfermarkt squad pages fetched for this build total {player_count} player rows rather than "
                    f"the full 1,248. The short pages are: {', '.join(short_teams)}."
                )
            else:
                transfermarkt_note = (
                    f"Transfermarkt squad pages fetched for this build total {player_count} player rows across all 48 teams."
                )
            ranked_players = players_df[players_df["market_value_eur"].notna()].copy()
            ranked_players = ranked_players.sort_values(
                ["market_value_eur", "age", "player"], ascending=[False, True, True]
            )
            top_market_players = [
                {
                    "player": row["player"],
                    "country": row["country"],
                    "position": row["position"],
                    "age": round_or_none(row["age"]),
                    "club": row["club"],
                    "market_value_eur": int(row["market_value_eur"]),
                    "market_value_text": row["market_value_text"],
                }
                for row in ranked_players.head(20).to_dict("records")
            ]
            low_players = ranked_players[ranked_players["market_value_eur"] > 0].sort_values(
                ["market_value_eur", "age", "player"], ascending=[True, True, True]
            )
            least_market_players = [
                {
                    "player": row["player"],
                    "country": row["country"],
                    "position": row["position"],
                    "age": round_or_none(row["age"]),
                    "club": row["club"],
                    "market_value_eur": int(row["market_value_eur"]),
                    "market_value_text": row["market_value_text"],
                }
                for row in low_players.head(12).to_dict("records")
            ]
        if isinstance(player_stats_df, pd.DataFrame) and not player_stats_df.empty:
            player_season_rows = [
                {
                    "player": row["player"],
                    "country": row["country"],
                    "club": row["club"],
                    "position": row["position"],
                    "age": round_or_none(row["age"]),
                    "current_season_id": int(row["current_season_id"]),
                    "appearances": int(row["appearances"]),
                    "minutes": int(row["minutes"]),
                    "goals": int(row["goals"]),
                    "assists": int(row["assists"]),
                    "goal_contributions": int(row["goal_contributions"]),
                }
                for row in player_stats_df.sort_values(
                    ["goal_contributions", "goals", "assists", "player"],
                    ascending=[False, False, False, True],
                ).head(50).to_dict("records")
            ]
        if isinstance(squad_values_df, pd.DataFrame) and not squad_values_df.empty:
            squad_market_values = [
                {
                    "country": row["country"],
                    "player_rows": int(row["player_rows"]),
                    "squad_market_value_eur": int(row["squad_market_value_eur"]),
                    "squad_market_value_text": format_money_short(int(row["squad_market_value_eur"])),
                    "average_player_market_value_eur": int(round(float(row["average_player_market_value_eur"]))),
                    "average_player_market_value_text": format_money_short(
                        int(round(float(row["average_player_market_value_eur"])))
                    ),
                    "market_value_rank": int(row["market_value_rank"]),
                }
                for row in squad_values_df.to_dict("records")
            ]
        if isinstance(clubs_df, pd.DataFrame) and not clubs_df.empty:
            global_club_counts = [
                {
                    "club": row["club"],
                    "club_country": row.get("club_country") if pd.notna(row.get("club_country")) else None,
                    "competition_name": (
                        row.get("competition_name") if pd.notna(row.get("competition_name")) else None
                    ),
                    "player_count": int(row["player_count"]),
                    "represented_countries": int(row["represented_countries"]),
                    "total_market_value_eur": int(row["total_market_value_eur"]),
                    "total_market_value_text": format_money_short(int(row["total_market_value_eur"])),
                    "player_count_rank": int(row["player_count_rank"]),
                }
                for row in clubs_df.to_dict("records")
            ]
        if isinstance(country_clubs_df, pd.DataFrame) and not country_clubs_df.empty:
            country_club_counts = [
                {
                    "country": row["country"],
                    "club": row["club"],
                    "club_country": row.get("club_country") if pd.notna(row.get("club_country")) else None,
                    "competition_name": (
                        row.get("competition_name") if pd.notna(row.get("competition_name")) else None
                    ),
                    "player_count": int(row["player_count"]),
                    "country_rank": int(row["country_rank"]),
                }
                for row in country_clubs_df.to_dict("records")
            ]
        if isinstance(club_benefits_df, pd.DataFrame) and not club_benefits_df.empty:
            club_benefits_rows = [
                {
                    "club_country": row["club_country"],
                    "player_count": int(row["player_count"]),
                    "club_count": int(row["club_count"]),
                    "represented_squads": int(row["represented_squads"]),
                    "estimated_floor_usd": int(row["estimated_floor_usd"]),
                    "estimated_ceiling_usd": int(row["estimated_ceiling_usd"]),
                    "estimated_floor_text": row["estimated_floor_text"],
                    "estimated_ceiling_text": row["estimated_ceiling_text"],
                    "ceiling_rank": int(row["ceiling_rank"]),
                }
                for row in club_benefits_df.to_dict("records")
            ]
        if isinstance(club_benefits_clubs_df, pd.DataFrame) and not club_benefits_clubs_df.empty:
            club_benefits_club_rows = [
                {
                    "club": row["club"],
                    "club_country": row["club_country"],
                    "player_count": int(row["player_count"]),
                    "represented_squads": int(row["represented_squads"]),
                    "estimated_floor_usd": int(row["estimated_floor_usd"]),
                    "estimated_ceiling_usd": int(row["estimated_ceiling_usd"]),
                    "estimated_floor_text": row["estimated_floor_text"],
                    "estimated_ceiling_text": row["estimated_ceiling_text"],
                    "ceiling_rank": int(row["ceiling_rank"]),
                }
                for row in club_benefits_clubs_df.to_dict("records")
            ]
        if isinstance(coaches_df, pd.DataFrame) and not coaches_df.empty:
            coach_rows = [
                {
                    "manager": row["manager"],
                    "country": row["country"],
                    "nationality": row["nationality"],
                    "age": round_or_none(row["age"]),
                    "tenure_text": row["tenure_text"],
                    "contract_until": row["contract_until"],
                    "former_player": bool(row["former_player"]),
                    "foreign_to_team": bool(row["foreign_to_team"]) if pd.notna(row["foreign_to_team"]) else None,
                    "total_titles_won": int(row["total_titles_won"]),
                    "title_types": int(row["title_types"]),
                    "honors_preview": row["honors_preview"],
                    "age_rank_oldest": int(row["age_rank_oldest"]),
                    "title_rank": int(row["title_rank"]),
                }
                for row in coaches_df.to_dict("records")
            ]
        if isinstance(coach_titles_df, pd.DataFrame) and not coach_titles_df.empty:
            coach_title_rows = [
                {
                    "manager": row["manager"],
                    "country": row["country"],
                    "nationality": row["nationality"],
                    "title": row["title"],
                    "count": int(row["count"]),
                }
                for row in coach_titles_df.head(80).to_dict("records")
            ]

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_note": (
                "The source file includes a 2026 cohort before kickoff. Treat 2026 as a dataset row, "
                "not completed tournament history."
            ),
            "transfermarkt_note": transfermarkt_note,
            "trend_window_start_year": 1990,
            "comparison_window_note": "Country history baselines prefer 1994, then 1986, then fall back to the latest earlier tournament.",
            "distribution_window_years": [2006, 2010, 2014, 2018, 2022, 2026],
        },
        "sources": source_catalog,
        "highlights": {
            "height_gain_cm_since_1930": round(float(latest["average_height_cm"] - first["average_height_cm"]), 2),
            "height_gain_pct_since_1930": round(
                float((latest["average_height_cm"] - first["average_height_cm"]) / first["average_height_cm"] * 100), 1
            ),
            "age_gain_years_since_1930": round(float(latest["average_age"] - first["average_age"]), 2),
            "latest_average_height_cm": round(float(latest["average_height_cm"]), 2),
            "latest_average_age": round(float(latest["average_age"]), 2),
            "previous_completed_average_age": round(float(previous_completed["average_age"]), 2),
            "tallest_team_2026": tallest_team["country"],
            "oldest_team_2026": oldest_team["country"],
            "tallest_group_2026": tallest_group["group"],
            "oldest_group_2026": oldest_group["group"],
        },
        "market_value_players_2026": top_market_players,
        "least_valuable_players_2026": least_market_players,
        "player_season_stats_2026": player_season_rows,
        "squad_market_values_2026": squad_market_values,
        "global_club_representation_2026": global_club_counts,
        "country_club_representation_2026": country_club_counts,
        "club_benefits_countries_2026": club_benefits_rows,
        "club_benefits_clubs_2026": club_benefits_club_rows,
        "coaches_2026": coach_rows,
        "coach_titles_2026": coach_title_rows,
        "trends": [
            {
                "tournament_year": int(row["tournament_year"]),
                "average_height_cm": round(float(row["average_height_cm"]), 2),
                "average_age": round(float(row["average_age"]), 2),
                "player_rows": int(row["player_rows"]),
                "countries": int(row["countries"]),
                "height_coverage_pct": round(float(row["height_coverage_pct"]), 1),
            }
            for row in trend_start.to_dict("records")
        ],
        "countries_2026": country_records,
        "groups_2026": [
            {
                "group": row["group"],
                "teams": int(row["teams"]),
                "player_rows": int(row["player_rows"]),
                "average_height_cm": round(float(row["average_height_cm"]), 2),
                "average_age": round(float(row["average_age"]), 2),
                "height_rank": int(row["height_rank"]),
                "age_rank": int(row["age_rank"]),
                "tallest_team": row["tallest_team"],
                "oldest_team": row["oldest_team"],
            }
            for row in group_snapshot.to_dict("records")
        ],
        "group_members_2026": group_members,
        "confederations_2026": [
            {
                "confederation": row["confederation"],
                "teams": int(row["teams"]),
                "average_height_cm": round(float(row["average_height_cm"]), 2),
                "average_age": round(float(row["average_age"]), 2),
            }
            for row in confed_summary.to_dict("records")
        ],
        "confederation_history": [
            {
                "tournament_year": int(row["tournament_year"]),
                "confederation": row["confederation"],
                "teams": int(row["teams"]),
                "player_rows": int(row["player_rows"]),
                "average_height_cm": round(float(row["average_height_cm"]), 2),
                "average_age": round(float(row["average_age"]), 2),
            }
            for row in confederation_history.to_dict("records")
        ],
        "country_history_2026": [
            {
                "country": row["country"],
                "baseline_tournament_year": (
                    int(row["baseline_tournament_year"])
                    if not pd.isna(row["baseline_tournament_year"])
                    else None
                ),
                "baseline_average_height_cm": round_or_none(row["baseline_average_height_cm"]),
                "baseline_average_age": round_or_none(row["baseline_average_age"]),
                "height_delta_cm": round_or_none(row["height_delta_cm"]),
                "age_delta_years": round_or_none(row["age_delta_years"]),
            }
            for row in country_history.to_dict("records")
        ],
        "position_trends": [
            {
                "tournament_year": int(row["tournament_year"]),
                "position": row["position"],
                "average_height_cm": round_or_none(row["average_height_cm"]),
                "player_rows": int(row["player_rows"]),
            }
            for row in position_trends[position_trends["tournament_year"] >= 1990].to_dict("records")
        ],
        "position_share_trends": [
            {
                "tournament_year": int(row["tournament_year"]),
                "Goalkeeper": round_or_none(row.get("Goalkeeper")),
                "Defender": round_or_none(row.get("Defender")),
                "Midfielder": round_or_none(row.get("Midfielder")),
                "Forward": round_or_none(row.get("Forward")),
            }
            for row in position_share_trends[position_share_trends["tournament_year"] >= 1990].to_dict("records")
        ],
        "story_manifest": story_manifest,
        "player_distribution_pool": player_distribution_pool,
    }


def fetch_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_wrapped_json(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("data", payload)


def normalize_football_data_team_name(name: str | None) -> str | None:
    if not name:
        return None
    return FOOTBALL_DATA_NAME_ALIASES.get(name, name)


def build_club_benefits_estimates(players_df: pd.DataFrame) -> pd.DataFrame:
    result = build_club_benefits_breakdowns(players_df)
    return result["countries"]


def build_club_benefits_breakdowns(players_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if players_df.empty or "club_country" not in players_df.columns:
        return {"countries": pd.DataFrame(), "clubs": pd.DataFrame()}
    matches_path = RAW_DIR / "football_data" / "matches_wc.json"
    if not matches_path.exists():
        return {"countries": pd.DataFrame(), "clubs": pd.DataFrame()}

    matches_payload = read_wrapped_json(matches_path)
    matches = matches_payload.get("matches", [])
    if not matches:
        return {"countries": pd.DataFrame(), "clubs": pd.DataFrame()}

    last_group_day_by_team: dict[str, date] = {}
    final_window_end = max(
        datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).date() + timedelta(days=1)
        for match in matches
        if match.get("utcDate")
    )
    for match in matches:
        if match.get("stage") != "GROUP_STAGE" or not match.get("utcDate"):
            continue
        match_end = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")).date() + timedelta(days=1)
        for side in ("homeTeam", "awayTeam"):
            team_name = normalize_football_data_team_name(match.get(side, {}).get("name"))
            if not team_name:
                continue
            current = last_group_day_by_team.get(team_name)
            if current is None or match_end > current:
                last_group_day_by_team[team_name] = match_end

    eligible = players_df.dropna(subset=["club", "club_country", "country"]).copy()
    if eligible.empty:
        return {"countries": pd.DataFrame(), "clubs": pd.DataFrame()}

    eligible["group_stage_end_date"] = eligible["country"].map(last_group_day_by_team)
    eligible = eligible[eligible["group_stage_end_date"].notna()].copy()
    if eligible.empty:
        return {"countries": pd.DataFrame(), "clubs": pd.DataFrame()}

    eligible["floor_days"] = eligible["group_stage_end_date"].apply(
        lambda value: int((value - FIFA_WORLD_CUP_2026_RELEASE_DATE).days)
    )
    eligible["ceiling_days"] = int((final_window_end - FIFA_WORLD_CUP_2026_RELEASE_DATE).days)
    eligible["estimated_floor_usd"] = eligible["floor_days"] * FIFA_WORLD_CUP_2026_APPROX_PLAYER_DAY_USD
    eligible["estimated_ceiling_usd"] = eligible["ceiling_days"] * FIFA_WORLD_CUP_2026_APPROX_PLAYER_DAY_USD

    grouped_countries = (
        eligible.groupby("club_country", as_index=False)
        .agg(
            player_count=("player", "size"),
            club_count=("club", "nunique"),
            represented_squads=("country", "nunique"),
            estimated_floor_usd=("estimated_floor_usd", "sum"),
            estimated_ceiling_usd=("estimated_ceiling_usd", "sum"),
        )
        .sort_values(["estimated_ceiling_usd", "player_count", "club_country"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    grouped_countries["ceiling_rank"] = grouped_countries["estimated_ceiling_usd"].rank(method="min", ascending=False).astype(int)
    grouped_countries["estimated_floor_text"] = grouped_countries["estimated_floor_usd"].apply(lambda value: f"${value/1_000_000:.2f}m")
    grouped_countries["estimated_ceiling_text"] = grouped_countries["estimated_ceiling_usd"].apply(lambda value: f"${value/1_000_000:.2f}m")

    grouped_clubs = (
        eligible.groupby(["club", "club_country"], as_index=False)
        .agg(
            player_count=("player", "size"),
            represented_squads=("country", "nunique"),
            estimated_floor_usd=("estimated_floor_usd", "sum"),
            estimated_ceiling_usd=("estimated_ceiling_usd", "sum"),
        )
        .sort_values(["estimated_ceiling_usd", "player_count", "club"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    grouped_clubs["ceiling_rank"] = grouped_clubs["estimated_ceiling_usd"].rank(method="min", ascending=False).astype(int)
    grouped_clubs["estimated_floor_text"] = grouped_clubs["estimated_floor_usd"].apply(lambda value: f"${value/1_000_000:.2f}m")
    grouped_clubs["estimated_ceiling_text"] = grouped_clubs["estimated_ceiling_usd"].apply(lambda value: f"${value/1_000_000:.2f}m")
    return {"countries": grouped_countries, "clubs": grouped_clubs}


def fetch_rest_countries() -> Path:
    data = fetch_json("https://restcountries.com/v3.1/all")
    out = REFERENCE_DIR / "rest_countries.json"
    write_json(data, out)
    return out


def fetch_football_data(token: str) -> list[Path]:
    headers = {"X-Auth-Token": token}
    endpoints = {
        "competition_wc.json": "https://api.football-data.org/v4/competitions/WC",
        "teams_wc.json": "https://api.football-data.org/v4/competitions/WC/teams",
        "matches_wc.json": "https://api.football-data.org/v4/competitions/WC/matches",
        "standings_wc.json": "https://api.football-data.org/v4/competitions/WC/standings",
    }
    written: list[Path] = []
    for filename, url in endpoints.items():
        data = fetch_json(url, headers=headers)
        out = RAW_DIR / "football_data" / filename
        write_json(data, out)
        written.append(out)
    return written


def build_local_layer(dataset_path: Path) -> list[Path]:
    ensure_dirs()
    df = load_players_dataset(dataset_path)
    tournament_trends = build_tournament_trends(df)
    country_snapshot, country_history = build_country_snapshot_2026(df)
    group_snapshot = build_group_snapshot_2026(country_snapshot)
    position_trends, position_share_trends = build_position_trends(df)
    confederation_history = build_confederation_history(df)
    transfermarkt_enrichment = build_transfermarkt_enrichment(RAW_DIR)
    players_df = transfermarkt_enrichment.get("players")
    if isinstance(players_df, pd.DataFrame) and not players_df.empty:
        club_benefits = build_club_benefits_breakdowns(players_df)
        transfermarkt_enrichment["club_benefits"] = club_benefits["countries"]
        transfermarkt_enrichment["club_benefits_clubs"] = club_benefits["clubs"]
    else:
        transfermarkt_enrichment["club_benefits"] = pd.DataFrame()
        transfermarkt_enrichment["club_benefits_clubs"] = pd.DataFrame()
    story_manifest = build_story_manifest(
        tournament_trends,
        country_snapshot,
        group_snapshot,
        transfermarkt_enrichment=transfermarkt_enrichment,
    )
    player_distribution_pool = build_player_distribution_pool(df)
    normalized_bundle = build_normalized_bundle(
        tournament_trends,
        country_snapshot,
        country_history,
        group_snapshot,
        position_trends,
        position_share_trends,
        confederation_history,
        story_manifest,
        player_distribution_pool,
        transfermarkt_enrichment=transfermarkt_enrichment,
    )

    written = [
        write_csv(tournament_trends, "tournament_trends.csv"),
        write_csv(country_snapshot, "country_snapshot_2026.csv"),
        write_csv(country_history, "country_history_2026.csv"),
        write_csv(group_snapshot, "group_snapshot_2026.csv"),
        write_csv(position_trends, "position_trends.csv"),
        write_csv(position_share_trends, "position_share_trends.csv"),
        write_csv(confederation_history, "confederation_history.csv"),
    ]
    if isinstance(players_df, pd.DataFrame) and not players_df.empty:
        written.append(write_csv(players_df, "transfermarkt_players_2026.csv"))
    global_clubs_df = transfermarkt_enrichment.get("global_clubs")
    if isinstance(global_clubs_df, pd.DataFrame) and not global_clubs_df.empty:
        written.append(write_csv(global_clubs_df, "club_representation_global_2026.csv"))
    country_clubs_df = transfermarkt_enrichment.get("country_clubs")
    if isinstance(country_clubs_df, pd.DataFrame) and not country_clubs_df.empty:
        written.append(write_csv(country_clubs_df, "club_representation_by_country_2026.csv"))
    club_benefits_df = transfermarkt_enrichment.get("club_benefits")
    if isinstance(club_benefits_df, pd.DataFrame) and not club_benefits_df.empty:
        written.append(write_csv(club_benefits_df, "club_benefits_country_estimates_2026.csv"))
    club_benefits_clubs_df = transfermarkt_enrichment.get("club_benefits_clubs")
    if isinstance(club_benefits_clubs_df, pd.DataFrame) and not club_benefits_clubs_df.empty:
        written.append(write_csv(club_benefits_clubs_df, "club_benefits_club_estimates_2026.csv"))
    squad_values_df = transfermarkt_enrichment.get("squad_values")
    if isinstance(squad_values_df, pd.DataFrame) and not squad_values_df.empty:
        written.append(write_csv(squad_values_df, "squad_market_values_2026.csv"))
    player_stats_df = transfermarkt_enrichment.get("player_season_stats")
    if isinstance(player_stats_df, pd.DataFrame) and not player_stats_df.empty:
        written.append(write_csv(player_stats_df, "player_season_stats_2026.csv"))
    coaches_df = transfermarkt_enrichment.get("coaches")
    if isinstance(coaches_df, pd.DataFrame) and not coaches_df.empty:
        written.append(write_csv(coaches_df, "coaches_2026.csv"))
    coach_titles_df = transfermarkt_enrichment.get("coach_titles")
    if isinstance(coach_titles_df, pd.DataFrame) and not coach_titles_df.empty:
        written.append(write_csv(coach_titles_df, "coach_titles_2026.csv"))
    manifest_path = CURATED_DIR / "story_manifest.json"
    write_json(story_manifest, manifest_path)
    written.append(manifest_path)
    normalized_bundle_path = NORMALIZED_DIR / "dashboard_bundle.json"
    write_json(normalized_bundle, normalized_bundle_path)
    written.append(normalized_bundle_path)
    countries_json_path = NORMALIZED_DIR / "countries_2026.json"
    write_json(normalized_bundle["countries_2026"], countries_json_path)
    written.append(countries_json_path)
    groups_json_path = NORMALIZED_DIR / "groups_2026.json"
    write_json(normalized_bundle["groups_2026"], groups_json_path)
    written.append(groups_json_path)
    trends_json_path = NORMALIZED_DIR / "trends_since_1990.json"
    write_json(normalized_bundle["trends"], trends_json_path)
    written.append(trends_json_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local World Cup dashboard data and optionally fetch API/reference sources."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=LOCAL_DATASET_PATH,
        help="Path to the World Cup players CSV.",
    )
    parser.add_argument(
        "--build-local",
        action="store_true",
        help="Build curated CSVs from the local players dataset.",
    )
    parser.add_argument(
        "--fetch-football-data",
        action="store_true",
        help="Fetch World Cup competition, teams, matches, and standings from football-data.org.",
    )
    parser.add_argument(
        "--fetch-rest-countries",
        action="store_true",
        help="Fetch country metadata from REST Countries.",
    )
    parser.add_argument(
        "--fetch-transfermarkt",
        action="store_true",
        help="Fetch current Transfermarkt World Cup 2026 participant, squad, manager, and coach achievement pages.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run local build plus any external fetches that have the necessary credentials.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file()
    ensure_dirs()

    run_local = args.build_local or args.all
    run_football_data = args.fetch_football_data or args.all
    run_rest_countries = args.fetch_rest_countries or args.all
    run_transfermarkt = args.fetch_transfermarkt or args.all

    if not any((run_local, run_football_data, run_rest_countries, run_transfermarkt)):
        run_local = True

    written: list[Path] = []

    if run_rest_countries:
        try:
            written.append(fetch_rest_countries())
        except (HTTPError, URLError) as exc:
            print(f"REST Countries fetch failed: {exc}")

    if run_transfermarkt:
        try:
            written.extend(fetch_transfermarkt_raw(RAW_DIR))
        except (HTTPError, URLError) as exc:
            print(f"Transfermarkt fetch failed: {exc}")

    if run_football_data:
        token = os.environ.get("FOOTBALL_DATA_API_TOKEN", "").strip()
        if not token:
            print("Skipping football-data.org fetch because FOOTBALL_DATA_API_TOKEN is not set.")
        else:
            try:
                written.extend(fetch_football_data(token))
            except (HTTPError, URLError) as exc:
                print(f"football-data.org fetch failed: {exc}")

    if run_local:
        written.extend(build_local_layer(args.dataset_path))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
