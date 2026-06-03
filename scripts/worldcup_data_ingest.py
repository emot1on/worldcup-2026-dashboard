from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


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
    return [
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

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_note": (
                "The source file includes a 2026 cohort before kickoff. Treat 2026 as a dataset row, "
                "not completed tournament history."
            ),
            "trend_window_start_year": 1990,
            "comparison_window_note": "Country history baselines prefer 1994, then 1986, then fall back to the latest earlier tournament.",
            "distribution_window_years": [2006, 2010, 2014, 2018, 2022, 2026],
        },
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
    story_manifest = build_story_manifest(tournament_trends, country_snapshot, group_snapshot)
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

    if not any((run_local, run_football_data, run_rest_countries)):
        run_local = True

    written: list[Path] = []

    if run_local:
        written.extend(build_local_layer(args.dataset_path))

    if run_rest_countries:
        try:
            written.append(fetch_rest_countries())
        except (HTTPError, URLError) as exc:
            print(f"REST Countries fetch failed: {exc}")

    if run_football_data:
        token = os.environ.get("FOOTBALL_DATA_API_TOKEN", "").strip()
        if not token:
            print("Skipping football-data.org fetch because FOOTBALL_DATA_API_TOKEN is not set.")
        else:
            try:
                written.extend(fetch_football_data(token))
            except (HTTPError, URLError) as exc:
                print(f"football-data.org fetch failed: {exc}")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
