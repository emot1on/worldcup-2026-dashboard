from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup


BASE_URL = "https://www.transfermarkt.com"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
CURRENT_SEASON_ID = 2026

TM_COUNTRY_TO_DASHBOARD = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Czechia": "Czech Republic",
    "Democratic Republic of the Congo": "DR Congo",
    "South Korea": "South Korea",
    "Turkiye": "Turkey",
}


@dataclass
class TransfermarktPaths:
    root: Path

    @property
    def participants_html(self) -> Path:
        return self.root / "participants_2026.html"

    @property
    def managers_html(self) -> Path:
        return self.root / "managers_2026.html"

    @property
    def squads_dir(self) -> Path:
        return self.root / "squads_2026"

    @property
    def coach_achievements_dir(self) -> Path:
        return self.root / "coach_achievements_2026"

    @property
    def player_performance_dir(self) -> Path:
        return self.root / "player_performance_2026"

    @property
    def fetch_manifest_json(self) -> Path:
        return self.root / "fetch_manifest_2026.json"


def normalize_country_name(name: str) -> str:
    cleaned = " ".join(name.split())
    return TM_COUNTRY_TO_DASHBOARD.get(cleaned, cleaned)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def extract_id_from_href(href: str | None, marker: str) -> int | None:
    if not href:
        return None
    match = re.search(rf"/{re.escape(marker)}/(\d+)", href)
    if not match:
        return None
    return int(match.group(1))


def parse_money_to_eur(value: str | None) -> int | None:
    if not value:
        return None
    cleaned = value.replace("\xa0", " ").strip()
    if cleaned in {"-", "?", ""}:
        return None
    cleaned = cleaned.replace("€", "").replace(",", "")
    multiplier = 1
    suffix = cleaned[-1].lower()
    numeric = cleaned
    if suffix == "m":
        multiplier = 1_000_000
        numeric = cleaned[:-1]
    elif suffix == "k":
        multiplier = 1_000
        numeric = cleaned[:-1]
    try:
        return int(round(float(numeric) * multiplier))
    except ValueError:
        return None


def format_money_short(value: int | None) -> str | None:
    if value is None:
        return None
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.2f}m"
    if value >= 1_000:
        return f"€{value / 1_000:.0f}k"
    return f"€{value}"


def fetch_html(url: str) -> str:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_json(url: str) -> Any:
    request = Request(url, headers={**DEFAULT_HEADERS, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def fetch_transfermarkt_raw(raw_dir: Path) -> list[Path]:
    paths = TransfermarktPaths(raw_dir / "transfermarkt")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.squads_dir.mkdir(parents=True, exist_ok=True)
    paths.coach_achievements_dir.mkdir(parents=True, exist_ok=True)
    paths.player_performance_dir.mkdir(parents=True, exist_ok=True)

    participants_url = "https://www.transfermarkt.com/weltmeisterschaft/teilnehmer/pokalwettbewerb/FIWC"
    managers_url = "https://www.transfermarkt.com/weltmeisterschaft/trainer/pokalwettbewerb/FIWC"

    participants_html = fetch_html(participants_url)
    write_text(paths.participants_html, participants_html)
    manager_html = fetch_html(managers_url)
    write_text(paths.managers_html, manager_html)

    participants = parse_participants_html(participants_html)
    managers = parse_manager_overview_html(manager_html)
    written = [paths.participants_html, paths.managers_html]

    for participant in participants:
        html = fetch_html(participant["squad_url"])
        filename = f"{slugify(participant['country'])}.html"
        out = paths.squads_dir / filename
        write_text(out, html)
        written.append(out)

    for manager in managers:
        achievements_url = manager["achievements_url"]
        html = fetch_html(achievements_url)
        manager_id = manager.get("manager_id") or slugify(manager["manager"])
        out = paths.coach_achievements_dir / f"{manager_id}.html"
        write_text(out, html)
        written.append(out)

    players = []
    for participant in participants:
        filename = paths.squads_dir / f"{slugify(participant['country'])}.html"
        if not filename.exists():
            continue
        players.extend(
            parse_squad_html(
                filename.read_text(encoding="utf-8"),
                country=participant["country"],
                tm_country=participant["tm_country"],
            )
        )

    unique_player_ids = sorted(
        {int(player["player_id"]) for player in players if player.get("player_id") is not None}
    )

    def fetch_performance(player_id: int) -> Path:
        out = paths.player_performance_dir / f"{player_id}.json"
        if out.exists():
            return out
        data = fetch_json(f"{BASE_URL}/ceapi/performance-game/{player_id}")
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_performance, player_id) for player_id in unique_player_ids]
        for future in as_completed(futures):
            written.append(future.result())

    manifest = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {"kind": "participants", "url": participants_url},
            {"kind": "managers", "url": managers_url},
        ],
        "participants": participants,
        "managers": [
            {
                "manager": manager["manager"],
                "country": manager["country"],
                "profile_url": manager["profile_url"],
                "achievements_url": manager["achievements_url"],
            }
            for manager in managers
        ],
    }
    paths.fetch_manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(paths.fetch_manifest_json)
    return written


def parse_participants_html(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if table is None:
        return []
    links: dict[str, str] = {}
    for tr in table.find_all("tr", class_=["odd", "even"]):
        anchor = tr.find("a", href=True)
        if anchor is None:
            continue
        href = anchor["href"]
        if "/startseite/verein/" not in href:
            continue
        text = anchor.get("title") or " ".join(anchor.get_text(" ", strip=True).split())
        if not text:
            continue
        absolute = urljoin(BASE_URL, href)
        links[absolute] = text

    participants: list[dict[str, Any]] = []
    for profile_url, tm_country in sorted(links.items(), key=lambda item: item[1]):
        squad_url = profile_url.replace("/startseite/", "/kader/") + f"/saison_id/{CURRENT_SEASON_ID}"
        participants.append(
            {
                "tm_country": tm_country,
                "country": normalize_country_name(tm_country),
                "profile_url": profile_url,
                "squad_url": squad_url,
            }
        )
    return participants


def parse_manager_overview_html(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if table is None:
        return []
    rows: list[dict[str, Any]] = []
    for tr in table.find_all("tr", class_=["odd", "even"]):
        tds = tr.find_all("td")
        if len(tds) < 8:
            continue
        profile_link = tds[1].find("a", href=True)
        if profile_link is None:
            continue
        manager_name = " ".join(profile_link.get_text(" ", strip=True).split())
        profile_href = profile_link["href"]
        profile_url = urljoin(BASE_URL, profile_href)
        nationality_img = tds[2].find("img")
        team_link = tds[3].find("a", href=True)
        team_img = tds[3].find("img")
        former_player_link = tds[7].find("a", href=True)
        achievements_url = profile_url.replace("/profil/trainer/", "/erfolge/trainer/")
        rows.append(
            {
                "manager": manager_name,
                "manager_id": extract_id_from_href(profile_href, "trainer"),
                "profile_url": profile_url,
                "achievements_url": achievements_url,
                "nationality": nationality_img.get("title") if nationality_img else None,
                "tm_country": team_img.get("title") if team_img else None,
                "country": normalize_country_name(team_img.get("title", "")) if team_img else None,
                "team_profile_url": urljoin(BASE_URL, team_link["href"]) if team_link else None,
                "age": pd.to_numeric(tds[4].get_text(strip=True), errors="coerce"),
                "tenure_text": " ".join(tds[5].get_text(" ", strip=True).split()),
                "contract_until": tds[6].get_text(strip=True) or None,
                "former_player": former_player_link is not None,
                "former_player_profile_url": (
                    urljoin(BASE_URL, former_player_link["href"]) if former_player_link else None
                ),
            }
        )
    return rows


def parse_squad_html(html: str, *, country: str, tm_country: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if table is None:
        return []
    rows: list[dict[str, Any]] = []
    for tr in table.find_all("tr", class_=["odd", "even"]):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 5:
            continue
        player_anchor = tds[1].find("a", href=True)
        club_anchor = tds[3].find("a", href=True)
        value_anchor = tds[4].find("a", href=True)
        if player_anchor is None:
            continue
        player_name = " ".join(player_anchor.get_text(" ", strip=True).split())
        position = None
        inline_rows = tds[1].find("table", class_="inline-table")
        if inline_rows is not None:
            nested_rows = inline_rows.find_all("tr", recursive=False)
            if len(nested_rows) > 1:
                position = " ".join(nested_rows[1].get_text(" ", strip=True).split())
        if not position:
            text_blob = " ".join(tds[1].get_text(" ", strip=True).split())
            position = text_blob.replace(player_name, "", 1).strip() or None
        market_value_text = value_anchor.get_text(strip=True) if value_anchor else tds[4].get_text(strip=True)
        club_name = None
        club_url = None
        club_id = None
        if club_anchor is not None:
            club_url = urljoin(BASE_URL, club_anchor["href"])
            club_name = club_anchor.get("title")
            club_id = extract_id_from_href(club_anchor["href"], "verein")
        rows.append(
            {
                "country": country,
                "tm_country": tm_country,
                "shirt_number": pd.to_numeric(tds[0].get_text(strip=True), errors="coerce"),
                "player": player_name,
                "player_id": extract_id_from_href(player_anchor["href"], "spieler"),
                "player_profile_url": urljoin(BASE_URL, player_anchor["href"]),
                "position": position or None,
                "age": pd.to_numeric(tds[2].get_text(strip=True), errors="coerce"),
                "club": club_name,
                "club_id": club_id,
                "club_url": club_url,
                "market_value_text": market_value_text,
                "market_value_eur": parse_money_to_eur(market_value_text),
            }
        )
    return rows


def parse_achievements_html(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if table is None:
        return []
    achievements: list[dict[str, Any]] = []
    for tr in table.find_all("tr"):
        extra = tr.find("td", class_="extrarow")
        if extra is None:
            continue
        label = " ".join(extra.get_text(" ", strip=True).split())
        match = re.match(r"(\d+)x\s+(.*)", label)
        if match:
            achievements.append({"count": int(match.group(1)), "title": match.group(2)})
    return achievements


def build_player_current_season_stats(players: pd.DataFrame, performance_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if players.empty:
        return pd.DataFrame(rows)

    for player in players.to_dict("records"):
        player_id = player.get("player_id")
        if pd.isna(player_id) or player_id is None:
            continue
        path = performance_dir / f"{int(player_id)}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("data", {})
        performances = data.get("performance", [])
        club_rows = [
            item
            for item in performances
            if not item.get("gameInformation", {}).get("isNationalGame")
        ]
        if not club_rows:
            continue
        current_season_id = max(
            int(item["gameInformation"]["seasonId"])
            for item in club_rows
            if item.get("gameInformation", {}).get("seasonId") is not None
        )
        current_rows = [
            item
            for item in club_rows
            if int(item.get("gameInformation", {}).get("seasonId", -1)) == current_season_id
        ]
        goals = 0
        assists = 0
        appearances = 0
        minutes = 0
        competition_ids = set()
        for item in current_rows:
            stats = item.get("statistics", {})
            goal_stats = stats.get("goalStatistics", {})
            general = stats.get("generalStatistics", {})
            participation = general.get("participationState")
            if participation in {"injured", "suspended", "notInSquad"}:
                continue
            appearances += 1
            goals += int(goal_stats.get("goalsScoredTotalOfficial") or 0)
            assists += int(goal_stats.get("assistsOfficial") or 0)
            minutes += int(stats.get("minuteStatistics", {}).get("minutesPlayedTotal") or 0)
            competition_id = item.get("gameInformation", {}).get("competitionId")
            if competition_id:
                competition_ids.add(str(competition_id))
        rows.append(
            {
                "player_id": int(player_id),
                "player": player["player"],
                "country": player["country"],
                "position": player["position"],
                "age": round(float(player["age"]), 2) if pd.notna(player.get("age")) else None,
                "club": player["club"],
                "market_value_eur": player.get("market_value_eur"),
                "market_value_text": player.get("market_value_text"),
                "current_season_id": current_season_id,
                "appearances": appearances,
                "minutes": minutes,
                "goals": goals,
                "assists": assists,
                "goal_contributions": goals + assists,
                "competition_count": len(competition_ids),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["goal_contributions", "goals", "assists", "player"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_transfermarkt_enrichment(raw_dir: Path) -> dict[str, pd.DataFrame | list[dict[str, Any]] | None]:
    paths = TransfermarktPaths(raw_dir / "transfermarkt")
    if not paths.participants_html.exists() or not paths.managers_html.exists():
        return {
            "players": pd.DataFrame(),
            "player_season_stats": pd.DataFrame(),
            "global_clubs": pd.DataFrame(),
            "country_clubs": pd.DataFrame(),
            "squad_values": pd.DataFrame(),
            "coaches": pd.DataFrame(),
            "coach_titles": pd.DataFrame(),
            "source_rows": [],
        }

    participants = parse_participants_html(paths.participants_html.read_text(encoding="utf-8"))

    player_rows: list[dict[str, Any]] = []
    for participant in participants:
        filename = paths.squads_dir / f"{slugify(participant['country'])}.html"
        if not filename.exists():
            continue
        html = filename.read_text(encoding="utf-8")
        player_rows.extend(
            parse_squad_html(html, country=participant["country"], tm_country=participant["tm_country"])
        )
    players = pd.DataFrame(player_rows)
    if players.empty:
        return {
            "players": players,
            "player_season_stats": pd.DataFrame(),
            "global_clubs": pd.DataFrame(),
            "country_clubs": pd.DataFrame(),
            "squad_values": pd.DataFrame(),
            "coaches": pd.DataFrame(),
            "coach_titles": pd.DataFrame(),
            "source_rows": [],
        }

    players["market_value_eur"] = pd.to_numeric(players["market_value_eur"], errors="coerce")
    players["age"] = pd.to_numeric(players["age"], errors="coerce")
    players["shirt_number"] = pd.to_numeric(players["shirt_number"], errors="coerce")
    player_season_stats = build_player_current_season_stats(players, paths.player_performance_dir)

    global_clubs = (
        players.dropna(subset=["club"])
        .groupby(["club", "club_id", "club_url"], as_index=False)
        .agg(
            player_count=("player", "size"),
            represented_countries=("country", "nunique"),
            total_market_value_eur=("market_value_eur", "sum"),
        )
        .sort_values(["player_count", "total_market_value_eur", "club"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    global_clubs["player_count_rank"] = global_clubs["player_count"].rank(method="min", ascending=False).astype(int)

    club_counts = (
        players.dropna(subset=["club"])
        .groupby(["country", "club", "club_id", "club_url"], as_index=False)
        .agg(player_count=("player", "size"))
    )
    club_counts["country_rank"] = (
        club_counts.groupby("country")["player_count"].rank(method="min", ascending=False).astype(int)
    )
    country_clubs = club_counts.sort_values(["country", "country_rank", "club"]).reset_index(drop=True)

    squad_values = (
        players.groupby("country", as_index=False)
        .agg(
            player_rows=("player", "size"),
            squad_market_value_eur=("market_value_eur", "sum"),
            average_player_market_value_eur=("market_value_eur", "mean"),
        )
        .sort_values(["squad_market_value_eur", "average_player_market_value_eur"], ascending=[False, False])
        .reset_index(drop=True)
    )
    squad_values["market_value_rank"] = (
        squad_values["squad_market_value_eur"].rank(method="min", ascending=False).astype(int)
    )

    manager_rows = parse_manager_overview_html(paths.managers_html.read_text(encoding="utf-8"))
    achievements_rows: list[dict[str, Any]] = []
    coach_rows: list[dict[str, Any]] = []
    for manager in manager_rows:
        manager_id = manager.get("manager_id")
        achievements_file = paths.coach_achievements_dir / f"{manager_id}.html"
        achievements = []
        if achievements_file.exists():
            achievements = parse_achievements_html(achievements_file.read_text(encoding="utf-8"))
        total_titles = sum(item["count"] for item in achievements)
        title_types = len(achievements)
        honors_preview = ", ".join(
            f"{item['count']}x {item['title']}" for item in achievements[:4]
        )
        coach_rows.append(
            {
                **manager,
                "age": round(float(manager["age"]), 2) if pd.notna(manager["age"]) else None,
                "total_titles_won": total_titles,
                "title_types": title_types,
                "honors_preview": honors_preview or None,
                "foreign_to_team": (
                    manager["nationality"] != manager["tm_country"]
                    if manager.get("nationality") and manager.get("tm_country")
                    else None
                ),
            }
        )
        for item in achievements:
            achievements_rows.append(
                {
                    "manager": manager["manager"],
                    "country": manager["country"],
                    "nationality": manager["nationality"],
                    "title": item["title"],
                    "count": item["count"],
                }
            )

    coaches = pd.DataFrame(coach_rows).sort_values(
        ["total_titles_won", "age", "manager"], ascending=[False, False, True]
    )
    if not coaches.empty:
        coaches["age_rank_oldest"] = coaches["age"].rank(method="min", ascending=False).astype(int)
        coaches["title_rank"] = coaches["total_titles_won"].rank(method="min", ascending=False).astype(int)
    coach_titles = pd.DataFrame(achievements_rows).sort_values(["count", "manager"], ascending=[False, True])

    source_rows = [
        {
            "key": "transfermarkt_squads_2026",
            "label": "Transfermarkt World Cup 2026 detailed squad pages",
            "scope": "current clubs, player ages, player market values, squad value rankings, club representation",
            "url": "https://www.transfermarkt.com/weltmeisterschaft/teilnehmer/pokalwettbewerb/FIWC",
        },
        {
            "key": "transfermarkt_managers_2026",
            "label": "Transfermarkt World Cup 2026 manager overview and achievements pages",
            "scope": "coach ages, nationalities, tenure, contract years, and trophy counts",
            "url": "https://www.transfermarkt.com/weltmeisterschaft/trainer/pokalwettbewerb/FIWC",
        },
        {
            "key": "transfermarkt_player_performance_2026",
            "label": "Transfermarkt player performance endpoint",
            "scope": "current club-season appearances, goals, assists, and goal contributions for squad players",
            "url": "https://www.transfermarkt.com/ceapi/performance-game/{player_id}",
        },
    ]

    return {
        "players": players,
        "player_season_stats": player_season_stats,
        "global_clubs": global_clubs,
        "country_clubs": country_clubs,
        "squad_values": squad_values,
        "coaches": coaches,
        "coach_titles": coach_titles,
        "source_rows": source_rows,
    }
