# World Cup Dashboard Source Plan

## Product Goal

Launch a strong World Cup dashboard for fans and journalists just before the tournament starts, then keep it updated during the competition without rebuilding the data layer from scratch.

The dashboard should answer three different jobs:

1. Pre-tournament context:
   - Which squads are tallest, oldest, youngest, or most unusual?
   - How does 2026 compare with the last 35 years?
   - Which groups look physically imposing or unusually experienced?
2. Tournament-time updates:
   - What happened today?
   - Which groups changed shape after new results?
   - Which teams are outperforming or underperforming expectations?
3. Explainers and embeds:
   - Map cards
   - Group comparison cards
   - Country profile pages
   - One-off charts for stories and social posts

## Recommended Source Stack

Use three layers, not one.

### 1. Core player trend layer

- Source: local World Cup players dataset
- Role: backbone for age, height, position, country, and tournament-year comparisons
- Why it matters: this is the source that already powers the strongest stories

### 2. Competition and live update layer

- Source: `football-data.org`
- Role: World Cup competition metadata, teams, matches, standings, and match-level updates
- Why it matters: this is the cleanest public developer API for keeping the dashboard fresh before and during the tournament

### 3. Country enrichment layer

- Sources: `REST Countries`, optionally `World Bank`, and map geometry from `Natural Earth`
- Role: flags, regions, ISO codes, population, and country-level context for maps and richer comparisons
- Why it matters: sports APIs are weak at geography and country context

### Truth layer

- Source: official FIFA pages
- Role: verify 2026 groups, fixtures, venues, naming, and tournament framing
- Why it matters: official pages are the canonical source when there is any doubt

## Source Matrix

| Source | Key fields we want | Use cases | Reliability | License / risk |
| --- | --- | --- | --- | --- |
| Local World Cup players CSV | `tournament_year`, `country`, `position`, `height_cm`, `age_at_tournament_years`, player names | Age and height trends, position mix, country comparisons, group summaries, historical deltas | High for our current stories, but early height coverage is thin | Internal file; must document gaps and provenance |
| FIFA official tournament pages | groups, fixtures, stadiums, dates, official names | Final validation, editorial framing, QA for 2026 modules | High | Public web pages, but not a normal open API |
| football-data.org | competitions, teams, matches, standings, squad/team resources, trend resource | Live dashboard updates, fixtures pages, group tables, match pages, knockout tracker | High | API key, rate limits, commercial access for deeper usage |
| REST Countries | names, flags, regions, capital, population, ISO codes | Country cards, flags, map labels, normalization | High | Low risk |
| World Bank API | population, GDP, income group, development indicators | Optional enrichment for contextual stories | High | Low risk |
| Natural Earth / GeoJSON country polygons | map geometry | Choropleths and geographic overlays | High | Low risk |
| Wikidata | aliases, multilingual labels, stadium metadata, coordinates | Name cleanup, multilingual pages, rich metadata joins | Moderate | Variable data quality, query complexity |
| OpenFootball | historical competition structure and results | Open backup for historical fixture/result comparisons | Moderate | Good open-data option, thinner for player-level work |

## What We Should Build First

For launch, the winning move is not “collect everything.” It is to build one reliable spine.

### Phase 1: pre-tournament spine

Build and publish:

- tournament trend tables from the existing CSV
- 2026 country summary table
- 2026 group summary table
- 2026 country-vs-history table
- map-ready country dataset with ISO codes
- football-data.org raw caches for:
  - competition
  - teams
  - matches
  - standings

That is enough to power:

- tallest teams
- oldest teams
- group comparison
- country profile cards
- fixtures module
- group table module

### Phase 2: live tournament layer

Add:

- match status refreshes
- standings refreshes
- daily results summary
- “today / tomorrow” fixture modules
- lightweight alerting for changed group standings

### Phase 3: richer editorial tools

Add:

- story-specific scenario tables
- upset tracker
- confederation tracker
- country profile pages
- reusable embeddable chart cards

## Data Model

The dashboard should separate `raw`, `reference`, and `curated` data.

### Raw tables

- `raw/football_data/competition_wc.json`
- `raw/football_data/teams_wc.json`
- `raw/football_data/matches_wc.json`
- `raw/football_data/standings_wc.json`
- `raw/reference/rest_countries.json`

These should be treated as caches of source responses.

### Curated tables

- `curated/tournament_trends.csv`
  - one row per tournament year
  - average age, average height, player counts, country counts, coverage metrics
- `curated/country_snapshot_2026.csv`
  - one row per 2026 country
  - average height, average age, player rows, confederation, group, ranks, historical deltas
- `curated/group_snapshot_2026.csv`
  - one row per 2026 group
  - average height, average age, team count, player rows
- `curated/country_history_2026.csv`
  - one row per 2026 country
  - baseline tournament year, then-vs-now height and age changes
- `curated/position_trends.csv`
  - one row per tournament year and position
- `curated/position_share_trends.csv`
  - one row per tournament year with position share columns

### Reference tables

- `reference/country_metadata.csv`
  - ISO codes, region, subregion, population, capital, flag URL

## Update Cadence

### Before kickoff

Refresh cadence:

- player-derived curated tables: when the source CSV changes
- FIFA validation: manual check before major publishes
- football-data.org pulls: daily
- country metadata: on demand only

### During the tournament

Refresh cadence:

- matches: every 5 to 15 minutes on match days
- standings: every 5 to 15 minutes on match days
- curated live tables: rebuild after match/standings refresh
- heavier editorial summary pages: once or twice daily

## Launch Dashboard Modules

This stack supports a strong launch set immediately:

1. Tallest and oldest teams in 2026
2. Group-by-group comparison cards
3. Interactive world map with height/age toggle
4. Country profile cards with 2026 vs roughly 30 to 40 years ago
5. Tournament trend charts since 1990 or 1986
6. Fixtures and standings panels
7. Matchday update card

## Risks

### Real risk

- Early historical height coverage is incomplete, especially before late modern tournaments.
- The 2026 row in the current CSV appears before kickoff, so it must stay clearly labeled as a dataset cohort.
- football-data.org requires API access and may have plan or rate constraints depending on the endpoints used.

### Not a real risk

- Country enrichment and map geometry are cheap and stable. Do not overthink those.

## Recommendation

The right architecture for this project is:

1. Keep the current player CSV as the historical storytelling backbone.
2. Add football-data.org as the updateable competition backbone.
3. Add a small reference layer for geography and country metadata.
4. Keep FIFA as the final editorial truth check.

That is enough to launch something very good before the tournament starts and keep it alive once play begins.
