# Dataviz Lab

This is a small, reproducible starting point for report-style data visualization work that can be exported to static HTML without needing a live notebook runtime.

## Current shape

- `scripts/worldcup_height_ft.py`: first FT-style static chart.
- `scripts/worldcup_report_ft.py`: static multi-chart report.
- `scripts/worldcup_report_interactive.py`: interactive Plotly report for shipping as a standalone HTML page.
- `scripts/worldcup_data_ingest.py`: local ingestion scaffold for dashboard-ready tables and optional external source fetches.
- `scripts/worldcup_dashboard.py`: separate World Cup dashboard surface for fans and journalists, built from the normalized bundle.
- `docs/worldcup-dashboard-source-plan.md`: concrete data-source and dashboard plan for a pre-tournament launch that can keep updating during the World Cup.

## First artifact

- `scripts/worldcup_height_ft.py`: reads the World Cup players CSV and renders an FT-style Matplotlib chart.
- `output/worldcup-height/`: generated PNG, SVG, HTML, and summary CSV artifacts.

## Run

```bash
./.venv/bin/python dataviz-lab/scripts/worldcup_height_ft.py
```

Build the dashboard-ready local data layer:

```bash
./.venv/bin/python dataviz-lab/scripts/worldcup_data_ingest.py --build-local
```

Try the full scaffold:

```bash
FOOTBALL_DATA_API_TOKEN=... ./.venv/bin/python dataviz-lab/scripts/worldcup_data_ingest.py --all
```

If `dataviz-lab/.env` exists, the ingest script will load it automatically before checking environment variables.

Build the separate dashboard:

```bash
./.venv/bin/python dataviz-lab/scripts/worldcup_dashboard.py
```

Package the canonical dashboard for static hosting:

```bash
mkdir -p dataviz-lab/deploy/cloudflare-pages
cp dataviz-lab/output/worldcup-dashboard/worldcup-dashboard-v2.html dataviz-lab/deploy/cloudflare-pages/index.html
cp dataviz-lab/output/worldcup-dashboard/worldcup-dashboard-v2.html dataviz-lab/deploy/cloudflare-pages/404.html
```

Deploy to Cloudflare Pages with the local Wrangler install already present on this machine:

```bash
node "/Users/emot/Documents/New project/daily-news-worker-fix/node_modules/wrangler/bin/wrangler.js" pages deploy "/Users/emot/Documents/New project/dataviz-lab/deploy/cloudflare-pages" --project-name worldcup-2026-dashboard --branch main --commit-dirty true --commit-message "Publish World Cup dashboard"
```

Token template:

- `dataviz-lab/.env.example`

## Why this structure

- Keep analysis code as plain scripts so it can run in CI or a build step.
- Emit static assets that can be dropped into an event microsite or report page.
- Stay close to notebook ergonomics while avoiding notebook lock-in for publishing.
- Separate `raw`, `reference`, and `curated` data so pre-tournament analysis and live-tournament refreshes can share the same pipeline.
- Add a normalized bundle so a dashboard, a longform report, and future embeds can all read the same core data model.
