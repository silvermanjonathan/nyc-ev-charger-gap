# NYC EV Charger Gap Finder

## Goal
Rank New York City's 59 community districts by how underserved they are for public EV charging, by comparing where public chargers are with what community boards ask for in their budget requests.

## Status
Week 1: repo set up and source data identified.

## Planned
- Load both datasets into SQLite
- Count chargers per district (point-in-polygon with shapely)
- Label budget requests with a model in Azure AI Foundry, checked against 40 hand-labeled requests
- An agent that answers questions about the rankings through SQL tool calls, with tested guardrails

## Data
- Register of Community Board Budget Requests, NYC Open Data (vn4m-mk4t)
- Electric vehicle charging stations, data.ny.gov (7rrd-248n)
- Community District boundaries, NYC Open Data (5crt-au7u)

The data is not committed. Download the CSVs into the `data/` folder.