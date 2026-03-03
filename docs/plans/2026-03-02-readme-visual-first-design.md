# README Visual-First Redesign

## Goal

Simplify the README and lead with visuals so users immediately see the value
of the 3D embedding visualization and semantic search. Keep the main content
to roughly 1–2 screens, with advanced CLI and development details hidden in
collapsible sections.

## Target Audience

Data/ML practitioners, product/data teams, and engineers. The layout prioritizes
visuals and quick start for non-technical users while still providing a path to
advanced CLI and dev workflows.

## Information Architecture (Approved)

1. **Hero section**
   - Title + one-line tagline
   - Wide hero GIF showing 3D plot rotation + hover tooltip
   - A grid of 3 screenshots below: cluster plot, semantic search panel,
     collections/dashboard page

2. **Quick Start**
   - 3 steps for server mode: clone, `uv sync --all-extras`, run server
   - Optional step to index sample data if no collections exist

3. **Features (concise)**
   - 6–8 bullets, grouped into: Embeddings & Storage, Clustering & Plot,
     Search & Collections, Web UI

4. **Advanced details (collapsible)**
   - `<details>` Advanced CLI (index + plot examples, key env vars)
   - `<details>` Development (lint, tests, type check)

## Visual Assets

Required:

- **Hero GIF**: 3D plot rotation with hover tooltip
- **Screenshot**: 3D cluster plot (static)
- **Screenshot**: semantic search results (static)
- **Screenshot**: collections dashboard

Nice-to-have:

- **Screenshot**: image sprites render mode
- **Screenshot**: cluster suggestion chart
- **GIF**: semantic search query flow

## Success Criteria

- README appears simple and visual-first
- Quick start is obvious and short
- Advanced CLI and dev details are accessible but hidden by default
- All visuals are referenced with correct paths under `docs/screenshots/`
