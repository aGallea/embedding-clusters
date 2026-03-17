# Sub-cluster Preview Card Design

## Summary

Refine the expanded sub-cluster preview rows in `ClusterDetailDrawer` so they visually match the main item list more closely. Instead of showing only a simple title and id, each preview row should present richer product details in the same general style as the primary cluster item list: thumbnail when available, id, optional distance, and a few metadata fields. These preview rows remain display-only.

## Problem

The current expanded sub-cluster rows are too minimal. They help confirm that expansion works, but they do not give the user enough product context to understand what is inside a sub-cluster. The user wants the preview list to feel like the main product list, especially for image-based collections.

## Goals

- Make preview rows visually consistent with the main drawer item list.
- Show richer product context inside expanded sub-cluster previews.
- Preserve the current accordion flow and dedicated **Drill deeper** action.
- Keep preview rows display-only.

## Non-goals

- Adding selection/highlighting behavior to preview rows.
- Replacing the main item list.
- Adding nested pagination or a new backend endpoint in this pass.

## Recommended UX

Inside the expanded sub-cluster panel, each preview row should use the same visual structure as the main item list, but in compact read-only form.

Each preview card should show:

- product thumbnail when `imageField` is available
- item id as the primary label
- `dist` line when a distance value is available from the preview data source
- a few metadata rows beneath, following the same text styling pattern as the main list

These rows should not be clickable and should not alter selection state.

## Data Strategy

Reuse the preview source already present in the drawer:

- `currentLevel.subClusterData.points`

Each `SubClusterPoint` already provides:

- `id`
- `metadata`
- `sub_cluster`

If a preview row does not have a distance field available from this source, omit the `dist` line rather than inventing or approximating one.

## Visual Rules

- Use the main drawer list as the styling reference.
- Keep preview rows slightly compact to avoid making the accordion excessively tall.
- Keep the preview cap at 10 items.
- Keep the **Drill deeper** button below the preview list.

## Testing

Update E2E coverage so expanded preview rows verify richer content, such as:

- image is present when available
- id is visible
- metadata lines are visible
- expansion remains display-only and does not affect selection

## Files Expected to Change

- `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- `frontend/e2e/cluster-drawer.spec.ts`

## Recommended Implementation Direction

Refactor the preview-row rendering inside the expanded sub-cluster accordion to reuse the same structural pattern as the main drawer item rows while keeping the preview rows non-interactive.
