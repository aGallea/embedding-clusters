# Interactive Sub-cluster Item Lists Design

## Summary

Refine drilled-mode behavior in `ClusterDetailDrawer` so expanded sub-cluster panels stop behaving as read-only previews and instead become the interactive product lists for that drilled level. Remove the 10-item cap, make expanded panels scrollable, and let product-card selection reuse the existing global highlight behavior on the 3D plot.

## Problem

The current drilled-mode sub-cluster panels only show a limited preview and do not allow product selection. That creates two problems:

- users cannot inspect the full item set inside a sub-cluster
- users cannot click drilled-mode items to highlight them on the 3D plot

The user explicitly wants:

- full product lists inside expanded sub-cluster panels
- scrolling instead of a 10-item limit
- interactive product selection in drilled mode
- immediate selection highlighting on the 3D plot

## Goals

- Remove the 10-item preview cap in drilled mode
- Make expanded sub-cluster panels scrollable
- Make drilled-mode product cards clickable and highlightable on the 3D plot
- Reuse the existing global item-selection mechanism
- Preserve the separate row-selection model for recursive compute

## Non-goals

- Reintroducing the old lower main product list in drilled mode
- Replacing the current breadcrumb model
- Adding a new backend endpoint in this pass

## Recommended UX

### Expanded sub-cluster panels

Expanded sub-cluster panels become the primary item lists for drilled mode.

Behavior:

- remove the “Showing up to 10 products” preview framing
- render the full set of products for that sub-cluster
- keep the panel scrollable within the drawer

### Product card interaction

Product cards inside expanded sub-cluster panels should behave like the old lower list items:

- click to toggle selection
- selected styling mirrors the old main list styling as closely as possible
- selection updates the 3D plot highlight immediately

### Global selection model

Product selection stays global across the drawer:

- selected products from one expanded sub-cluster remain selected even if another sub-cluster is expanded
- drilled-mode item cards reuse the same `selectedPointIds` and highlight flow used by the original lower list

### Keep interaction roles separate

- row click = select recursion source sub-cluster
- chevron click = expand/collapse product list
- product card click = select/deselect product for highlight

These interactions must remain distinct and not leak into one another.

## Data Strategy

Use the existing drilled-mode sub-cluster point data already available in the drawer.

The current sub-cluster point records already include:

- `id`
- `metadata`
- sub-cluster assignment

That is sufficient for rendering item cards without introducing a new fetch in the first pass.

## Scrolling Model

The expanded panel should become the scrollable container for its item list.

Requirements:

- full product list is available inside the expanded panel
- the panel is height-constrained within the drawer
- scrolling remains usable even when the drawer contains multiple expanded sections

## Testing

Update frontend E2E coverage to verify:

1. Expanded sub-cluster panel no longer behaves like a fixed 10-item preview.
2. Product cards inside expanded panels are clickable.
3. Clicking a drilled-mode product card toggles selection.
4. Selection affects the 3D plot highlight state.
5. Selection persists across expanded panels because it is global.
6. Row selection for recursive compute still works independently of product-card selection.

## Files Expected to Change

- `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- `frontend/e2e/cluster-drawer.spec.ts`

## Recommended Implementation Direction

Replace the current preview-only expanded panel rendering with a fully interactive item-list rendering path that reuses the existing product-selection behavior and styling patterns from the former lower main list.
