# Sub-cluster Row Expansion in ClusterDetailDrawer

## Summary

Adjust the recursive sub-clustering UX in the right sidebar so the sub-cluster row chevron no longer triggers recursive drilling. The chevron should only expand or collapse a specific sub-cluster row to reveal its products. Recursive drilling remains available, but only through a dedicated **Drill deeper** button inside the expanded row. The existing **Compute Sub-clusters** button remains the only action that generates sub-clusters for the current level.

## Problem

The current drawer behavior overloads the down-arrow chevron in the **Current sub-clusters** list. Clicking it immediately computes the next recursive level, which is surprising because the icon suggests disclosure or expansion. This creates a mismatch between the control's visual language and its behavior.

The desired interaction is:

- **Compute Sub-clusters** generates the current level's sub-clusters.
- The row chevron expands a sub-cluster to reveal its products.
- A clearly labeled **Drill deeper** button performs recursive sub-clustering for that specific sub-cluster.

## Goals

- Make the chevron a pure expand/collapse control.
- Let users inspect the products inside a sub-cluster before deciding to recurse.
- Preserve recursive sub-clustering with a dedicated, explicit action.
- Keep the existing top-level **Compute Sub-clusters** button as the only control that generates the current level.
- Keep the bottom legend display-only.

## Non-goals

- Reworking breadcrumb behavior.
- Changing main-plot drill visualization behavior.
- Adding a second compute control outside the existing sub-clustering section.
- Building full independent pagination inside each expanded row in the first pass.

## Recommended UX

### Sub-clustering controls

Keep the current sub-clustering control section unchanged:

- k slider
- Suggest k button
- Compute Sub-clusters button

This section remains responsible for generating sub-clusters for the currently selected cluster or drilled level.

### Current sub-clusters list

When a drilled level exists, render the current sub-cluster list as an accordion.

Each row header shows:

- sub-cluster color dot
- sub-cluster label (`Sub 0`, `Sub 1`, ...)
- point count
- chevron button on the right

The chevron:

- toggles row expansion only
- never computes sub-clusters
- never navigates the breadcrumb

### Expanded row content

Expanding a row reveals:

- a compact preview list of products belonging to that sub-cluster
- a dedicated **Drill deeper** button

The preview list should show a limited number of products in the first implementation, rather than a fully paginated nested list. This keeps the drawer readable and avoids introducing a second pagination model inside the sidebar.

Recommended preview behavior:

- show the first 10 products from the selected sub-cluster
- reuse the existing compact item display style where possible
- include product id and a small amount of metadata
- include thumbnail if the current drawer item rendering already has image data available

### Recursive drill action

The expanded row's **Drill deeper** button triggers recursive sub-clustering for that specific sub-cluster.

Behavior:

- enabled only when the sub-cluster contains enough points to recurse
- shows loading state while the recursive request is running
- preserves the existing breadcrumb and plot drill behavior

## Data Strategy

## Product preview source

The first implementation should prefer reusing already available client data instead of adding a new backend endpoint.

Candidate sources:

1. `currentLevel.subClusterData.points` provides point ids and sub-cluster assignments.
2. `clusterDetail.items` provides the currently fetched cluster item records.

For the first pass, the drawer can derive the preview list by:

- filtering `currentLevel.subClusterData.points` to ids belonging to the expanded sub-cluster
- matching those ids against currently available item records
- rendering a limited preview from that matched set

If this does not provide sufficient preview coverage, a follow-up can introduce a dedicated fetch for sub-cluster item details. That is intentionally deferred.

## State Model

Add local drawer state for expansion, for example:

- `expandedSubClusterIndex: number | null`

Rules:

- only one expanded sub-cluster row at a time
- changing selected cluster resets the expanded row
- computing a new current-level sub-cluster result resets the expanded row
- drilling deeper resets the expanded row for the new level

## Loading and Error Handling

- Expanding or collapsing rows is local and immediate.
- **Compute Sub-clusters** continues to use the existing drill loading state for generating the current level.
- **Drill deeper** uses the same recursive drill request path, but only from the expanded row button.
- If a sub-cluster has too few points, the row can still expand, but **Drill deeper** should be disabled or hidden.
- If recursive drill fails, keep the row expanded and preserve the current level so the user does not lose context.

## Testing

Update frontend E2E coverage to verify:

1. Clicking the sub-cluster row chevron expands the row instead of drilling.
2. Expanded content shows a product preview for that sub-cluster.
3. Clicking **Drill deeper** from the expanded row performs recursive drill and shows the breadcrumb/deeper level.
4. The existing **Compute Sub-clusters** button still generates the first-level sub-clusters.

## Files Expected to Change

- `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- `frontend/e2e/cluster-drawer.spec.ts`

Potentially, if a small presentational extraction improves readability:

- `frontend/src/components/plot/` sub-component for expanded sub-cluster rows

## Recommended Implementation Direction

Implement this as a focused UI refinement inside `ClusterDetailDrawer.tsx`:

- replace chevron-triggered recursion with local accordion state
- render preview content inside expanded rows
- move recursive drill to an explicit button inside expanded content

This change keeps the underlying recursive drill architecture intact while making the drawer behavior align with user expectations.
