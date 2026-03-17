# Selection-Gated Recursive Sub-clustering Design

## Summary

Refine `ClusterDetailDrawer` so that once sub-clusters have been created, the lower main product list is removed from the drawer and recursive sub-clustering is driven only by selecting a specific sub-cluster row and pressing **Compute Sub-clusters**. This replaces the current mixed mode where the drawer still shows the original lower item list and offers multiple recursion paths.

## Problem

After the first sub-cluster compute, the drawer currently shows both:

- the new sub-cluster section
- the original lower product list with pagination

That creates a confusing split-purpose UI. At the same time, recursive drill behavior is still available through separate controls rather than making the user explicitly choose which sub-cluster the next compute should operate on.

The requested behavior is:

- once sub-clusters exist, remove the lower main product list from the drawer
- require the user to select a specific sub-cluster before **Compute Sub-clusters** is enabled for recursion

## Goals

- Focus the drawer on one mode at a time
- Hide the lower main product list after sub-clustering begins
- Make recursive compute act only on an explicitly selected sub-cluster
- Keep the sub-cluster preview accordion behavior
- Prevent ambiguous recursion targets

## Non-goals

- Reworking breadcrumb behavior
- Changing the main plot drill visualization behavior
- Adding backend state for hierarchy tracking

## Recommended UX

### Before first sub-cluster compute

Keep the current behavior:

- lower product list is visible
- **Compute Sub-clusters** acts on the currently selected cluster

### After sub-clusters exist

The drawer switches into sub-cluster-focused mode:

- hide the lower main product list
- hide its pagination controls
- keep the sub-cluster controls section
- keep the current sub-cluster list and preview expansion

### Sub-cluster selection

Use row selection as the recursion source model:

- clicking a sub-cluster row selects it
- selected row gets active styling
- only one sub-cluster can be selected at a time
- clicking the selected row again can either keep it selected or toggle off; first implementation should prefer toggle-off for clarity

### Chevron behavior

Keep the current chevron behavior unchanged:

- chevron expands/collapses preview content only
- chevron does not recurse

### Compute button behavior

After sub-clusters exist:

- **Compute Sub-clusters** is disabled until a sub-cluster row is selected
- clicking compute uses the selected sub-cluster's point ids as the next recursive source
- after successful compute, clear selected-row state for the new level

### Recursive controls

Once compute becomes selection-gated, the separate **Drill deeper** button becomes redundant and should be removed to avoid duplicate recursion paths.

## State Model

Add local drawer state for row selection, for example:

- `selectedSubClusterIndex: number | null`

Rules:

- reset selection when cluster changes
- reset selection when a deeper level is computed successfully
- reset selection when current-level sub-cluster data changes

## Data Flow

After sub-clusters exist, recursion should use the currently selected row's `sub_cluster` index to derive `point_ids` from `currentLevel.subClusterData.points`, then call the existing generic recursive endpoint.

This keeps the backend contract unchanged.

## Testing

Update frontend E2E coverage to verify:

1. Lower main product list is hidden once sub-clusters are created.
2. **Compute Sub-clusters** is disabled until a sub-cluster row is selected.
3. Clicking a row selects it and enables compute.
4. Clicking compute after row selection deepens the breadcrumb.
5. Chevron expansion still works independently of selection.

## Files Expected to Change

- `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- `frontend/e2e/cluster-drawer.spec.ts`

## Recommended Implementation Direction

Implement this as a focused drawer-state refinement:

- introduce explicit selected sub-cluster state
- hide the lower product list in drilled mode
- gate compute on row selection in drilled mode
- remove the redundant explicit drill button

This produces a cleaner, less ambiguous recursive workflow while preserving the existing backend and plot drill architecture.
