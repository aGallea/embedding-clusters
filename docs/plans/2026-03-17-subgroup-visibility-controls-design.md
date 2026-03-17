# Sub-group Visibility Controls Design

## Summary

Add hide/show/isolate/show-all controls for drilled sub-groups so the bottom legend in drilled mode mirrors the same visibility interactions already available for top-level groups. These controls should affect only the current drilled level and should not modify parent/top-level cluster visibility state.

## Problem

Top-level groups already support:

- hide/show via the eye icon
- isolate via Ctrl/Cmd+click
- Show All

But drilled sub-groups currently do not. In drilled mode, the legend becomes display-only for sub-groups, which makes it impossible to temporarily hide a sub-group, isolate one, or restore them all while exploring the drilled view.

## Goals

- Give drilled sub-groups the same legend visibility controls as top-level groups
- Scope drilled visibility only to the current drilled level
- Preserve top-level visibility state independently
- Keep interaction location consistent in the bottom legend

## Non-goals

- Changing parent/top-level visibility when drilling
- Moving visibility controls into the right drawer
- Adding backend state for visibility

## Recommended UX

### Bottom legend in drilled mode

For each drilled sub-group row, add:

- eye icon for hide/show
- Ctrl/Cmd+click on eye icon for isolate

Add a drilled-mode **Show All** action in the legend header.

### Visibility scope

Drilled sub-group visibility affects only the current drilled view.

Examples:

- hiding `Sub 1` hides only that sub-group in the current drilled level
- isolating `Sub 2` hides the other sub-groups only within that drilled level
- navigating back restores the parent-level visibility behavior already in place

### State reset rules

Reset drilled sub-group visibility state when:

- drilling deeper into a new level
- navigating back to a parent level
- resetting the drill entirely

## State Model

Add a dedicated visibility state for drilled sub-groups, separate from top-level `visibleClusters`.

Suggested shape:

- `visibleSubClusters: Set<number>` or equivalent drilled-level visibility state

This state should represent the currently visible sub-group indices for the active drilled level only.

## Rendering Behavior

In drilled mode, plot renderers should respect drilled visibility state when deciding whether to show a point. Hidden sub-groups should disappear from the current drilled view.

This should integrate with the existing drilled color/highlight logic rather than creating a second rendering path.

## Testing

Update frontend E2E coverage to verify:

1. Hide/show works for drilled sub-groups.
2. Ctrl/Cmd+click isolates a drilled sub-group.
3. Show All restores all drilled sub-groups.
4. Navigating back does not leak drilled visibility state into the parent/top-level view.

## Files Expected to Change

- `frontend/src/stores/plotStore.ts`
- `frontend/src/components/plot/ClusterLegend.tsx`
- relevant plot renderers if they need to respect drilled sub-group visibility state
- `frontend/e2e/cluster-drawer.spec.ts` and/or legend-related Playwright coverage

## Recommended Implementation Direction

Mirror the existing top-level visibility model for drilled mode, but keep it as a separate state channel scoped to the active drill level. This gives users a familiar interaction without coupling drilled visibility to parent cluster visibility.
