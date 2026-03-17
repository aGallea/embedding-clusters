# Sub-group Visibility Controls Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add hide/show/isolate/show-all controls for drilled sub-groups that affect only the current drilled level.

**Architecture:** Mirror the existing top-level visibility model for drilled mode using a separate drilled-level visibility state. Extend the bottom legend to expose the same controls in drilled mode, and make the plot renderers respect drilled visibility when rendering sub-group points.

**Tech Stack:** React, TypeScript, Zustand, Playwright E2E

---

### Task 1: Add drilled-level visibility state to the plot store

**Files:**
- Modify: `frontend/src/stores/plotStore.ts`

**Step 1: Add dedicated state for visible drilled sub-groups**

Introduce a separate visibility set for the current drilled level, rather than reusing `visibleClusters`.

**Step 2: Add drilled visibility actions**

Mirror the top-level API with drilled equivalents such as:

- toggle drilled sub-group visibility
- isolate drilled sub-group
- reset drilled sub-group visibility

**Step 3: Reset drilled visibility at the right lifecycle points**

Clear or rebuild drilled visibility when:

- drilling into a new level
- navigating back
- resetting drill

### Task 2: Add drilled-mode legend controls

**Files:**
- Modify: `frontend/src/components/plot/ClusterLegend.tsx`

**Step 1: Add Show All in drilled mode**

Render a drilled-mode **Show All** action that restores all sub-groups for the current drilled level.

**Step 2: Add eye icons to drilled sub-group rows**

Match top-level legend behavior:

- click = hide/show
- Ctrl/Cmd+click = isolate

**Step 3: Preserve existing drilled labels and counts**

Only add controls; do not regress the rest of the drilled legend display.

### Task 3: Make plot rendering respect drilled visibility state

**Files:**
- Modify: relevant drilled-mode plot renderers / shared visibility logic

**Step 1: Identify the renderer path that applies drilled color/dim logic**

Reuse the current drilled-mode branching rather than duplicating rendering code.

**Step 2: Hide points belonging to hidden drilled sub-groups**

In drilled mode, only render points whose `sub_cluster` index is visible in the drilled visibility set.

**Step 3: Keep top-level visibility behavior unchanged**

Top-level `visibleClusters` must continue to behave exactly as before.

### Task 4: Keep drilled visibility scoped to the current level only

**Files:**
- Modify: `frontend/src/stores/plotStore.ts`
- Potentially modify: components that trigger navigation/back/reset

**Step 1: Ensure drilled visibility does not leak across levels**

When moving between drill levels, visibility should reflect the active level only.

**Step 2: Ensure back navigation restores parent-level defaults**

On back/reset, the parent/top-level view should not inherit hidden drilled sub-groups from a deeper level.

### Task 5: Add E2E coverage for drilled visibility controls

**Files:**
- Modify: `frontend/e2e/cluster-drawer.spec.ts` and/or another legend-focused E2E file

**Step 1: Add drilled hide/show test**

Verify a drilled sub-group can be hidden and shown again via the legend eye icon.

**Step 2: Add drilled isolate test**

Verify Ctrl/Cmd+click isolates a drilled sub-group.

**Step 3: Add drilled Show All test**

Verify Show All restores all drilled sub-groups.

**Step 4: Add navigation-scope test**

Verify drilled visibility state does not leak after navigating back.

### Task 6: Verify frontend correctness

**Files:**
- Verify: `frontend/src/stores/plotStore.ts`
- Verify: `frontend/src/components/plot/ClusterLegend.tsx`
- Verify: relevant drilled-mode renderers
- Verify: updated E2E tests

**Step 1: Run frontend build**

```bash
cd frontend && npm run build
```

Expected: PASS

**Step 2: Run focused legend/drawer tests**

```bash
cd frontend && npx playwright test e2e/cluster-drawer.spec.ts --workers=1
```

Expected: PASS

**Step 3: Run full Playwright suite**

```bash
cd frontend && npx playwright test --workers=1
```

Expected: PASS

### Task 7: Final review and polish

**Files:**
- Review: store, legend, renderers, tests

**Step 1: Check interaction consistency**

Manual checklist:

- drilled eye icon hide/show works
- drilled isolate works
- drilled Show All works
- top-level visibility still works

**Step 2: Keep naming clear**

Use state/action names that clearly separate top-level cluster visibility from drilled sub-group visibility.

## Risks

- Visibility logic can become tangled if drilled and top-level state are mixed or checked in the wrong order.
- Renderers may need careful updates so hidden drilled sub-groups disappear cleanly without breaking highlight/dim logic.
- Navigation lifecycle bugs could leave stale hidden sub-groups after moving between drill levels.

## Notes

- Do not reuse `visibleClusters` for drilled sub-groups.
- Scope drilled visibility to the active drilled level only.
