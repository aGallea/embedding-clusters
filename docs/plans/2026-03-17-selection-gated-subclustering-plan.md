# Selection-Gated Recursive Sub-clustering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hide the lower main product list after sub-clusters are created and require explicit sub-cluster row selection before recursive compute can run.

**Architecture:** Keep the backend and plot drill mechanics unchanged. Refine `ClusterDetailDrawer.tsx` so drilled mode becomes sub-cluster-focused: the lower item list disappears, local row-selection state determines the recursive source, and **Compute Sub-clusters** becomes the single recursive action in drilled mode.

**Tech Stack:** React, TypeScript, Zustand, Playwright E2E

---

### Task 1: Add selected sub-cluster state

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Add local selected-row state**

Add state such as:

```ts
const [selectedSubClusterIndex, setSelectedSubClusterIndex] = useState<number | null>(null)
```

**Step 2: Reset it when drawer context changes**

Clear selected row when:

- cluster changes
- current drilled level changes
- compute succeeds into a new level

**Step 3: Keep selection independent from expansion**

Do not overload `expandedSubClusterIndex`.

### Task 2: Make row click select the recursion source

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Add row selection handler**

Use row click to toggle the selected sub-cluster.

**Step 2: Add selected styling**

Make the selected row visually distinct.

**Step 3: Keep chevron behavior unchanged**

Chevron still expands/collapses preview only and should not trigger selection unless intentionally bubbled through a row click strategy is avoided.

### Task 3: Gate recursive compute on selected row

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Preserve top-level compute behavior**

Before sub-clusters exist, **Compute Sub-clusters** should still work on the selected cluster.

**Step 2: Change drilled-mode compute behavior**

When `isDrilled` is true:

- disable compute if `selectedSubClusterIndex === null`
- derive point ids from the selected sub-cluster only
- recurse using those point ids

**Step 3: Remove the whole-level drilled compute path**

Do not recurse on the entire current level anymore once sub-clusters already exist.

### Task 4: Remove redundant explicit drill button

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Remove the separate `Drill deeper` button from expanded rows**

Once compute is selection-gated, the explicit button duplicates the same action and should be removed.

**Step 2: Keep preview rows display-only**

Expanded preview content remains informational only.

### Task 5: Hide the lower main item list in drilled mode

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Gate the lower item list rendering**

When sub-clusters exist, do not render:

- the lower main product list
- pagination for that list

**Step 2: Keep non-drilled mode unchanged**

The initial cluster-detail experience should stay intact until the first sub-cluster compute happens.

### Task 6: Update E2E coverage for the new interaction model

**Files:**
- Modify: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Add a test for hidden lower list**

After first sub-cluster compute, assert the old lower product list is no longer visible.

**Step 2: Add a test for disabled compute**

In drilled mode, assert **Compute Sub-clusters** is disabled before row selection.

**Step 3: Add a test for row selection enabling compute**

Click a sub-cluster row, then assert compute becomes enabled.

**Step 4: Update recursive drill test**

Use row selection + compute instead of the explicit **Drill deeper** button.

### Task 7: Verify frontend correctness

**Files:**
- Verify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- Verify: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Run frontend build**

```bash
cd frontend && npm run build
```

Expected: PASS

**Step 2: Run focused drawer tests**

```bash
cd frontend && npx playwright test e2e/cluster-drawer.spec.ts --workers=1
```

Expected: PASS

**Step 3: Run full Playwright suite**

```bash
cd frontend && npx playwright test --workers=1
```

Expected: PASS

### Task 8: Final review and polish

**Files:**
- Review: `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- Review: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Confirm the mode switch is clear**

Manual checklist:

- top-level mode shows lower product list
- drilled mode hides lower product list
- row selection is visible
- compute is disabled until selection

**Step 2: Remove stale copy or comments**

Delete any labels or comments that still imply the old drill-button flow.

## Risks

- Users may not immediately realize row selection is required unless the selected styling and disabled-state messaging are clear.
- Row click and chevron click need careful event handling so expansion and selection do not conflict.
- Removing the lower list in drilled mode changes established drawer density, so layout spacing may need small polish.

## Notes

- Do not change backend endpoints in this pass.
- Keep the recursive source selection entirely in drawer-local UI state.
