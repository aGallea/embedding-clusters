# Sub-cluster Row Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the right-sidebar sub-cluster chevron so it expands a sub-cluster row to show product previews, while recursive drill-down moves to a dedicated button inside the expanded row.

**Architecture:** Keep the existing recursive sub-cluster compute flow and store APIs intact. Refine `ClusterDetailDrawer.tsx` so current-level sub-clusters are rendered as an accordion with local expansion state, derive preview items from existing client data, and move recursive drilling to an explicit action inside expanded content.

**Tech Stack:** React, TypeScript, Zustand, Playwright E2E

---

### Task 1: Add accordion state for sub-cluster rows

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Add local state for the expanded row**

Add a single-expand accordion state near the other local drawer state:

```ts
const [expandedSubClusterIndex, setExpandedSubClusterIndex] = useState<number | null>(null)
```

**Step 2: Reset expansion when cluster/drill context changes**

Add/reset logic so expansion clears when:

- selected cluster changes
- a new sub-cluster compute finishes
- recursive drill succeeds

Expected behavior: stale expanded content never survives navigation into a different context.

**Step 3: Replace chevron intent**

Remove the current assumption that the chevron means drill-down. The row toggle handler should become:

```ts
const toggleSubClusterExpanded = (subClusterIndex: number) => {
  setExpandedSubClusterIndex((current) =>
    current === subClusterIndex ? null : subClusterIndex,
  )
}
```

**Step 4: Verify locally with typecheck via build**

Run:

```bash
cd frontend && npm run build
```

Expected: build still passes.

### Task 2: Derive preview items for an expanded sub-cluster

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Build a helper for sub-cluster point ids**

Add a helper that filters `currentLevel.subClusterData.points` by `sub_cluster` and returns matching ids.

**Step 2: Match those ids against available item records**

Use available drawer item data to derive preview items:

- first prefer `clusterDetail.items`
- only show a limited preview slice

Suggested shape:

```ts
const previewItems = clusterDetail?.items.filter((item) => subClusterPointIds.has(item.id)).slice(0, 10) ?? []
```

**Step 3: Keep the first version intentionally small**

Do not add nested pagination or a new backend request in this pass.

**Step 4: Verify empty-state behavior**

If no preview items are available from current client data, render a compact fallback message such as:

```tsx
<div className="text-[10px] text-gray-500">Preview not available for this page yet.</div>
```

### Task 3: Render expanded row content inside the drawer

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Convert the row to an accordion layout**

Keep the row header compact:

- color dot
- sub-cluster label
- count
- chevron button

**Step 2: Make the chevron expand/collapse only**

Replace the current `onClick={() => handleDrillSubCluster(sc.index)}` with the accordion toggle handler.

**Step 3: Render expanded content below the row header**

When `expandedSubClusterIndex === sc.index`, render:

- preview product list
- dedicated action area

**Step 4: Reuse compact item styling**

Use the drawer's existing visual language for preview rows, but keep them lighter than the main item list.

Expected UI: expanding a row reveals products instead of triggering recursion.

### Task 4: Add a dedicated “Drill deeper” button inside expanded content

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Keep `handleDrillSubCluster` as the recursive action**

Do not remove the existing recursive API logic. Reuse it for the new explicit button.

**Step 2: Move the drill trigger into expanded content**

Render a button such as:

```tsx
<button onClick={() => handleDrillSubCluster(sc.index)}>
  Drill deeper
</button>
```

**Step 3: Respect point-count constraints**

If `sc.count < 4`, disable or hide the button, but still allow the row to expand.

**Step 4: Preserve loading UX**

While recursive drill is running:

- disable the button
- keep the row visible
- show existing loading feedback text or spinner

### Task 5: Add stable test ids for the new drawer behavior

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Add test ids for row expansion**

Add stable selectors such as:

- `drawer-subcluster-toggle-{index}`
- `drawer-subcluster-panel-{index}`
- `drawer-subcluster-preview-{index}`
- `drawer-subcluster-drill-deeper-{index}`

**Step 2: Keep existing ids unless they are now misleading**

If `drawer-subcluster-drill-{index}` currently implies the chevron drills, rename it to reflect the new semantics.

**Step 3: Verify there are no duplicate ids**

Build once after updating the JSX.

### Task 6: Update E2E coverage for expand-vs-drill behavior

**Files:**
- Modify: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Update the existing drill-related tests**

Adjust tests so they no longer expect the chevron to recurse immediately.

**Step 2: Add/modify a test for expansion**

Test flow:

1. Compute plot
2. Open drawer
3. Compute sub-clusters
4. Click `drawer-subcluster-toggle-0`
5. Assert expanded panel becomes visible
6. Assert breadcrumb does **not** change just from expansion

**Step 3: Add/modify a test for preview rendering**

Assert the expanded panel contains at least one preview item or the preview fallback message.

**Step 4: Add/modify a test for explicit recursive drill**

Test flow:

1. Expand a sub-cluster row
2. Click `drawer-subcluster-drill-deeper-0`
3. Assert breadcrumb appears or deepens

### Task 7: Run focused verification

**Files:**
- Verify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- Verify: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Run frontend build**

```bash
cd frontend && npm run build
```

Expected: PASS

**Step 2: Run focused drawer E2E tests**

```bash
cd frontend && npx playwright test e2e/cluster-drawer.spec.ts --workers=1
```

Expected: PASS

**Step 3: If drawer tests pass, run full E2E suite**

```bash
cd frontend && npx playwright test --workers=1
```

Expected: PASS

### Task 8: Review and polish

**Files:**
- Review: `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- Review: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Remove any comments that no longer add value**

Keep only comments that explain non-obvious branching logic.

**Step 2: Check spacing and button labels**

Make sure the row header, expanded preview, and explicit drill action are visually distinct and readable.

**Step 3: Confirm the UX contract**

Manual checklist:

- chevron expands only
- compute button generates current-level sub-clusters
- drill deeper button performs recursive drill
- bottom legend remains display-only

## Risks

- Preview items may be incomplete if the currently loaded `clusterDetail.items` page does not contain many ids from the selected sub-cluster.
- The drawer could become visually crowded if preview rows are too large.
- Existing test ids may need careful migration to avoid brittle E2E failures.

## Notes

- Do not add a new backend endpoint in this pass unless the preview experience proves unusable.
- Prefer a minimal client-only implementation first.
- Keep commits focused and small if this plan is executed step-by-step.
