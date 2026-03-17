# Interactive Sub-cluster Item Lists Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn drilled-mode expanded sub-cluster panels into the interactive item lists for that level, with scrolling and plot-linked product selection.

**Architecture:** Keep the current row-selection model for recursive compute, but replace preview-only panel content with full interactive item lists that reuse the existing drawer selection/highlight flow. Expanded sub-cluster panels become scrollable item containers and no longer use a 10-item preview cap.

**Tech Stack:** React, TypeScript, Zustand, Playwright E2E

---

### Task 1: Remove the fixed preview cap model

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Remove the “up to 10 products” assumption**

Stop slicing sub-cluster items to a fixed preview limit.

**Step 2: Remove preview-only copy**

Delete UI text that frames the expanded panel as a limited preview.

**Step 3: Preserve row/chevron behavior**

Do not change the separation of:

- row click = recursion source selection
- chevron = expand/collapse

### Task 2: Make expanded sub-cluster panels render full item lists

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Build full list data from sub-cluster points**

Derive all items belonging to the expanded sub-cluster from the existing current-level point data.

**Step 2: Reuse the main item-card layout pattern**

Render each item with the same general structure as the previous lower list:

- image when available
- id
- distance if available
- metadata rows

**Step 3: Keep product cards non-button-like only if still display-only**

In this task, they are no longer display-only, so prepare them for item selection behavior.

### Task 3: Reuse the existing global selection/highlight mechanism

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Wire drilled-mode item cards to the existing selection handler**

Reuse the same global selection flow that powers the old lower list:

- `selectedPointIds`
- `setSelectedPointIds`
- `setHighlightedIds`

**Step 2: Make selection global across expanded panels**

Do not scope selection to a single sub-cluster panel.

**Step 3: Apply selected styling**

Selected drilled-mode item cards should visually match the old selected item style as closely as practical.

### Task 4: Make expanded sub-cluster item lists scrollable

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Constrain expanded panel height**

The expanded sub-cluster item list should become its own scrollable region within the drawer.

**Step 2: Preserve drawer-level scrolling behavior**

Do not reintroduce the earlier “list won’t scroll” regression.

**Step 3: Keep multiple expanded states usable**

If the drawer still allows only one expanded row at a time, keep that behavior unless implementation naturally changes it.

### Task 5: Update E2E coverage for interactive drilled-mode item lists

**Files:**
- Modify: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Update the expansion test**

Assert expanded sub-cluster panels contain a real item list, not a fixed preview cap.

**Step 2: Add a test for drilled-mode item selection**

Click a drilled-mode product card and verify the selection state updates.

**Step 3: Verify highlight coupling**

Use the existing drawer/plot selection checks or store assertions to confirm that drilled-mode item selection highlights the same ids on the 3D plot.

**Step 4: Verify selection persists across panel changes**

Expand/select in one sub-cluster, then interact with another panel and confirm the original selection remains active.

### Task 6: Verify frontend correctness

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

### Task 7: Final review and polish

**Files:**
- Review: `frontend/src/components/plot/ClusterDetailDrawer.tsx`
- Review: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Confirm interaction separation**

Manual checklist:

- row click selects recursion source
- chevron expands/collapses
- product-card click toggles product selection

**Step 2: Check scroll usability**

Make sure long sub-cluster item lists remain usable inside the drawer.

**Step 3: Keep copy accurate**

Remove any remaining “preview” or “showing up to 10 products” language.

## Risks

- Mixing row selection, chevron expansion, and item-card selection could create event-propagation bugs if handlers are not isolated carefully.
- Full sub-cluster item rendering could increase drawer rendering cost for large sub-clusters.
- Scroll-container nesting needs to stay clean to avoid a regression of the prior scrolling bug.

## Notes

- Do not add a new backend fetch in this pass.
- Reuse the existing global item-selection/highlight mechanism rather than creating a second drilled-mode selection store.
