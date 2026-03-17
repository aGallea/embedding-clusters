# Sub-cluster Preview Card Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade expanded sub-cluster preview rows so they display richer product details in the same visual style as the main drawer item list while remaining display-only.

**Architecture:** Keep the current accordion and drill flow unchanged. Refactor the preview-row rendering inside `ClusterDetailDrawer.tsx` to reuse the main item-list structure in a compact, non-interactive form using the already available `SubClusterPoint` data.

**Tech Stack:** React, TypeScript, Zustand, Playwright E2E

---

### Task 1: Identify the shared visual structure from the main item list

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Locate the main item row markup**

Use the existing main item list as the styling reference:

- thumbnail when `imageField` exists
- item id
- `dist` line
- metadata rows

**Step 2: Define the compact preview contract**

For preview rows, keep:

- image when available
- item id
- optional `dist` line only if available from preview source
- first few metadata rows

Do not add click handlers or selected styling.

**Step 3: Keep the preview limit unchanged**

Continue showing at most 10 preview rows.

### Task 2: Refactor expanded sub-cluster preview rows to match the main list visually

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Replace the current minimal preview card markup**

Swap the simple title/id boxes for richer preview rows using the same layout pattern as the main list.

**Step 2: Render the thumbnail when available**

Use the same `imageField` logic already used in the main list.

**Step 3: Render item id as the primary label**

Do not invent a separate display title if the main list uses the id as the leading line.

**Step 4: Render metadata rows using the same filtering pattern**

Follow the same metadata display convention as the main list, excluding `imageField` and limiting the number of visible metadata rows.

### Task 3: Handle optional distance cleanly

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Inspect preview data shape**

Use only data that actually exists on `SubClusterPoint`.

**Step 2: Conditionally render distance**

If the preview source has no distance value, omit the `dist` line entirely.

**Step 3: Avoid fake values**

Do not compute or guess a distance value just to mirror the main list visually.

### Task 4: Keep preview rows display-only

**Files:**
- Modify: `frontend/src/components/plot/ClusterDetailDrawer.tsx`

**Step 1: Ensure preview rows are not buttons**

Use non-interactive wrappers such as `div` instead of clickable elements.

**Step 2: Remove selection affordances**

Do not reuse selected-row styling, highlighted borders, or `handleItemClick`.

**Step 3: Preserve drill controls outside the preview row**

Keep **Drill deeper** below the preview list so row content stays read-only.

### Task 5: Update E2E coverage for richer preview cards

**Files:**
- Modify: `frontend/e2e/cluster-drawer.spec.ts`

**Step 1: Extend the expanded preview test**

After opening a sub-cluster preview, assert richer item details are visible.

Good candidates:

- image is present when available
- id line is visible
- at least one metadata line is visible

**Step 2: Preserve the non-interactive contract**

Do not add selection assertions for preview rows.

**Step 3: Keep recursive drill coverage intact**

Ensure the explicit **Drill deeper** action test still passes after the richer preview markup change.

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

**Step 1: Check visual density**

Make sure the richer preview rows do not make the accordion unreadably tall.

**Step 2: Keep comments minimal**

Retain only comments that explain non-obvious logic.

**Step 3: Confirm behavior contract**

Manual checklist:

- preview rows look like the main list
- preview rows are display-only
- chevron still expands only
- drill deeper still requires the explicit button

## Risks

- Preview metadata may be less complete than the main cluster detail API, so some rows may still look lighter than the main list.
- Adding images and metadata could make the expanded panel too tall if spacing is not kept compact.
- E2E assertions for image presence may need to be resilient because some collections may not include image data.

## Notes

- Do not add a new backend fetch in this pass.
- Prefer reusing the main list visual pattern rather than introducing a second preview-card style.
