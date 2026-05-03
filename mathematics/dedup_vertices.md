# `dedup_vertices` — worked example

`dedup_vertices(verts, faces, tol=1e-7)` from `mathematics/marching_tetrahedra.py`
merges vertices that share a position within `tol` and rewrites every face
index accordingly. Marching tetrahedra emits every edge-crossing point once
per tet, so the raw output has 4–6× more vertices than necessary; without
this merge, adjacent triangles wouldn't share vertex indices and downstream
iso-line tracing would fragment.

## Tiny input

To make the arithmetic readable, use `tol = 0.1` (the real default is `1e-7`).

```python
verts = np.array([
    [0.0, 0.0, 0.0],   # 0
    [1.0, 0.0, 0.0],   # 1
    [0.0, 1.0, 0.0],   # 2
    [0.0, 0.0, 0.0],   # 3   ← duplicate of 0
    [1.0, 0.0, 0.0],   # 4   ← duplicate of 1
])
faces = np.array([
    [0, 1, 2],
    [3, 4, 2],         # same triangle as the first, but using the dup verts
])
```

We expect `unique_verts` to have **3** rows and both faces to collapse to
`[0, 1, 2]`.

## Step 1 — quantize to integer keys

```python
keys = np.round(verts / tol).astype(np.int64)
```

Dividing by `tol` and rounding maps every coordinate inside the same
`tol`-wide bin to the same integer. With `tol = 0.1`:

```
keys =
  [ 0,  0,  0]   # row 0
  [10,  0,  0]   # row 1
  [ 0, 10,  0]   # row 2
  [ 0,  0,  0]   # row 3
  [10,  0,  0]   # row 4
```

Equal positions now have **bit-identical** integer rows — perfect for the
sort-based grouping that follows.

## Step 2 — sort so identical rows become adjacent

```python
order = np.lexsort(keys.T)
sorted_keys = keys[order]
```

`np.lexsort` takes the **last** array as the primary sort key, so passing
`keys.T = [x_col, y_col, z_col]` sorts primarily by z, then y, then x.
Any consistent total order is fine here — the only thing dedup needs is
that equal rows end up next to each other.

```
order        = [0, 3, 1, 4, 2]
sorted_keys =
  [ 0,  0,  0]   ← was index 0
  [ 0,  0,  0]   ← was index 3
  [10,  0,  0]   ← was index 1
  [10,  0,  0]   ← was index 4
  [ 0, 10,  0]   ← was index 2
```

## Step 3 — flag run boundaries

```python
diff = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
```

`diff[i]` is `True` iff the row at sorted-position `i+1` is *different*
from the row at position `i`. Adjacent equal rows give `False`.

```
sorted_keys[:-1]  vs  sorted_keys[1:]      diff
  [ 0, 0, 0]          [ 0, 0, 0]           False
  [ 0, 0, 0]          [10, 0, 0]           True
  [10, 0, 0]          [10, 0, 0]           False
  [10, 0, 0]          [ 0,10, 0]           True

diff = [False, True, False, True]
```

## Step 4 — assign a unique-id in sorted order

```python
new_idx_sorted = np.concatenate([[0], np.cumsum(diff)])
```

`cumsum` over the booleans turns each `True` into "start a new group":

```
cumsum(diff)        = [0, 1, 1, 2]
new_idx_sorted      = [0, 0, 1, 1, 2]
```

Read row-by-row in sorted order: positions 0 and 1 are group 0, positions
2 and 3 are group 1, position 4 is group 2 — exactly the three distinct
positions we expected.

## Step 5 — invert the permutation back to the original ordering

```python
remap = np.empty(len(verts), dtype=np.int64)
remap[order] = new_idx_sorted
```

`order[i]` says "the i-th sorted row was originally row `order[i]`", so
`remap[order[i]] = new_idx_sorted[i]` gives "**original** row `order[i]`
belongs to deduped group `new_idx_sorted[i]`":

```
remap[0] = 0   (from order[0]=0, new_idx_sorted[0]=0)
remap[3] = 0   (from order[1]=3, new_idx_sorted[1]=0)
remap[1] = 1
remap[4] = 1
remap[2] = 2

remap = [0, 1, 2, 0, 1]
```

So original vertex 0 maps to deduped vertex 0, original 3 *also* maps to
deduped 0, originals 1 and 4 both map to deduped 1, etc.

## Step 6 — collect one representative per group

```python
n_unique = int(new_idx_sorted[-1]) + 1   # = 3
unique_verts = np.empty((n_unique, 3), dtype=verts.dtype)
unique_verts[remap] = verts              # last writer wins
```

This is fancy-indexed assignment: it walks `remap` and writes
`unique_verts[remap[i]] = verts[i]` for every `i`. When two originals
share a group (here `0` and `3`, or `1` and `4`), both write to the same
slot — but their positions agree within `tol` so "last writer wins"
is harmless.

```
unique_verts =
  [0.0, 0.0, 0.0]    ← written by row 0, then overwritten by row 3 (same)
  [1.0, 0.0, 0.0]    ← written by row 1, then overwritten by row 4 (same)
  [0.0, 1.0, 0.0]    ← written by row 2
```

## Step 7 — rewrite the faces

```python
new_faces = remap[faces]
```

Fancy-indexing `remap` with the face array translates every old vertex
index into its new deduped index in one shot:

```
faces        = [[0, 1, 2],
                [3, 4, 2]]
new_faces    = [[remap[0], remap[1], remap[2]],
                [remap[3], remap[4], remap[2]]]
             = [[0, 1, 2],
                [0, 1, 2]]
```

Both triangles now reference the same three deduped vertices — exactly
the property iso-line tracing relies on (adjacent triangles must share
indices, otherwise every level set fragments at every triangle boundary).

## Summary of what the function does

| Step | Operation | Purpose |
|---|---|---|
| 1 | `round(verts / tol)` | snap fuzzy floats into exact integer bins |
| 2 | `lexsort` | put equal bins next to each other |
| 3 | `any(diff)` along rows | detect group boundaries |
| 4 | `cumsum` | label each group with a new id |
| 5 | inverse permutation | translate sorted-order ids back to original order |
| 6 | `unique_verts[remap] = verts` | gather one position per group |
| 7 | `new_faces = remap[faces]` | rewrite face indices in one vectorised step |

Cost is dominated by the sort: **O(n log n)** in the number of input
vertices, with no Python-level loops — the whole routine is four numpy
calls plus an indexing assignment.
