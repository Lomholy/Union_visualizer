---
satisfies: [R1, R2, R3, R4, R5, R6, R7]
---
# fn-1-union-priority-clipping-mesher.1 Implement Union priority-clipping mesher

## Description
TBD

## Acceptance
Every R-ID in the parent spec's ## Acceptance Criteria is satisfied; judge this task against the spec's criteria directly.

## Done summary
Added the "union" mesher (src/union_clip.py), a Python port of McStas Union's priority display clipping: boxes (tapered too), cylinders, cones, spheres and meshes, ALL/ANY masks, and the clip plane as a 10000 m box. Offered in the viewer dock and as --mesher union.

Optimised by running the recursion breadth first in numpy batches and tracing meshes through a BVH: NEUTRA builds in 1.7 s (5.4 s before optimisation; brep mesher 17.4 s on 4 cores, same machine). Areas vs McStas --trace=2: within 0.02% on NEUTRA and on tests_unit/data/union_clip_shapes.instr (R7 bound 0.2%).

Review fixes: rebuild signature includes cutters' masks; union incremental rebuild builds shapes once.

stage: impl-review - ran [3-reviewer panel: correctness SHIP, integration SHIP, contracts failed (usage limit, no vote); fixes re-reviewed SHIP] (model: claude-opus-5-5)

stage: plan-sync - skipped(config: planSync.enabled != true)
## Evidence
- Commits: 6cb2eb4417d4b71225aea44951a8590fd064a077, 2c1e1c4f65fe3a35bb3e6597fd105c924afa5ab6, d8a24187edff2ece423dfc535c5277f7d474a3b7, f730b3c093acf603f4316d683d3b95c63ff9bb97
- Tests: python -m unittest tests_unit.test_union_clip tests_unit.test_bbox_dependency tests_unit.test_gui_helpers tests_unit.test_cli tests_unit.test_union_viewer (100 OK)
- PRs: