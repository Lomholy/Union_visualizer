# Union priority-clipping mesher (Python port of McStas display clipping)

## Conversation Evidence

> user (turn 1): "Could we improve the speed of it? McStas can do it much faster than ours does it."
> user (turn 2): "McStas does do this when used with mcdisplay-webgl --trace=2"
> user (turn 3): "Investigate a python port of McStas' clipping algorithm"
> user (turn 4): "Implement the shapes and masks as well. Treat the clip plane as a very big box (10000 m along the clip plane). Optimize afterwards, and open a PR on the Union_visualizer repository"

## Goal & Context

<!-- Source: 40% user / 30% [paraphrase] / 30% [inferred] -->

The viewer's brep mesher builds every Union component as an OpenCASCADE solid and subtracts each higher-priority component with Boolean cuts. On the NEUTRA instrument (five STL meshes, about 70k triangles) that takes about 31 s of CPU, against 1.3 s for McStas's own `--trace=2` display. McStas never builds solids. For each surface triangle of a volume it keeps only the parts where that volume has the highest priority, splitting the triangles that straddle a boundary.

A throwaway Python and numpy prototype of that algorithm, meshes only, reproduced McStas's NEUTRA output within 0.02% of surface area per volume, in 5.7 s single-process. This spec turns it into a real mesher covering every Union shape, masks and the clip plane, then optimises it.

## Architecture & Data Models

<!-- Source: 20% [paraphrase] / 80% [inferred] -->

- A third mesher (beside Marching Cubes, Dual Contouring and brep) takes the same inputs as the brep mesher (Union geometries, world matrices, clip settings) and returns one display triangle mesh per component. [inferred]
- Each Union shape answers two questions in world space: is a point inside it, and where a straight segment crosses its surface. The clipping algorithm needs nothing else from a shape. [inferred]
- The clipping follows McStas Union's priority display algorithm (`union_tri_add`, `union_edge_cuts`, `r_has_highest_priority` in McCode's `union-lib.c`), including its split budget and straight-cut tolerance. [paraphrase]
- When the clip plane is enabled, it is a box 10000 m along the clip plane, sitting on the removed side and taking priority over every component. [paraphrase]

## Edge Cases & Constraints

- Surfaces that two components share exactly, as CAD contact faces do, must not crash the mesher. A triangle may land on either side, as it does in McStas. [inferred]
- A component that is fully hidden by higher priorities, by its masks or by the clip box yields no mesh, and the viewer treats it as empty. [inferred]
- Curved shapes (cylinder, cone, sphere) are tessellated the way McStas tessellates them, so their output can be compared with McStas. [inferred]

## Acceptance Criteria

- **R1:** The viewer offers a new mesher that shows each Union component only where it has the highest priority, built with a Python port of McStas's clipping algorithm and no OpenCASCADE Boolean operations. [paraphrase] Errors: a component of a type the mesher does not know is skipped with a printed warning, and the other components still mesh. [inferred]
- **R2:** The mesher handles every Union shape: `Union_mesh`, `Union_box` (including tapered boxes), `Union_cylinder`, `Union_cone` and `Union_sphere`. [paraphrase] Errors: a mesh file that cannot be loaded raises the same error the brep mesher raises for it. [inferred]
- **R3:** Masks behave as in McStas. A masked component is shown only inside its masks: inside all of them for the ALL setting, inside at least one for ANY. Mask components themselves are not meshed. [paraphrase] (No error surface beyond R1.)
- **R4:** With the clip plane enabled, the clip plane is treated as a very big box (10000 m along the clip plane) that removes the clipped side of every component. Disabling the clip plane gives the unclipped result. [paraphrase] (No error surface beyond R1.)
- **R5:** For NEUTRA and for a test instrument using every shape and both mask settings, each component's surface area is within 0.1% of the matching McStas `--trace=2` polyhedron. [inferred] *(Superseded by R7.)*
- **R6:** After optimisation, the mesher builds NEUTRA faster than the brep mesher, with the before and after timings recorded on the same machine. [paraphrase]
- **R7:** Supersedes R5's tolerance. For NEUTRA and for a test instrument using every shape and both mask settings, each component's surface area is within 0.2% of the matching McStas `--trace=2` polyhedron. [paraphrase] (No error surface beyond R1.)

## Boundaries

- The clipped face at the clip plane is not filled in: components show open where the clip box removes them, as McStas does. [inferred]
- The brep, Marching Cubes and Dual Contouring meshers keep their current behaviour. [inferred]
- The mesher does not compile or run McStas. [inferred]

## Decision Context

R5's 0.1% bound was loosened to 0.2% (R7) at the user's choice. McStas's algorithm decides boundary vertices that a split places exactly on a curved surface by floating-point rounding: moving a cone by 1e-15 m changed a cylinder's clipped area by 0.02%, and a faithful port landed between 0.9985 and 0.9991 of McStas for that cylinder. The port stays an exact copy of the algorithm rather than being hardened away from it.

Using McStas's own `--trace=2` polyhedra was rejected for now. Master writes them under one component with no volume names, so mapping them to components is fragile, and it needs a compile on every geometry change. Tuning the OCC path (caching, sewing-free conversion, merging coplanar faces) was estimated at only 2–3× faster. The prototype showed the port is both accurate and much faster. [paraphrase]
