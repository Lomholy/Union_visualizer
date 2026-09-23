"""Tests for gui_helpers.py: material normalisation/grouping, the
vacuum/exit predicate, default colour assignment, and the mesher
capability table.

The GUI itself is not unit-testable without a display, so this covers the
display-independent logic instead; union_viewer.py's own wiring was
exercised interactively with QT_QPA_PLATFORM=offscreen against real test
instruments while implementing this change.

Not under tests/: .github/workflows/run-tests.yml feeds every file there
to src/mcstas_to_cad.py as a pipeline smoke test.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gui_helpers  # noqa: E402
from meshing import MESHER_CAPABILITIES  # noqa: E402


class FakeComponent:
    """Stand-in for a mcstasscript component: the code under test only
    reads .material_string, so a plain object keeps these tests runnable
    without a McStas installation (mirrors tests_unit/test_tapered_box.py's
    FakeBox)."""

    def __init__(self, material_string):
        self.material_string = material_string


class TestNormalizeMaterialString(unittest.TestCase):
    def test_none_becomes_empty_string(self):
        self.assertEqual(gui_helpers.normalize_material_string(None), "")

    def test_plain_value_passes_through(self):
        self.assertEqual(gui_helpers.normalize_material_string("Al"), "Al")

    def test_embedded_quotes_are_stripped(self):
        # Confirmed on tests/crack_height.instr: one Union_mesh component's
        # material_string comes back as '"sample_0"' (quotes included) and
        # a sibling's comes back as plain 'gas' - the .instr text parser is
        # inconsistent about this, the same wrinkle
        # preprocess._resolve_relative_path already handles for filenames.
        self.assertEqual(gui_helpers.normalize_material_string('"sample_0"'), "sample_0")

    def test_single_quote_character_is_left_alone(self):
        # Only a matched pair of double quotes wrapping the whole string
        # counts - a single stray character isn't the same wrinkle.
        self.assertEqual(gui_helpers.normalize_material_string('"'), '"')

    def test_whitespace_is_trimmed(self):
        self.assertEqual(gui_helpers.normalize_material_string("  Al  "), "Al")


class TestIsVacuumMaterial(unittest.TestCase):
    def test_recognised_spellings(self):
        for value in ("vacuum", "Vacuum", "exit", "Exit"):
            with self.subTest(value=value):
                self.assertTrue(gui_helpers.is_vacuum_material(value))

    def test_quoted_spellings_are_still_recognised(self):
        self.assertTrue(gui_helpers.is_vacuum_material('"Vacuum"'))

    def test_other_casings_are_not_vacuum(self):
        # Union_box.comp checks material_string via a plain C strcmp
        # against exactly "vacuum"/"Vacuum" (and "exit"/"Exit") - no other
        # casing is special to McStas, so none should be treated as vacuum
        # here either, even though a human might read them the same way.
        for value in ("VACUUM", "VaCuUm", "EXIT", "vac", "Exiting"):
            with self.subTest(value=value):
                self.assertFalse(gui_helpers.is_vacuum_material(value))

    def test_ordinary_material_is_not_vacuum(self):
        self.assertFalse(gui_helpers.is_vacuum_material("Al"))

    def test_none_or_unset_is_not_vacuum(self):
        # An unset material_string means "auto-link a material", not
        # vacuum - McStas's own material_string=0/NULL default is a
        # distinct case from explicitly writing "vacuum".
        self.assertFalse(gui_helpers.is_vacuum_material(None))
        self.assertFalse(gui_helpers.is_vacuum_material("NULL"))
        self.assertFalse(gui_helpers.is_vacuum_material("0"))


class TestGroupComponentsByMaterial(unittest.TestCase):
    def test_groups_by_normalised_material(self):
        geometries = {
            "a": FakeComponent("Al"),
            "b": FakeComponent('"Al"'),  # same material, quoted differently
            "c": FakeComponent("Vacuum"),
        }
        groups = gui_helpers.group_components_by_material(geometries)
        self.assertEqual(groups, {"Al": ["a", "b"], "Vacuum": ["c"]})

    def test_unset_materials_share_one_group(self):
        geometries = {"a": FakeComponent(None), "b": FakeComponent(None)}
        groups = gui_helpers.group_components_by_material(geometries)
        self.assertEqual(groups, {"": ["a", "b"]})

    def test_preserves_first_appearance_order(self):
        geometries = {
            "z": FakeComponent("B"),
            "a": FakeComponent("A"),
            "y": FakeComponent("B"),
        }
        groups = gui_helpers.group_components_by_material(geometries)
        self.assertEqual(list(groups.keys()), ["B", "A"])

    def test_empty_input(self):
        self.assertEqual(gui_helpers.group_components_by_material({}), {})


class TestGroupMeshesByMaterial(unittest.TestCase):
    """trimesh is a real dependency of this module, so use real (tiny,
    cheap) trimesh.Trimesh boxes rather than mocking concatenation."""

    def _box(self, extent=1.0):
        import trimesh
        return trimesh.creation.box(extents=(extent, extent, extent))

    def test_single_component_material_returned_unconcatenated(self):
        mesh = self._box()
        geometries = {"a": FakeComponent("Al")}
        meshes = {"a": mesh}
        grouped = gui_helpers.group_meshes_by_material(geometries, meshes)
        self.assertEqual(set(grouped), {"Al"})
        self.assertIs(grouped["Al"], mesh)

    def test_multiple_components_are_concatenated(self):
        geometries = {"a": FakeComponent("Al"), "b": FakeComponent("Al")}
        meshes = {"a": self._box(), "b": self._box()}
        grouped = gui_helpers.group_meshes_by_material(geometries, meshes)
        self.assertEqual(set(grouped), {"Al"})
        # Two unit boxes concatenated: 8 vertices each, 12 faces each.
        self.assertEqual(len(grouped["Al"].vertices), 16)
        self.assertEqual(len(grouped["Al"].faces), 24)

    def test_none_mesh_is_excluded(self):
        """A mask component (its mesh is None - see
        brep.build_single_brep_mesh's mask_string check) must not break
        concatenation, and a material left with nothing but None meshes
        must not appear in the result at all."""
        geometries = {
            "a": FakeComponent("Al"),
            "b": FakeComponent("Al"),
            "c": FakeComponent("Mask"),
        }
        meshes = {"a": self._box(), "b": None, "c": None}
        grouped = gui_helpers.group_meshes_by_material(geometries, meshes)
        self.assertEqual(set(grouped), {"Al"})
        self.assertEqual(len(grouped["Al"].vertices), 8)

    def test_material_missing_from_meshes_dict_is_excluded(self):
        geometries = {"a": FakeComponent("Al")}
        meshes = {}
        grouped = gui_helpers.group_meshes_by_material(geometries, meshes)
        self.assertEqual(grouped, {})


class TestAssignDefaultColor(unittest.TestCase):
    def test_first_key_gets_first_cycle_color(self):
        colors = {}
        self.assertEqual(gui_helpers.assign_default_color(colors, "a"), "#E40303")

    def test_distinct_keys_cycle_through_the_palette(self):
        colors = {}
        assigned = [
            gui_helpers.assign_default_color(colors, k) for k in ("a", "b", "c")
        ]
        self.assertEqual(assigned, gui_helpers.DEFAULT_COLOR_CYCLE[:3])

    def test_existing_entry_is_left_untouched(self):
        colors = {"a": "#123456"}
        self.assertEqual(gui_helpers.assign_default_color(colors, "a"), "#123456")

    def test_cycle_wraps_around(self):
        colors = {}
        n = len(gui_helpers.DEFAULT_COLOR_CYCLE)
        for i in range(n + 2):
            gui_helpers.assign_default_color(colors, f"k{i}")
        self.assertEqual(colors["k0"], colors[f"k{n}"])

    def test_position_survives_deleting_and_reassigning_a_key(self):
        """Regression: deleting a stale entry and letting it be reassigned
        must not repeat the colour a sibling key was just given. This
        happened in practice when un-hiding an entry parked at the grey
        default: delete + reassign left len(colors) unchanged, so every
        freshly-reassigned key landed on the same cycle position."""
        colors = {}
        gui_helpers.assign_default_color(colors, "a")
        del colors["a"]
        first = gui_helpers.assign_default_color(colors, "a")
        second = gui_helpers.assign_default_color(colors, "b")
        self.assertNotEqual(first, second)

    def test_cycle_has_at_least_thirty_distinct_colors(self):
        cycle = gui_helpers.DEFAULT_COLOR_CYCLE
        self.assertGreaterEqual(len(cycle), 30)
        self.assertEqual(len(cycle), len(set(cycle)))

    def test_cycle_starts_with_the_requested_colors(self):
        self.assertEqual(
            gui_helpers.DEFAULT_COLOR_CYCLE[:11],
            [
                "#E40303", "#FF8C00", "#FFED00", "#008026", "#004DFF",
                "#750787", "#000000", "#613915", "#74D7EE", "#FFAFC8",
                "#FFFFFF",
            ],
        )


class TestMesherCapabilities(unittest.TestCase):
    """Cross-checks against what union_viewer.py actually offers and does,
    without needing a QApplication - see union_viewer.MESHER_KEYS, a
    module-level constant kept in sync with the dock's combo box precisely
    so this comparison doesn't need to construct one."""

    def test_covers_exactly_the_meshers_the_combo_offers(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import union_viewer

        self.assertEqual(set(MESHER_CAPABILITIES), set(union_viewer.MESHER_KEYS))

    def test_every_entry_has_the_same_capability_keys(self):
        expected_keys = {"resolution", "deflection", "clip", "incremental_rebuild"}
        for mesher, caps in MESHER_CAPABILITIES.items():
            with self.subTest(mesher=mesher):
                self.assertEqual(set(caps), expected_keys)

    def test_dc_is_marked_as_not_supporting_incremental_rebuild(self):
        # meshing.build_mesh has `if mesher == "dc": return` unconditionally
        # - this flag is what makes union_viewer.Viewer.reload_meshes force
        # a full rebuild instead of silently blanking a changed component.
        self.assertFalse(MESHER_CAPABILITIES["dc"]["incremental_rebuild"])

    def test_only_mc_uses_the_resolution_grid(self):
        self.assertTrue(MESHER_CAPABILITIES["mc"]["resolution"])
        self.assertFalse(MESHER_CAPABILITIES["dc"]["resolution"])
        self.assertFalse(MESHER_CAPABILITIES["brep"]["resolution"])

    def test_only_brep_uses_deflection(self):
        self.assertFalse(MESHER_CAPABILITIES["mc"]["deflection"])
        self.assertFalse(MESHER_CAPABILITIES["dc"]["deflection"])
        self.assertTrue(MESHER_CAPABILITIES["brep"]["deflection"])

    def test_all_three_meshers_respect_clip(self):
        # mc and dc both mesh final_sdfs, which already has the clip
        # half-space intersected in (signed_distance_functions.build_sdfs);
        # brep applies the same clip dict directly via brep.clip_component.
        for mesher in MESHER_CAPABILITIES:
            with self.subTest(mesher=mesher):
                self.assertTrue(MESHER_CAPABILITIES[mesher]["clip"])


class WrapLabelTest(unittest.TestCase):
    def test_short_names_are_unchanged(self):
        self.assertEqual(gui_helpers.wrap_label("sample"), "sample")
        self.assertEqual(gui_helpers.wrap_label("x" * 18), "x" * 18)

    def test_breaks_after_a_separator(self):
        self.assertEqual(
            gui_helpers.wrap_label("sample_holder_aluminium_can"),
            "sample_holder_\naluminium_can",
        )

    def test_cuts_mid_word_without_separators(self):
        self.assertEqual(gui_helpers.wrap_label("a" * 40), "\n".join(["a" * 18, "a" * 18, "a" * 4]))

    def test_every_line_fits_and_no_text_is_lost(self):
        name = "Guide_gravity_segment-12.upper wall_left"
        wrapped = gui_helpers.wrap_label(name)
        self.assertTrue(all(len(line) <= 18 for line in wrapped.split("\n")))
        self.assertEqual(wrapped.replace("\n", "").replace(" ", ""), name.replace(" ", ""))


class TraceHelpersTest(unittest.TestCase):
    def test_component_color_key_does_not_collide_with_union_names(self):
        self.assertNotEqual(gui_helpers.component_color_key("box"), "box")

    def test_clip_planes_keep_the_same_side_as_brep(self):
        # pygfx keeps points where a*x + b*y + c*z + d >= 0.
        clip = {"enable": True, "axis": "Z", "mode": "Above", "position": 2.0}
        (_, _, c, d), = gui_helpers.clip_planes(clip)
        self.assertGreater(c * 3.0 + d, 0)
        self.assertLess(c * 1.0 + d, 0)
        clip["mode"] = "Below"
        (_, _, c, d), = gui_helpers.clip_planes(clip)
        self.assertGreater(c * 1.0 + d, 0)
        self.assertLess(c * 3.0 + d, 0)

    def test_clip_disabled_has_no_planes(self):
        clip = {"enable": False, "axis": "X", "mode": "Above", "position": 0}
        self.assertEqual(gui_helpers.clip_planes(clip), [])

    def test_instrument_param_args(self):
        self.assertEqual(
            gui_helpers.instrument_param_args({"l_min": " 1 ", "l_max": "", "n": "3"}),
            ["l_min=1", "n=3"],
        )

    def test_clip_mesh_matches_clip_planes(self):
        import trimesh
        from clipping import resolve_clip_frame

        box = trimesh.creation.box(extents=(2, 2, 2))
        # A frame at x=0.5 rotated 90 degrees about z: its local x is world y.
        frame = np.eye(4)
        frame[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        frame[:3, 3] = (0.5, 0, 0)
        clip = resolve_clip_frame(
            {"enable": True, "axis": "X", "mode": "Above", "position": 0.25, "frame": "arm"},
            {"arm": frame},
        )
        clipped = gui_helpers.clip_mesh(box, clip)
        np.testing.assert_allclose(clipped.bounds, [[-1, 0.25, -1], [1, 1, 1]], atol=1e-9)
        self.assertTrue(clipped.is_watertight)
        (a, b, c, d), = gui_helpers.clip_planes(clip)
        self.assertGreater(b * 0.5 + d, 0)
        self.assertLess(b * 0.0 + d, 0)

    def test_clip_mesh_can_remove_everything(self):
        import trimesh

        box = trimesh.creation.box(extents=(1, 1, 1))
        clip = {"enable": True, "axis": "Z", "mode": "Above", "position": 5}
        self.assertIsNone(gui_helpers.clip_mesh(box, clip))
        clip["enable"] = False
        self.assertIs(gui_helpers.clip_mesh(box, clip), box)

if __name__ == "__main__":
    unittest.main()
