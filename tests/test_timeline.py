"""Interval algebra. Expectations here are computed by hand, not by running
the code and pasting the output -- a golden test that learned its answer from
the implementation cannot catch the implementation being wrong."""

from __future__ import annotations

import numpy as np
import pytest

from convlab.timeline import Segments, resample_to_grid


class TestConstruction:
    def test_empty(self):
        s = Segments.empty()
        assert len(s) == 0 and s.total == 0.0 and not s

    def test_sorts_and_merges_overlaps(self):
        s = Segments.from_pairs([(5, 7), (0, 2), (1, 3)])
        assert s.bounds.tolist() == [[0, 3], [5, 7]]

    def test_merges_touching_intervals(self):
        s = Segments.from_pairs([(0, 1), (1, 2)])
        assert s.bounds.tolist() == [[0, 2]]

    def test_drops_degenerate(self):
        s = Segments.from_pairs([(1, 1), (2, 1), (3, 4)])
        assert s.bounds.tolist() == [[3, 4]]

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            Segments(np.zeros((3, 3)))

    def test_nested_interval_absorbed(self):
        s = Segments.from_pairs([(0, 10), (2, 3)])
        assert s.bounds.tolist() == [[0, 10]]


class TestMask:
    def test_from_mask_single_frame_has_one_frame_duration(self):
        mask = np.array([False, True, False])
        s = Segments.from_mask(mask, frame_hz=100.0)
        assert s.bounds.tolist() == [[0.01, 0.02]]

    def test_from_mask_multiple_runs(self):
        mask = np.array([True, True, False, False, True])
        s = Segments.from_mask(mask, frame_hz=10.0)
        assert s.bounds.tolist() == [[0.0, 0.2], [0.4, 0.5]]

    def test_mask_roundtrip(self):
        mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
        s = Segments.from_mask(mask, frame_hz=50.0)
        assert s.to_mask(mask.size, 50.0).tolist() == mask.tolist()

    def test_from_mask_all_false(self):
        assert len(Segments.from_mask(np.zeros(10, bool), 100.0)) == 0

    def test_from_mask_trailing_run_closes(self):
        s = Segments.from_mask(np.array([False, True, True]), frame_hz=10.0)
        assert s.bounds.tolist() == [[0.1, 0.3]]


class TestShaping:
    def test_merge_gaps(self):
        s = Segments.from_pairs([(0, 1), (1.1, 2), (5, 6)])
        assert s.merge_gaps(0.2).bounds.tolist() == [[0, 2], [5, 6]]

    def test_merge_gaps_boundary_is_inclusive(self):
        s = Segments.from_pairs([(0, 1), (1.2, 2)])
        assert len(s.merge_gaps(0.2)) == 1
        assert len(s.merge_gaps(0.19)) == 2

    def test_drop_short(self):
        s = Segments.from_pairs([(0, 0.05), (1, 3)])
        assert s.drop_short(0.1).bounds.tolist() == [[1, 3]]

    def test_pad_and_remerge(self):
        s = Segments.from_pairs([(1, 2), (2.1, 3)])
        assert s.pad(0.1).bounds.tolist() == [[0.9, 3.1]]

    def test_pad_respects_limit(self):
        s = Segments.from_pairs([(0.05, 1)])
        assert s.pad(0.2, limit=(0.0, 10.0)).bounds.tolist() == [[0.0, 1.2]]

    def test_clip(self):
        s = Segments.from_pairs([(0, 5), (6, 8)])
        assert s.clip(1, 7).bounds.tolist() == [[1, 5], [6, 7]]


class TestSetAlgebra:
    def test_union(self):
        a = Segments.from_pairs([(0, 2)])
        b = Segments.from_pairs([(1, 4), (6, 7)])
        assert a.union(b).bounds.tolist() == [[0, 4], [6, 7]]

    def test_intersect_partial(self):
        a = Segments.from_pairs([(0, 5)])
        b = Segments.from_pairs([(3, 8)])
        assert a.intersect(b).bounds.tolist() == [[3, 5]]

    def test_intersect_many_to_many(self):
        a = Segments.from_pairs([(0, 4), (6, 10)])
        b = Segments.from_pairs([(2, 7), (9, 12)])
        assert a.intersect(b).bounds.tolist() == [[2, 4], [6, 7], [9, 10]]

    def test_intersect_disjoint_is_empty(self):
        a = Segments.from_pairs([(0, 1)])
        b = Segments.from_pairs([(2, 3)])
        assert len(a.intersect(b)) == 0

    def test_subtract_middle_splits(self):
        a = Segments.from_pairs([(0, 10)])
        b = Segments.from_pairs([(3, 5)])
        assert a.subtract(b).bounds.tolist() == [[0, 3], [5, 10]]

    def test_subtract_multiple_holes(self):
        a = Segments.from_pairs([(0, 10)])
        b = Segments.from_pairs([(1, 2), (4, 5), (9, 12)])
        assert a.subtract(b).bounds.tolist() == [[0, 1], [2, 4], [5, 9]]

    def test_subtract_covering_everything(self):
        a = Segments.from_pairs([(2, 4)])
        b = Segments.from_pairs([(0, 10)])
        assert len(a.subtract(b)) == 0

    def test_subtract_non_overlapping_is_identity(self):
        a = Segments.from_pairs([(0, 1), (5, 6)])
        b = Segments.from_pairs([(2, 3)])
        assert a.subtract(b).bounds.tolist() == [[0, 1], [5, 6]]

    def test_complement(self):
        s = Segments.from_pairs([(2, 4)])
        assert s.complement(0, 10).bounds.tolist() == [[0, 2], [4, 10]]

    def test_gaps(self):
        s = Segments.from_pairs([(0, 1), (2, 3), (5, 6)])
        assert s.gaps().bounds.tolist() == [[1, 2], [3, 5]]

    def test_gaps_needs_two(self):
        assert len(Segments.from_pairs([(0, 1)]).gaps()) == 0

    def test_subtract_is_inverse_of_union_on_disjoint(self):
        a = Segments.from_pairs([(0, 3)])
        b = Segments.from_pairs([(5, 7)])
        assert a.union(b).subtract(b).bounds.tolist() == a.bounds.tolist()


class TestQueries:
    def test_totals_and_durations(self):
        s = Segments.from_pairs([(0, 1.5), (4, 5)])
        assert s.total == pytest.approx(2.5)
        assert s.durations.tolist() == [1.5, 1.0]

    def test_contains_is_half_open(self):
        s = Segments.from_pairs([(1, 2)])
        assert s.contains(np.array([0.999, 1.0, 1.5, 2.0])).tolist() == [
            False, True, True, False
        ]

    def test_coverage(self):
        s = Segments.from_pairs([(0, 5)])
        assert s.coverage(0, 10) == pytest.approx(0.5)

    def test_overlap_duration(self):
        a = Segments.from_pairs([(0, 5)])
        b = Segments.from_pairs([(4, 20)])
        assert a.overlap_duration(b) == pytest.approx(1.0)

    def test_span(self):
        assert Segments.from_pairs([(1, 2), (8, 9)]).span == (1.0, 9.0)


class TestResampling:
    def test_linear_interpolation(self):
        out = resample_to_grid(
            np.array([0.0, 1.0]), np.array([0.0, 10.0]), n_frames=3, frame_hz=2.0
        )
        assert out.tolist() == pytest.approx([0.0, 5.0, 10.0])

    def test_nan_outside_source_range(self):
        out = resample_to_grid(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_frames=5, frame_hz=1.0
        )
        assert np.isnan(out[0]) and np.isnan(out[3]) and np.isnan(out[4])
        assert out[1] == pytest.approx(1.0) and out[2] == pytest.approx(2.0)

    def test_long_gap_is_not_interpolated_across(self):
        # A 5 s hole must not become a smooth ramp.
        t = np.array([0.0, 0.5, 5.5, 6.0])
        y = np.array([1.0, 1.0, 9.0, 9.0])
        out = resample_to_grid(t, y, n_frames=13, frame_hz=2.0, max_gap_s=1.0)
        assert out[0] == pytest.approx(1.0)
        assert np.isnan(out[3:11]).all(), "gap should be NaN, not interpolated"
        assert out[11] == pytest.approx(9.0)

    def test_short_gap_is_interpolated(self):
        t = np.array([0.0, 0.4])
        y = np.array([0.0, 4.0])
        out = resample_to_grid(t, y, n_frames=5, frame_hz=10.0, max_gap_s=1.0)
        assert out.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_empty_source(self):
        out = resample_to_grid(np.array([]), np.array([]), n_frames=4, frame_hz=1.0)
        assert np.isnan(out).all()

    def test_unsorted_source_is_handled(self):
        out = resample_to_grid(
            np.array([1.0, 0.0]), np.array([10.0, 0.0]), n_frames=3, frame_hz=2.0
        )
        assert out.tolist() == pytest.approx([0.0, 5.0, 10.0])

    def test_nan_values_are_skipped_not_propagated(self):
        t = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, np.nan, 2.0])
        out = resample_to_grid(t, y, n_frames=3, frame_hz=2.0)
        assert out.tolist() == pytest.approx([0.0, 1.0, 2.0])
