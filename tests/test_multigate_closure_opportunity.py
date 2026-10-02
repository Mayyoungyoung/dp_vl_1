import io
import itertools
import zipfile

import numpy as np
import pytest

from routeset.multigate import generate_dataset, path_validity, route_from_openings
from scripts.audit_multigate_closure_opportunity import (
    aggregate, analyze_parent, close_for_diagnostic, k2_selections, load_train_only)


def scene(gates=2):
    rows = []
    for x in (-.3, .3):
        intervals = np.zeros((4, 2))
        intervals[:gates] = [(-.75, -.25), (.25, .75)][:gates]
        if gates == 1:
            intervals[0] = (-.3, .3)
        rows.append(np.r_[x, .03, intervals.ravel(), np.arange(4) < gates])
    return np.concatenate([[-.92, 0, .5], [.92, 0, .5]] + rows).astype(np.float32)


def bank(value, gates=2):
    pairs = list(itertools.product(range(gates), repeat=2))
    return np.asarray([route_from_openings(value, *pair) for pair in pairs]), np.asarray([4*a+b for a,b in pairs])


def test_diagnostic_last_closure_retained_and_input_not_mutated():
    value = scene(1)
    original = value.copy()
    refs, _ = bank(value, 1)
    changed = close_for_diagnostic(value, 0, 0)
    np.testing.assert_array_equal(value, original)
    assert path_validity(refs, value)["valid"].all()
    assert not path_validity(refs, changed)["valid"].any()
    assert np.flatnonzero(changed != value).tolist() == [16]
    with pytest.raises(ValueError):
        close_for_diagnostic(value, 0, 1)


def test_oracle_is_one_fixed_pair_not_per_closure_cherry_pick():
    refs = np.zeros((3, 2, 3))
    refs[:, :, 1] = np.asarray([0, 1, 2])[:, None]
    responses = np.eye(3, dtype=bool)
    selected = k2_selections(refs, responses, np.ones(3, bool), .25)
    assert selected["response_oracle"] == (0, 1)
    assert selected["response_oracle_covered_solvable"] == 2
    assert selected["farthest"] == selected["geometric_dpp"] == (0, 2)
    assert selected["pair_count"] == 3


def test_dpp_map_and_farthest_exact_equal_with_stable_tie():
    refs = np.zeros((4, 3, 3))
    refs[:, :, 1] = np.asarray([-1, 0, 0, 1])[:, None]
    selected = k2_selections(refs, np.ones((2, 4), bool), np.ones(2, bool), .25)
    assert selected["farthest"] == selected["geometric_dpp"] == (0, 3)
    expected = -np.expm1(-np.square(refs[0]-refs[3]).mean() / .25**2)
    assert selected["geometric_dpp_determinant"] == expected


def test_complete_two_by_two_and_bad_prefix_are_distinguished():
    value = scene()
    refs, modes = bank(value)
    parent, closures = analyze_parent(value, refs, modes, refs, "p", "s", .25)
    assert len(closures) == 4 and all(row["has_solution"] for row in closures)
    assert parent["methods"]["joint4_actual"]["any_valid_solvable_closures"] == 1.
    assert parent["methods"]["joint4_prefix2_zero_adaptation"]["any_valid_solvable_closures"] == .75
    assert parent["methods"]["reference_response_oracle_k2"]["any_valid_solvable_closures"] == 1.
    assert parent["methods"]["reference_farthest_k2"]["any_valid_solvable_closures"] == 1.
    result = aggregate([parent], .05)
    assert result["screening"]["unadapted_prefix_gap_at_least_threshold"]
    assert not result["screening"]["mechanism_opportunity_beyond_geometric_selection"]
    assert result["screening"]["decision"] == "train_genuine_K2_saturation_before_any_new_mechanism"


def test_single_reference_duplicate_and_all_no_solution_not_discarded():
    value = scene(1)
    refs, modes = bank(value, 1)
    parent, closures = analyze_parent(value, refs, modes, np.repeat(refs, 4, axis=0), "p", "s", .25)
    assert parent["selections"]["farthest"] == (0, 0)
    assert parent["unsolvable_closure_count"] == 2
    assert all(not row["has_solution"] for row in closures)
    result = aggregate([parent], .05)
    assert result["overall"]["parents"] == 1
    assert result["overall"]["evaluable_parents"] == 0
    assert result["overall"]["methods"]["joint4_actual"]["any_valid_solvable_closures"] is None


def test_incomplete_reference_bank_and_wrong_prediction_shape_rejected():
    value = scene()
    refs, modes = bank(value)
    with pytest.raises(ValueError, match="one known positive"):
        analyze_parent(value, refs[:-1], modes[:-1], refs, "p", "s", .25)
    with pytest.raises(ValueError, match="four actual"):
        analyze_parent(value, refs, modes, refs[:2], "p", "s", .25)


def _npy(values):
    out = io.BytesIO()
    np.lib.format.write_array(out, np.asarray(values), allow_pickle=False)
    return out.getvalue()


def test_train_loader_never_decodes_nontrain_payload(tmp_path):
    # Metadata promises two rows, but payload deliberately contains only the
    # TRAIN row. A full-array read would fail on the missing DEV payload.
    path = tmp_path / "development.npz"
    with zipfile.ZipFile(path, "w") as z:
        for key, values in dict(splits=["TRAIN", "DEV_MODEL"], parent_ids=["p0", "p1"], scene_ids=["s0", "s1"]).items():
            z.writestr(key + ".npy", _npy(values))
        payload = dict(scenes=np.zeros((1, 34), np.float32), paths=np.zeros((1, 16, 24, 3), np.float32),
                       modes=np.zeros((1, 16), np.int64), path_mask=np.ones((1, 16), bool))
        for key, values in payload.items():
            raw = io.BytesIO()
            np.lib.format.write_array_header_1_0(raw, dict(descr=np.lib.format.dtype_to_descr(values.dtype),
                fortran_order=False, shape=(2,) + values.shape[1:]))
            raw.write(values.tobytes())
            z.writestr(key + ".npy", raw.getvalue())
    data = load_train_only(path, 1)
    assert data["scenes"].shape == (1, 34)
    assert data["splits"].tolist() == ["TRAIN"]


def test_locked_archive_or_crossrole_parent_rejected_before_payload(tmp_path):
    for splits, parents in ((["TRAIN", "TEST_LOCKED"], ["p0", "p1"]), (["TRAIN", "DEV_MODEL"], ["p0", "p0"])):
        path = tmp_path / "invalid.npz"
        with zipfile.ZipFile(path, "w") as z:
            for key, values in dict(splits=splits, parent_ids=parents, scene_ids=["s0", "s1"]).items():
                z.writestr(key + ".npy", _npy(values))
        with pytest.raises(ValueError):
            load_train_only(path, 1)


def test_fixture_all_present_closures_and_response_identity(tmp_path):
    path = generate_dataset(tmp_path / "toy.npz", train=8, dev_model=1, dev_score=0,
                            calibration=0, test_locked=0, ood_locked=0, seed=7)
    data = load_train_only(path, 8)
    for index, value in enumerate(data["scenes"]):
        refs, modes = data["paths"][index, data["path_mask"][index]], data["modes"][index, data["path_mask"][index]]
        prediction = refs[np.arange(4) % len(refs)]
        parent, changes = analyze_parent(value, refs, modes, prediction, str(index), str(index), .25)
        assert len(changes) == int(value[16:20].sum() + value[30:34].sum())
        assert parent["reference_types"] == len(refs)
