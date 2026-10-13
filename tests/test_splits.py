import numpy as np
import pytest

from bew.splits import assert_disjoint, paper_split, random_group_splits


def test_paper_split_sizes_and_membership():
    s = paper_split(41, 43, 40)
    assert len(s["train"]) == 41 and len(s["primary_test"]) == 43 and len(s["secondary_test"]) == 40
    assert 0 in s["primary_test"] and 83 in s["primary_test"] and 1 in s["train"]
    assert 82 in s["primary_test"] and 81 in s["train"]
    assert set(s["secondary_test"]) == set(range(84, 124))
    allidx = np.concatenate(list(s.values()))
    assert len(allidx) == len(set(allidx)) == 124


def test_random_group_splits_are_by_group():
    groups = np.repeat(np.array([f"c{i}" for i in range(20)]), 5)
    for tr, te in random_group_splits(groups, 10, 0.3, seed=1):
        assert not set(groups[tr]) & set(groups[te])
        assert len(set(groups[te])) == 6
        assert len(tr) + len(te) == len(groups)


def test_assert_disjoint_raises():
    g = np.array(["a", "a", "b"])
    with pytest.raises(AssertionError):
        assert_disjoint(g, np.array([0]), np.array([1]))
