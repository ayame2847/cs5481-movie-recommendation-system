import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mf import BiasedMF, SharedData, evaluate, recommend


def fixture():
    columns = ['UserID', 'MovieID', 'Rating', 'Timestamp', 'label']
    train = pd.DataFrame([(1, 10, 5, 1, 1), (1, 20, 1, 2, 0), (2, 10, 2, 1, 0),
                          (2, 30, 5, 2, 1), (3, 20, 2, 1, 0)], columns=columns)
    val = pd.DataFrame([(1, 30, 4, 3, 1), (2, 20, 5, 3, 1), (4, 60, 5, 3, 1)], columns=columns)
    test = pd.DataFrame([(1, 40, 5, 4, 1), (2, 50, 4, 4, 1), (3, 60, 5, 4, 1), (4, 70, 4, 4, 1)], columns=columns)
    movies = pd.DataFrame({'MovieID': list(range(10, 90, 10)), 'Title': [f'M{i}' for i in range(8)], 'Genres': ['Drama'] * 8})
    return SharedData({'train': train, 'val': val, 'test': test}, movies)


class MFTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.model = BiasedMF(2, 1, 42).initialise(self.data.frames['train'], self.data.user_ids, self.data.item_ids)

    def test_shared_cohort_excludes_no_training_positive_or_no_val(self):
        self.assertEqual(self.data.test_users, [1, 2])
        self.assertEqual(self.data.val_users, [1, 2])
        self.assertEqual(list(self.data.item_ids), [10, 20, 30, 40, 50, 60, 70])

    def test_als_block_matches_independent_ridge_solution(self):
        m = self.model
        movie_index = [self.data.item_index[10], self.data.item_index[20]]
        x = np.column_stack((np.ones(2), m.Q[movie_index]))
        y = np.array([5, 1]) - m.mean - m.bi[movie_index]
        expected = np.linalg.inv(x.T @ x + np.eye(3)) @ x.T @ y
        m._update(m.matrix, m.Q, m.bi, m.P, m.bu)
        np.testing.assert_allclose(np.r_[m.bu[0], m.P[0]], expected, atol=1e-12)

    def test_als_objective_is_nonincreasing(self):
        previous = self.model.objective()[1]
        for _ in range(5):
            self.model.epoch()
            current = self.model.objective()[1]
            self.assertLessEqual(current, previous + 1e-10)
            previous = current

    def test_missing_ratings_do_not_enter_loss(self):
        m = self.model
        m.P[:] = 0; m.Q[:] = 0; m.bu[:] = 0; m.bi[:] = 0
        expected = np.sum((self.data.frames['train'].Rating - m.mean) ** 2)
        self.assertAlmostEqual(m.objective()[1], expected)

    def test_exact_rank_and_binary_metrics(self):
        self.model.scores = lambda users: np.tile(np.arange(1, 8, dtype=float), (len(users), 1))
        metrics, rows = evaluate(self.model, self.data, 'test')
        self.assertEqual(list(rows['rank']), [4, 3])
        self.assertEqual(metrics['Recall@5'], 1)
        self.assertAlmostEqual(metrics['NDCG@5'], (1 / np.log2(5) + 1 / np.log2(4)) / 2)

    def test_seen_filter_excludes_low_ratings_and_correct_history(self):
        self.assertEqual(self.data.seen(1, 'val'), {10, 20})
        self.assertEqual(self.data.seen(1, 'test'), {10, 20, 30})
        self.assertEqual(set(recommend(self.model, self.data, 1, 'final', 10).MovieID), {50, 60, 70})

    def test_score_ties_use_original_movie_id(self):
        self.model.scores = lambda users: np.ones((len(users), 7))
        self.assertEqual(list(recommend(self.model, self.data, 1, 'final', 3).MovieID), [50, 60, 70])
        _, rows = evaluate(self.model, self.data, 'test')
        self.assertEqual(list(rows['rank']), [1, 2])

    def test_heldout_values_cannot_change_training_fit(self):
        changed = {s: f.copy() for s, f in self.data.frames.items()}
        changed['test']['Rating'] = 4
        changed['val']['Rating'] = 4
        other = SharedData(changed, self.data.movies.reset_index())
        fitted = BiasedMF(2, 1, 42).initialise(other.frames['train'], other.user_ids, other.item_ids)
        fitted.epoch(); self.model.epoch()
        np.testing.assert_array_equal(fitted.P, self.model.P)
        np.testing.assert_array_equal(fitted.Q, self.model.Q)

    def test_cold_users_and_items_have_finite_fallbacks(self):
        self.model.epoch()
        np.testing.assert_allclose(self.model.scores([999])[0], self.model.mean + self.model.bi)
        self.assertTrue(np.isfinite(self.model.scores([4])).all())
        np.testing.assert_array_equal(self.model.Q[self.data.item_index[70]], 0)

    def test_checkpoint_roundtrip(self):
        self.model.epoch()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'; self.model.save(path)
            loaded = BiasedMF.load(path)
            np.testing.assert_array_equal(loaded.scores([1, 2, 999]), self.model.scores([1, 2, 999]))

    def test_invalid_labels_and_duplicate_pairs_fail(self):
        for kind in ('label', 'overlap'):
            frames = {s: f.copy() for s, f in self.data.frames.items()}
            if kind == 'label': frames['train'].loc[0, 'label'] = 0
            else: frames['test'].loc[0, 'MovieID'] = 10
            with self.assertRaises(ValueError): SharedData(frames, self.data.movies.reset_index())

    def test_metadata_is_required(self):
        with self.assertRaises(ValueError):
            SharedData(self.data.frames, self.data.movies.iloc[:-1].reset_index())


if __name__ == '__main__':
    unittest.main(verbosity=2)
