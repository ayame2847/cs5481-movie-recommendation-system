"""Explicit-rating matrix factorisation for the CS5481 movie project.

The model estimates a rating as mu + user_bias + movie_bias + p_u @ q_i.
Only observed training ratings enter the squared-error objective; a missing
rating is not a zero-star vote. L2 regularisation penalises both biases and
latent vectors. The global training mean mu stays fixed.

Alternating least squares (ALS) makes this objective practical on a CPU.
Holding movie parameters fixed, each user has a small ridge-regression problem
whose predictors are [1, q_i]. The intercept becomes the user bias. We then
hold users fixed and solve the analogous problem for each movie. One pass
over both sides is an epoch. Initial factors use a local seeded generator.

The SharedData class defines the team's evaluation population and catalogue.
Validation/test ratings never enter an ALS update. Their IDs define targets,
the fixed catalogue and the history filter, following the repository protocol.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

KS = (1, 5, 10, 20, 50)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SharedData:
    """Validate A's tables and reconstruct LRURec's common comparison cohort."""

    def __init__(self, frames, movies, seed=42, threshold=4):
        self.frames = {s: f.copy() for s, f in frames.items()}
        self.seed, self.threshold = int(seed), float(threshold)
        required = ['UserID', 'MovieID', 'Rating', 'Timestamp', 'label']
        pairs = []
        for name in ('train', 'val', 'test'):
            f = self.frames[name]
            if f.empty or not set(required) <= set(f.columns):
                raise ValueError(f'{name}: nonempty table with {required} required')
            for col in required:
                if not pd.api.types.is_numeric_dtype(f[col]):
                    raise ValueError(f'{name}: {col} must be numeric')
                if not np.isfinite(f[col]).all() or (f[col] % 1 != 0).any():
                    raise ValueError(f'{name}: {col} must contain finite integers')
            if (f[['UserID', 'MovieID', 'Timestamp']] <= 0).any().any():
                raise ValueError(f'{name}: IDs and timestamps must be positive')
            if not f.Rating.between(1, 5).all() or not f.label.eq(f.Rating.ge(threshold).astype(int)).all():
                raise ValueError(f'{name}: rating or relevance-label mismatch')
            if f.duplicated(['UserID', 'MovieID']).any():
                raise ValueError(f'{name}: duplicate user/movie pair')
            if name != 'train' and (not f.label.eq(1).all() or f.UserID.duplicated().any()):
                raise ValueError(f'{name}: require one positive target per included user')
            pairs.append(set(zip(f.UserID, f.MovieID)))
        if any(pairs[i] & pairs[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError('Overlapping train/val/test pairs')
        allf = pd.concat(list(self.frames.values()), ignore_index=True)
        self.item_ids = np.sort(allf.MovieID.unique()).astype('int64')
        self.user_ids = np.sort(allf.UserID.unique()).astype('int64')
        self.item_index = {int(x): i for i, x in enumerate(self.item_ids)}
        self.user_index = {int(x): i for i, x in enumerate(self.user_ids)}
        if not {'MovieID', 'Title', 'Genres'} <= set(movies.columns):
            raise ValueError('MovieID, Title and Genres metadata required')
        if movies.MovieID.duplicated().any() or not set(self.item_ids) <= set(movies.MovieID):
            raise ValueError('Duplicate or missing movie metadata')
        self.movies = movies.set_index('MovieID').loc[self.item_ids, ['Title', 'Genres']].copy()
        if self.movies.isna().any().any() or any(self.movies[c].astype(str).str.strip().eq('').any() for c in self.movies):
            raise ValueError('Missing title or genres')
        self.histories = {
            s: {int(u): set(g.MovieID.astype(int)) for u, g in f.groupby('UserID')}
            for s, f in self.frames.items()
        }
        positive_counts = self.frames['train'].loc[lambda f: f.label == 1].groupby('UserID').size()
        seq_users = set(positive_counts.index)
        self.val_users = sorted(seq_users & set(self.frames['val'].UserID))
        self.test_users = sorted(seq_users & set(self.frames['val'].UserID) & set(self.frames['test'].UserID))
        if not self.val_users or not self.test_users:
            raise ValueError('No eligible validation/test users under the common protocol')
        self.population = pd.DataFrame([
            {'UserID': int(u), 'n_train_positive': int(positive_counts.get(u, 0)),
             'train_eligible': bool(positive_counts.get(u, 0) >= 2),
             'val_eligible': u in set(self.val_users), 'test_eligible': u in set(self.test_users),
             'has_val': u in self.histories['val'], 'has_test': u in self.histories['test']}
            for u in self.user_ids
        ])
        # This qualification table reproduces the sequence model's cohort.
        # MF itself fits ALL rows of train.csv, including users outside it.
        self.targets = {s: dict(zip(f.UserID.astype(int), f.MovieID.astype(int)))
                        for s, f in self.frames.items() if s != 'train'}

    def seen(self, user, stage):
        parts = {'val': ('train',), 'test': ('train', 'val'), 'final': ('train', 'val', 'test')}[stage]
        return set().union(*(self.histories[s].get(int(user), set()) for s in parts))


def load_shared(project):
    """Read existing A outputs; do not silently regenerate or change the split."""
    project = Path(project)
    config = yaml.safe_load((project / 'configs/config.yaml').read_text())
    folder = project / config['paths']['processed_dir']
    paths = {s: folder / f'{s}.csv' for s in ('train', 'val', 'test')}
    frames = {s: pd.read_csv(p) for s, p in paths.items()}
    movies_path = folder / 'movies_features.csv'
    data = SharedData(frames, pd.read_csv(movies_path), config['seed'], config['labels']['positive_threshold'])
    data.input_hashes = {s: digest(p) for s, p in paths.items()}
    data.metadata_hash = digest(movies_path)
    return data


class BiasedMF:
    """Minimise observed SSE + regularisation * squared parameter norms."""

    def __init__(self, factors=32, regularisation=10.0, seed=42):
        if not isinstance(factors, int) or factors < 1:
            raise ValueError('factors must be a positive integer')
        if not np.isfinite(regularisation) or regularisation <= 0:
            raise ValueError('regularisation must be finite and positive')
        self.factors, self.regularisation, self.seed = factors, float(regularisation), int(seed)

    def initialise(self, train, user_ids, item_ids):
        self.user_ids, self.item_ids = np.array(user_ids), np.array(item_ids)
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self.item_index = {int(m): i for i, m in enumerate(self.item_ids)}
        u = train.UserID.map(self.user_index).to_numpy()
        i = train.MovieID.map(self.item_index).to_numpy()
        if np.isnan(u).any() or np.isnan(i).any() or train.empty:
            raise ValueError('Training IDs must belong to the supplied catalogue')
        if train.duplicated(['UserID', 'MovieID']).any():
            raise ValueError('Duplicate training pairs')
        ratings = train.Rating.to_numpy(float)
        if not np.isfinite(ratings).all() or not ((ratings >= 1) & (ratings <= 5)).all():
            raise ValueError('Training ratings must be finite and in [1,5]')
        self.mean = float(ratings.mean())
        self.matrix = sparse.csr_matrix((ratings, (u, i)), shape=(len(user_ids), len(item_ids)))
        self.transpose = self.matrix.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.05, (len(user_ids), self.factors))
        self.Q = rng.normal(0, 0.05, (len(item_ids), self.factors))
        self.bu, self.bi = np.zeros(len(user_ids)), np.zeros(len(item_ids))
        self.P[np.diff(self.matrix.indptr) == 0] = 0
        self.Q[np.diff(self.transpose.indptr) == 0] = 0
        return self

    def _update(self, matrix, opposite_factors, opposite_bias, own_factors, own_bias):
        penalty = self.regularisation * np.eye(self.factors + 1)
        for index in range(matrix.shape[0]):
            start, end = matrix.indptr[index:index + 2]
            neighbours = matrix.indices[start:end]
            if not len(neighbours):
                own_bias[index], own_factors[index] = 0.0, 0.0
                continue
            # The first predictor is 1, so its coefficient is the bias.
            # Penalising this coefficient too matches the stated objective.
            design = np.column_stack((np.ones(len(neighbours)), opposite_factors[neighbours]))
            response = matrix.data[start:end] - self.mean - opposite_bias[neighbours]
            solution = np.linalg.solve(design.T @ design + penalty, design.T @ response)
            own_bias[index], own_factors[index] = solution[0], solution[1:]

    def epoch(self):
        self._update(self.matrix, self.Q, self.bi, self.P, self.bu)
        self._update(self.transpose, self.P, self.bu, self.Q, self.bi)

    def objective(self):
        """Return training RMSE and the exact penalised sum objective."""
        coo = self.matrix.tocoo()
        error = 0.0
        for start in range(0, len(coo.data), 50000):
            sl = slice(start, start + 50000)
            u, i = coo.row[sl], coo.col[sl]
            pred = self.mean + self.bu[u] + self.bi[i] + np.sum(self.P[u] * self.Q[i], axis=1)
            error += float(np.sum((coo.data[sl] - pred) ** 2))
        penalty = self.regularisation * sum(float(np.sum(x * x)) for x in [self.P, self.Q, self.bu, self.bi])
        return float(np.sqrt(error / len(coo.data))), error + penalty

    def scores(self, users):
        """Raw estimated ratings; do not clip away distinctions for ranking."""
        vectors = np.zeros((len(users), self.factors))
        bias = np.zeros(len(users))
        for n, u in enumerate(users):
            index = self.user_index.get(int(u))
            if index is not None:
                vectors[n], bias[n] = self.P[index], self.bu[index]
        return vectors @ self.Q.T + bias[:, None] + self.bi[None, :] + self.mean

    def save(self, path):
        np.savez_compressed(path, P=self.P, Q=self.Q, bu=self.bu, bi=self.bi, mean=self.mean,
                            user_ids=self.user_ids, item_ids=self.item_ids, factors=self.factors,
                            regularisation=self.regularisation, seed=self.seed)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            model = cls(int(z['factors']), float(z['regularisation']), int(z['seed']))
            for name in ['P', 'Q', 'bu', 'bi', 'user_ids', 'item_ids']:
                setattr(model, name, z[name].copy())
            model.mean = float(z['mean'])
        model.user_index = {int(u): i for i, u in enumerate(model.user_ids)}
        model.item_index = {int(i): j for j, i in enumerate(model.item_ids)}
        return model


def evaluate(model, data, stage):
    """Exact full-catalogue target ranks for the common leave-one-out cohort."""
    users = data.val_users if stage == 'val' else data.test_users
    rows = []
    for start in range(0, len(users), 128):
        batch = users[start:start + 128]
        scores = model.scores(batch)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite predictions')
        for n, user in enumerate(batch):
            blocked = [data.item_index[m] for m in data.seen(user, stage)]
            scores[n, blocked] = -np.inf
            target_id = data.targets[stage][user]
            target = data.item_index[target_id]
            value = scores[n, target]
            if not np.isfinite(value):
                raise ValueError('Held-out target excluded by history')
            # Count strictly better scores, then equal scores with lower IDs.
            # This is equivalent to sorting (-score, MovieID), without sorting
            # thousands of unneeded tail items for every evaluation user.
            rank = 1 + int(np.sum(scores[n] > value)) + int(np.sum((scores[n] == value) & (data.item_ids < target_id)))
            row = {'UserID': user, 'target_item': target + 1, 'target_MovieID': target_id, 'rank': rank}
            for k in KS:
                hit = int(rank <= k)
                row.update({f'Recall@{k}': hit, f'NDCG@{k}': hit / np.log2(rank + 1), f'MRR@{k}': hit / rank})
            rows.append(row)
    frame = pd.DataFrame(rows)
    return {c: float(frame[c].mean()) for c in frame if '@' in c}, frame


def recommend(model, data, user, stage='final', n=5):
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    scores = model.scores([user])[0]
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite predictions')
    blocked = data.seen(user, stage)
    allowed = np.array([i for i, m in enumerate(data.item_ids) if m not in blocked], dtype=int)
    indices = allowed[np.lexsort((data.item_ids[allowed], -scores[allowed]))[:n]]
    ids = data.item_ids[indices]
    result = data.movies.loc[ids].reset_index()
    result.insert(0, 'UserID', int(user))
    result.insert(1, 'rank', np.arange(1, len(result) + 1))
    result['score'] = scores[indices]
    return result
