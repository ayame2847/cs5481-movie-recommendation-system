"""Select an MF model on validation, then evaluate the frozen model on test.

Three configurations are declared before training. Each epoch is a complete
ALS update of users and movies. Validation Recall@10 selects both the epoch
and configuration, matching the team's LRURec selection metric. Exact ties
keep the earlier epoch/configuration. The test result is computed only after
selection is finished. No train+validation refit is performed.
"""

import os
# Small ALS solves are faster and easier to reproduce with one BLAS thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
import yaml

from mf import BiasedMF, digest, evaluate, load_shared, recommend

TEAM_COMMIT = '417d758ebe439c4a9a842c0d16503b0c8b1ff480'
REPOSITORY = Path(__file__).resolve().parents[1]


def verify_reference(project, data):
    """Require identical input bytes, population and item IDs before comparison."""
    folder = Path(project) / 'results/lrurec'
    reference = json.loads((folder / 'run_metadata.json').read_text())
    if data.input_hashes != reference['input_sha256']:
        raise ValueError('A input hashes differ from the recorded LRURec run')
    if data.seed != reference['seed'] or data.threshold != reference['threshold']:
        raise ValueError('Shared seed/relevance threshold differs from LRURec')
    expected = pd.read_csv(folder / 'user_population.csv').sort_values('UserID').reset_index(drop=True)
    pd.testing.assert_frame_equal(data.population, expected, check_dtype=False)
    mapping = json.loads((folder / 'item_mapping.json').read_text())
    if mapping != {str(m): i + 1 for i, m in enumerate(data.item_ids)}:
        raise ValueError('Candidate catalogue differs from LRURec')
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='Team repository with A processed data')
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    parser.add_argument('--config', type=Path, default=REPOSITORY / 'configs/mf.yaml')
    parser.add_argument('--epochs', type=int, help='Override the configured epoch limit')
    parser.add_argument('--patience', type=int, help='Override the configured patience')
    args = parser.parse_args()
    # Keep experiment choices in configs/, while the shared data configuration
    # remains authoritative for the seed, relevance threshold and input paths.
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    trials = [(entry['factors'], entry['regularisation']) for entry in config['trials']]
    if not trials or any(type(f) is not int or f < 1 or type(r) not in (int, float)
                         or not np.isfinite(r) or r <= 0 for f, r in trials):
        parser.error('trials require positive integer factors and finite positive regularisation')
    args.epochs = config['maximum_epochs'] if args.epochs is None else args.epochs
    args.patience = config['patience'] if args.patience is None else args.patience
    if type(args.epochs) is not int or type(args.patience) is not int:
        parser.error('epochs and patience must be integers')
    if args.epochs < 1 or args.patience < 1:
        parser.error('epochs and patience must be positive')
    data = load_shared(args.project)
    reference = verify_reference(args.project, data)
    args.output.mkdir(parents=True, exist_ok=False)
    data.population.to_csv(args.output / 'user_population.csv', index=False)
    (args.output / 'item_mapping.json').write_text(json.dumps({str(m): i + 1 for i, m in enumerate(data.item_ids)}, indent=2) + '\n')
    histories, selection = [], []
    selected_value, selected = -np.inf, None
    started = time.perf_counter()
    for trial, (factors, penalty) in enumerate(trials, 1):
        model = BiasedMF(factors, penalty, data.seed).initialise(data.frames['train'], data.user_ids, data.item_ids)
        best, stale, best_epoch = -np.inf, 0, 0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.epoch()
            rmse, loss = model.objective()
            metrics, _ = evaluate(model, data, 'val')
            row = {'trial': trial, 'factors': factors, 'regularisation': penalty, 'epoch': epoch,
                   'train_RMSE': rmse, 'train_objective': loss, 'seconds': time.perf_counter() - epoch_start, **metrics}
            histories.append(row)
            value = metrics['Recall@10']
            print(f'Trial {trial}, epoch {epoch:02d}: train RMSE={rmse:.5f}, val Recall@10={value:.6f}', flush=True)
            if value > best:
                best, best_epoch, stale = value, epoch, 0
            else:
                stale += 1
            if value > selected_value:
                selected_value = value
                selected = {k: row[k] for k in ['trial', 'factors', 'regularisation', 'epoch']}
                model.save(args.output / 'best_model.npz')
            # Save completed validation evidence progressively; no test scores
            # are produced during configuration/epoch selection.
            pd.DataFrame(histories).to_csv(args.output / 'validation_history.csv', index=False)
            if stale >= args.patience:
                break
        selection.append({'trial': trial, 'factors': factors, 'regularisation': penalty,
                          'best_epoch': best_epoch, 'completed_epochs': epoch, 'best_validation_Recall@10': best})
    training_seconds = time.perf_counter() - started
    pd.DataFrame(selection).to_csv(args.output / 'model_selection.csv', index=False)
    model = BiasedMF.load(args.output / 'best_model.npz')
    test_start = time.perf_counter()
    metrics, per_user = evaluate(model, data, 'test')
    test_seconds = time.perf_counter() - test_start
    per_user.to_csv(args.output / 'test_per_user.csv', index=False)
    pd.DataFrame([{'Model': 'MF', 'Seed': data.seed, 'Best_Epoch': selected['epoch'],
                   'N_Test_Users': len(data.test_users), **metrics}]).to_csv(args.output / 'mf_metrics.csv', index=False)
    for stage in ['test', 'final']:
        examples = []
        for user in data.test_users[:5]:
            rec = recommend(model, data, user, stage, 5)
            if len(rec) < 5:
                raise ValueError('Fewer than five unseen candidates for the demonstration')
            examples.append(rec)
        pd.concat(examples, ignore_index=True).to_csv(args.output / f'{stage}_top5.csv', index=False)
    lrurec_path = args.project / 'results/tables/10_lrurec_metrics.csv'
    reference_table = pd.read_csv(lrurec_path)
    if len(reference_table) != 1 or int(reference_table.N_Test_Users.iloc[0]) != len(data.test_users):
        raise ValueError('Reference metric table has a different test population')
    comparison = [{'Model': 'MF', 'N_Test_Users': len(data.test_users),
                   **{m: metrics[m] for m in ['Recall@10', 'NDCG@10']}, 'Evidence': 'current_run'},
                  {'Model': 'LRURec', 'N_Test_Users': len(data.test_users),
                   **{m: float(reference_table[m].iloc[0]) for m in ['Recall@10', 'NDCG@10']}, 'Evidence': 'team_recorded_run'}]
    pd.DataFrame(comparison).to_csv(args.output / 'comparison.csv', index=False)
    metadata = {
        'algorithm': 'biased_explicit_matrix_factorisation_ALS', 'status': 'shared_split_completed',
        'team_reference_commit': TEAM_COMMIT, 'seed': data.seed, 'threshold': data.threshold,
        'selected': selected, 'selection_metric': 'validation Recall@10; exact ties retain earlier trial/epoch',
        'declared_grid': [{'factors': f, 'regularisation': r} for f, r in trials],
        'maximum_epochs': args.epochs, 'patience': args.patience,
        'training_rows': len(data.frames['train']), 'training_users': data.frames['train'].UserID.nunique(),
        'validation_users': len(data.val_users), 'test_users': len(data.test_users), 'catalogue_items': len(data.item_ids),
        'training_policy': 'All original train ratings, including low ratings; observed-only squared loss with L2 on biases and factors',
        'frozen_model': 'No validation/test ratings enter fitting; no refit after selection',
        'candidate_policy': reference['candidate_policy'], 'filter_policy': reference['filter_policy'],
        'tie_policy': 'Exact equal scores ordered by original MovieID ascending',
        'time_limitation': 'Shared train contains low ratings later than some held-out positives; MF fits these ratings. Not a strict temporal backtest.',
        'comparison_limitation': 'LRURec uses train-positive sequences plus validation context for test; MF uses all train ratings with frozen user vectors. Single seeds; unequal tuning budgets.',
        'input_sha256': data.input_hashes, 'movies_metadata_sha256': data.metadata_hash,
        'reference_metadata_sha256': digest(args.project / 'results/lrurec/run_metadata.json'),
        'reference_metrics_sha256': digest(lrurec_path),
        'test_metrics': metrics, 'validation_selection_recall10': selected_value,
        'training_validation_seconds': training_seconds, 'test_seconds': test_seconds,
        'checkpoint_sha256': digest(args.output / 'best_model.npz'),
        'experiment_config': config,
        'experiment_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
        'source_sha256': {n: digest(REPOSITORY / n) for n in
                          ['src/mf.py', 'src/run_mf.py', 'src/mf_demo.py', 'requirements-mf.txt']},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'PyYAML': yaml.__version__},
        'execution': 'CPU; BLAS thread limits set to one',
    }
    (args.output / 'metrics.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'selected': selected, 'test': metrics}, indent=2), flush=True)


if __name__ == '__main__':
    main()
