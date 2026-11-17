"""项目 LRURec 入口：复用官方模型，明确实现数据、训练与统一评估。

官方模型来源：https://github.com/yueqirex/LRURec
论文：Zhenrui Yue, Yueqi Wang et al., WSDM 2024.
不读取官方 dataset.pkl，也不改写官方模型或原始 train/val/test。
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PIN = 'e02a559bd8c42342f5c53d69cefea2589becb22c'
KS = (1, 5, 10, 20, 50)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(project, threshold=4):
    frames = {s: pd.read_csv(project / 'data/processed' / f'{s}.csv')
              for s in ('train', 'val', 'test')}
    cols = ['UserID', 'MovieID', 'Rating', 'Timestamp', 'label']
    for name, f in frames.items():
        assert set(cols) <= set(f.columns), f'{name} 缺列'
        assert not f[cols].isna().any().any(), f'{name} 有空值'
        for col in cols:
            assert np.isfinite(f[col]).all() and (f[col] % 1 == 0).all(), (name, col)
        assert f.Rating.isin([1, 2, 3, 4, 5]).all()
        assert f.Timestamp.gt(0).all()
        assert (f.label == f.Rating.ge(threshold).astype(int)).all()
        assert not f.duplicated(['UserID', 'MovieID']).any()
        if name != 'train':
            assert f.label.eq(1).all()
            assert f.groupby('UserID').size().le(1).all()
    allf = pd.concat([f.assign(split=s) for s, f in frames.items()])
    assert allf.groupby(['UserID', 'MovieID']).split.nunique().le(1).all()
    # 固定全量已评分电影词表，只决定输出空间，不使用留出评分训练。
    raw_items = sorted(int(x) for x in allf.MovieID.unique())
    mapping = {raw: idx + 1 for idx, raw in enumerate(raw_items)}
    def sequences(f):
        f = f.sort_values(['UserID', 'Timestamp', 'MovieID'], kind='stable')
        return {int(u): [mapping[int(x)] for x in g.MovieID]
                for u, g in f.groupby('UserID')}
    tr, va, te = (frames[s] for s in ('train', 'val', 'test'))
    pos = tr[tr.label == 1]
    seq = sequences(pos)
    val = {u: xs[0] for u, xs in sequences(va).items()}
    test = {u: xs[0] for u, xs in sequences(te).items()}
    seen_train = {u: set(xs) for u, xs in sequences(tr).items()}
    seen_test = {u: seen_train.get(u, set()) | ({val[u]} if u in val else set())
                 for u in set(allf.UserID)}
    seen_all = {u: set(xs) for u, xs in sequences(allf).items()}
    vu = sorted(set(seq) & set(val))
    tu = sorted(set(seq) & set(val) & set(test))
    train_users = sorted(u for u, xs in seq.items() if len(xs) >= 2)
    assert train_users and vu and tu
    last = pos.groupby('UserID').Timestamp.max()
    vt = va.set_index('UserID').Timestamp
    tt = te.set_index('UserID').Timestamp
    assert (last.reindex(vu) <= vt.reindex(vu)).all(), '训练正例晚于验证目标'
    assert (vt.reindex(tu) <= tt.reindex(tu)).all(), '验证晚于测试'
    for u in vu:
        assert val[u] not in seen_train.get(u, set())
    for u in tu:
        assert test[u] not in seen_test[u]
    population = []
    for u in sorted(int(x) for x in allf.UserID.unique()):
        population.append(dict(UserID=u, n_train_positive=len(seq.get(u, [])),
            train_eligible=u in train_users, val_eligible=u in vu, test_eligible=u in tu,
            has_val=u in val, has_test=u in test))
    return dict(frames=frames, mapping=mapping, seq=seq, final_seq=sequences(allf[allf.label == 1]), val=val, test=test,
                seen_train=seen_train, seen_test=seen_test, seen_all=seen_all,
                val_users=vu, test_users=tu, train_users=train_users, population=population)


def windows(sequence, length):
    """每窗最多 length+1 个正例，移位一位构造真实下一项标签。
    stride=length，相邻窗口共享一个边界物品；每个相邻转移只训练一次。
    """
    for start in range(0, len(sequence) - 1, length):
        part = sequence[start:start + length + 1]
        yield part[:-1], part[1:]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def run(args):
    project = Path(args.project).resolve()
    out = Path(args.output).resolve()
    # 避免无意覆盖上一次实验，包括 smoke 与正式实验。
    assert not out.exists() or not any(out.iterdir()), f'输出目录非空，请换新目录：{out}'
    out.mkdir(parents=True, exist_ok=True)
    import yaml
    cfg = yaml.safe_load((project / 'configs/config.yaml').read_text(encoding='utf-8'))
    assert cfg['paths']['processed_dir'] == 'data/processed', '当前入口要求 data/processed；请先确认路径配置'
    data = prepare(project, cfg['labels']['positive_threshold'])
    pd.DataFrame(data['population']).to_csv(out / 'user_population.csv', index=False)
    write_json(out / 'item_mapping.json', data['mapping'])
    summary = {s: len(data[s]) for s in ('train_users', 'val_users', 'test_users')}
    summary['n_items'] = len(data['mapping'])
    write_json(out / 'population_summary.json', summary)
    print('用户及电影范围：', summary, flush=True)
    if args.audit_only:
        return

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    assert torch.cuda.is_available(), '当前入口用于 GPU 训练，请启用 GPU'
    official = Path(args.official).resolve()
    head = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == PIN, f'官方版本不是固定版本 {PIN}，当前是 {head}'
    # 直接比对模型源码，避免工作区改动与 commit 编号不一致。
    original_model = subprocess.check_output(['git', '-C', str(official), 'show', f'{PIN}:model/lru.py'])
    assert (official / 'model/lru.py').read_bytes() == original_model, '官方模型文件有本地修改'
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 不承诺跨设备、跨 PyTorch 版本逐位一致。
    model_cfg = dict(dataset_code='ml-1m', num_items=len(data['mapping']),
        bert_max_len=args.max_len, bert_hidden_units=64, bert_num_blocks=2,
        bert_dropout=0.2, bert_attn_dropout=0.2)
    spec = importlib.util.spec_from_file_location('official_lrurec_model', official / 'model/lru.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.LRU(SimpleNamespace(**model_cfg)).cuda()
    count = sum(p.numel() for p in model.parameters())
    metadata = dict(seed=seed, official_commit=head, model_sha256=digest(official / 'model/lru.py'),
        runner_sha256=digest(__file__), python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, pandas=pd.__version__, gpu=torch.cuda.get_device_name(0),
        configuration=vars(args), model_configuration=model_cfg, model_parameters=count,
        population=summary, threshold=cfg['labels']['positive_threshold'],
        candidate_policy='全部已评分电影词表；不按训练出现次数额外过滤；未训练物品保留并记录',
        filter_policy='val:全部train评分；test:全部train+val评分；final:全部train+val+test评分',
        time_limitation='沿用共享划分；低分train记录可能晚于留出正例；本模型只用训练正例建序列，过滤仍按共享split定义。',
        train_policy='训练正例序列；相邻下一项交叉熵；窗口length+1、stride=length；不伪造验证项',
        tie_policy='分数相同时按连续item编号升序（原MovieID升序）',
        input_sha256={s: digest(project / 'data/processed' / f'{s}.csv') for s in ('train','val','test')})
    train_items = set(x for xs in data['seq'].values() for x in xs)
    metadata['n_items_without_train_positive'] = len(data['mapping']) - len(train_items)
    write_json(out / 'run_metadata.json', metadata)
    print('模型参数量：', count, flush=True)

    def pad(xs):
        xs = xs[-args.max_len:]
        return [0] * (args.max_len - len(xs)) + xs

    class TrainData(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = [pair for u in data['train_users']
                            for pair in windows(data['seq'][u], args.max_len)]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, i):
            x, y = self.samples[i]
            return torch.tensor(pad(x)), torch.tensor(pad(y))

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TrainData(), batch_size=args.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    assert len(loader) > 0
    groups = [dict(params=[p for n,p in model.named_parameters() if 'bias' not in n and 'layer_norm' not in n], weight_decay=0.01),
              dict(params=[p for n,p in model.named_parameters() if 'bias' in n or 'layer_norm' in n], weight_decay=0.0)]
    optimizer = torch.optim.AdamW(groups, lr=0.001, eps=1e-9)

    def prediction_batch(users, stage):
        sequences = [data['final_seq'][u] if stage == 'final' else data['seq'][u] + ([data['val'][u]] if stage == 'test' else []) for u in users]
        x = torch.tensor([pad(xs) for xs in sequences], device='cuda')
        with torch.no_grad():
            scores = model(x)[0][:, -1, :]
        assert torch.isfinite(scores).all(), '预测中出现NaN/Inf'
        scores[:, 0] = -torch.inf
        seen = data['seen_train'] if stage == 'val' else data['seen_test'] if stage == 'test' else data['seen_all']
        for i,u in enumerate(users):
            ids = list(seen.get(u, set()))
            scores[i, ids] = -torch.inf
        # 固定处理并列得分，避免不同TopK实现带来不一致。
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        return scores, order

    def evaluate(stage, limited=False):
        model.eval()
        users = data[stage + '_users']
        if limited:
            users = users[:min(len(users), 128)]
        answers = data[stage]
        rows = []
        for offset in range(0, len(users), args.eval_batch_size):
            batch = users[offset:offset + args.eval_batch_size]
            scores, order = prediction_batch(batch, stage)
            for i,u in enumerate(batch):
                target = answers[u]
                assert torch.isfinite(scores[i, target]), '评估目标被错误过滤'
                rank = int((order[i] == target).nonzero()[0].item()) + 1
                row = dict(UserID=u, target_item=target, rank=rank)
                for k in KS:
                    hit = int(rank <= k)
                    row[f'Recall@{k}'] = hit
                    row[f'NDCG@{k}'] = hit / math.log2(rank + 1)
                    row[f'MRR@{k}'] = hit / rank
                rows.append(row)
        frame = pd.DataFrame(rows)
        means = {col: float(frame[col].mean()) for col in frame if '@' in col}
        return means, frame

    best, best_epoch, stale = -1.0, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = n_targets = 0
        for step, (x,y) in enumerate(loader):
            x,y = x.cuda(), y.cuda()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)[0]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0)
            assert torch.isfinite(loss), '训练损失NaN/Inf'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            valid = int(y.ne(0).sum())
            loss_sum += float(loss.detach()) * valid
            n_targets += valid
            if args.smoke and step >= 2:
                break
        metrics, _ = evaluate('val', limited=args.smoke)
        row = dict(epoch=epoch, loss=loss_sum / n_targets, **metrics)
        history.append(row)
        print(f'epoch={epoch} loss={row["loss"]:.4f} val_Recall@10={metrics["Recall@10"]:.6f}', flush=True)
        pd.DataFrame(history).to_csv(out / 'validation_history.csv', index=False)
        if metrics['Recall@10'] > best:
            best, best_epoch, stale = metrics['Recall@10'], epoch, 0
            torch.save(dict(model_state_dict=model.state_dict(), model_configuration=model_cfg,
                            best_epoch=epoch, best_val_recall10=best), out / 'best_model.pth')
        else:
            stale += 1
        if stale >= args.patience:
            break
    metadata['train_and_validation_seconds'] = time.perf_counter() - started
    metadata['best_epoch'] = best_epoch
    metadata['completed_epochs'] = epoch
    if args.smoke:
        metadata['status'] = 'smoke_only_no_test_metrics'
        write_json(out / 'run_metadata.json', metadata)
        print('SMOKE PASSED：仅短跑验证；未计算正式测试指标。', flush=True)
        return
    checkpoint = torch.load(out / 'best_model.pth', map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    started = time.perf_counter()
    # 冻结最佳模型后仅进行一次最终测试。
    metrics, individual = evaluate('test')
    metadata['test_seconds'] = time.perf_counter() - started
    individual.to_csv(out / 'test_per_user.csv', index=False)
    pd.DataFrame([dict(Model='LRURec', Seed=seed, Best_Epoch=best_epoch,
        N_Test_Users=len(individual), **metrics)]).to_csv(out / '10_lrurec_metrics.csv', index=False)
    print('FINAL TEST:', json.dumps(metrics), flush=True)

    inv = {v:k for k,v in data['mapping'].items()}
    movies = pd.read_csv(project / cfg['paths']['raw_dir'] / 'movies.dat', sep='::', engine='python',
                         names=['MovieID','Title','Genres'], encoding='latin-1').set_index('MovieID')
    sample = [u for u in (1,2,3,4,5) if u in data['test_users']]
    for stage, name in [('test','11_lrurec_test_top5.csv'), ('final','12_lrurec_final_top5.csv')]:
        rows = []
        scores, order = prediction_batch(sample, stage)
        for i,u in enumerate(sample):
            ids = order[i,:5].tolist()
            assert len(set(ids)) == 5 and all(torch.isfinite(scores[i,j]) for j in ids)
            seen = data['seen_test'][u] if stage == 'test' else data['seen_all'][u]
            assert not set(ids) & seen
            for rank,j in enumerate(ids,1):
                raw = inv[j]
                info = movies.loc[raw]
                row = dict(UserID=u, Rank=rank, MovieID=raw, Title=info.Title,
                           Genres=info.Genres, Pred_Score=float(scores[i,j]))
                if stage == 'test':
                    row.update(Test_Target_MovieID=inv[data['test'][u]], Hit_Test=int(j==data['test'][u]))
                rows.append(row)
        pd.DataFrame(rows).to_csv(out / name,index=False)
    metadata['status'] = 'completed'
    metadata['checkpoint_sha256'] = digest(out / 'best_model.pth')
    write_json(out / 'run_metadata.json', metadata)
    print('DONE：', out, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', required=True)
    p.add_argument('--official', default='/content/LRURec_project')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--max-len', type=int, default=200)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    assert args.epochs > 0 and args.max_len > 0 and args.patience > 0
    assert args.batch_size > 0 and args.eval_batch_size > 0
    run(args)

if __name__ == '__main__':
    main()
