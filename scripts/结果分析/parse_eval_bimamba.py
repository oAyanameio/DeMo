import re, json, statistics as st

log = open('outputs/missing_v1_bimamba_logs/eval_rerun.log').read()
blocks = re.split(r'^===== ', log, flags=re.M)[1:]
latest = {}
for b in blocks:
    first = b.split('\n', 1)[0]          # e.g. "complete / ETH (/path/epoch=37.ckpt) ====="
    toks = first.split()
    cond, fold = toks[0], toks[2]
    m = re.search(r'TEST METRICS: (\{.*\})', b)
    if m:
        t = re.sub(r"tensor\(([-\d.eE+]+),?\s*device='cuda:\d+'\)", r'\1', m.group(1))
        d = json.loads(t.replace("'", '"'))
        latest[(cond, fold)] = {'cond': cond, 'fold': fold,
                                'ADE1': d['test_minADE1'], 'FDE1': d['test_minFDE1'],
                                'ADE6': d['test_minADE6'], 'FDE6': d['test_minFDE6'],
                                'MR': d['test_MR'], 'bFDE6': d['test_b-minFDE6']}
CONDS = ['complete', 'random_block2']
FOLDS = ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2']
rows = [latest[(c, f)] for c in CONDS for f in FOLDS if (c, f) in latest]
missing = [(c, f) for c in CONDS for f in FOLDS if (c, f) not in latest]
print(len(rows), 'rows; missing:', missing)

summary = {}
print()
print(f"{'cond':<15}{'ADE1':>16}{'FDE1':>16}{'ADE6':>16}{'FDE6':>16}   (5-fold mean±std)")
for c in CONDS:
    sub = [r for r in rows if r['cond'] == c and 'ADE6' in r]
    if len(sub) < 5:
        print(f'{c}: only {len(sub)} folds ok'); continue
    summary[c] = {}
    line = f'{c:<15}'
    for k in ['ADE1', 'FDE1', 'ADE6', 'FDE6']:
        v = [r[k] for r in sub]
        summary[c][k] = {'mean': st.mean(v), 'std': st.stdev(v)}
        line += f'{st.mean(v):>8.3f}±{st.stdev(v):.3f}'
    print(line)

if len(summary) == len(CONDS):
    print()
    print('vs complete (相对变化, 正=更差):')
    for c in CONDS[1:]:
        line = f'{c:<15}'
        for k in ['ADE1', 'FDE1', 'ADE6', 'FDE6']:
            base = summary['complete'][k]['mean']
            d = (summary[c][k]['mean'] - base) / base * 100
            line += f'{d:>15.1%}'
        print(line)
    print()
    for c in CONDS:
        print(c.ljust(15), ' '.join(f"{f}={latest[(c,f)]['FDE6']:.3f}" for f in FOLDS if (c,f) in latest))
    # 配对 t 检验 (n=5, df=4, 双侧临界 t=2.776)
    print()
    import math
    for c in CONDS[1:]:
        for k in ['ADE6', 'FDE6', 'MR']:
            diffs = [latest[(c,f)][k] - latest[('complete',f)][k] for f in FOLDS]
            md = st.mean(diffs); sd = st.stdev(diffs)
            t = md / (sd / math.sqrt(5)) if sd > 0 else 0.0
            sig = '*' if abs(t) > 2.776 else ''
            print(f'paired t {c} {k}: mean_diff={md:+.4f} t={t:+.2f}{sig}')

json.dump(rows, open('outputs/missing_v1_bimamba_eval_rerun.json', 'w'), indent=1)
print('saved outputs/missing_v1_bimamba_eval_rerun.json')
