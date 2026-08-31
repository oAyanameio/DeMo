import re, json, statistics as st

log = open('outputs/missing_v1_round1_logs/eval_rerun.log').read()
blocks = re.split(r'^===== ', log, flags=re.M)[1:]
latest = {}
for b in blocks:
    first = b.split('\n', 1)[0]          # e.g. "complete / ETH (rerun2) ====="
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
rows = [latest[(c, f)] for c in ['complete', 'random_single', 'random_block2']
        for f in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2'] if (c, f) in latest]
missing = [(c, f) for c in ['complete', 'random_single', 'random_block2']
           for f in ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2'] if (c, f) not in latest]
print(len(rows), 'rows; missing:', missing)

summary = {}
print()
print(f"{'cond':<15}{'ADE1':>16}{'FDE1':>16}{'ADE6':>16}{'FDE6':>16}   (5-fold mean±std)")
for c in ['complete', 'random_single', 'random_block2']:
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

if len(summary) == 3:
    print()
    print('vs complete (相对变化, 正=更差):')
    for c in ['random_single', 'random_block2']:
        line = f'{c:<15}'
        for k in ['ADE1', 'FDE1', 'ADE6', 'FDE6']:
            pct = (summary[c][k]['mean'] / summary['complete'][k]['mean'] - 1) * 100
            line += f'{pct:>+15.1f}%'
        print(line)

print()
for c in ['complete', 'random_single', 'random_block2']:
    sub = [r for r in rows if r['cond'] == c and 'FDE6' in r]
    print(f"{c:<15}" + ' '.join(f"{r['fold']}={r['FDE6']:.3f}" for r in sub))

json.dump({'rows': rows, 'summary': summary},
          open('outputs/missing_v1_round1_eval_rerun.json', 'w'), indent=1)
print('saved outputs/missing_v1_round1_eval_rerun.json')
