"""Per-session rows for the dashboard's contest dataset.

    python experiments/c13_bybit_contest/dashboard/build.py contest.json   # from the repo root
"""
import json, sys
sys.path[:0] = ['experiments/c13_bybit_contest', 'experiments/c10_forward']
import contest as k, c8, forward as f
OUT = sys.argv[1]
FROZEN = {'XRPPERP': ['XRPUSDC OBI → XRPPERP'], 'SUIPERP': ['SUIUSDC OBI → SUIPERP'],
          'XRPUSDT': ['XRPUSDT OBI → XRPUSDT'], 'SUIUSDT': ['SUIUSDT OBI → SUIUSDT', 'SUIUSDC OBI → SUIUSDT'],
          'AVAXUSDT': ['AVAXUSDC OBI → AVAXUSDT']}
BLOCKS = [('training', k.TRAIN), ('validation', k.VALID), ('test', k.TEST)]
rows = []
def add(contract, entry, role, stage, cfg, dates):
    if any(c8.result(cfg, d) is None for d in dates):
        return  # a partial path would not be comparable
    for d in dates:
        r = c8.result(cfg, d)
        if r is None: continue
        block = next(b for b, ds in BLOCKS if d in ds)
        rows.append({'contract': contract, 'entry': entry, 'role': role, 'stage': stage, 'config': k.describe(cfg) + (f", skew {cfg['sets']['skew_bp']:g} bp" if 'skew_bp' in cfg['sets'] else ''),
                     'block': block, 'date': d, 'return_pct': round(r['Return'] * 100, 4),
                     'fills': r['DailyNumberOfTrades'], 'turnover': round(r.get('DailyTurnover', 0.0), 3)})
for contract in k.CONTRACTS:
    top = k.screen_pick(contract, 3)
    if top is None: continue
    for i, c in enumerate(top):
        add(contract, f'Screen #{i + 1}', 'leader' if i == 0 else 'runner-up', 'Stage 1: screen', c, k.SCREEN)
    add(contract, 'Screen #1 on all 48 training sessions', 'base', 'Stage 2: families', top[0], k.TRAIN)
    for rule in k.RULES:
        best = k.family_best(contract, rule)
        if best is None: continue
        r = k.ranked(list(best.values()), k.TRAIN, rule)
        name = next(n for n, v in best.items() if c8.key(v) == c8.key(r[0][0]))
        add(contract, f'Stage 2 best ({rule}): {name}', 'stage2' if rule == k.RULES[0] else 'stage2b',
            'Stage 2: families', r[0][0], k.TRAIN)
    for rule in k.RULES:
        cur, pending, added = k.greedy(contract, rule)
        if cur is None or not added: continue
        state = '' if not pending else ', still adding'
        add(contract, f'Stage 3 greedy ({rule}): ' + ' + '.join(added) + state, 'stage3' if rule == k.RULES[0] else 'stage3b',
            'Stage 3: greedy', cur, k.TRAIN)
    for rule in k.RULES:
        fp = k.final_pick(contract, rule)
        if fp is None: continue
        tested = c8.results(fp[1], k.TEST) is not None
        add(contract, f'Final pick ({rule}): {fp[0]}', 'pick' if rule == k.RULES[0] else 'pickb',
            'Stage 5: test' if tested else 'Stage 4: validation', fp[1], k.TRAIN + k.VALID + (k.TEST if tested else []))
    for label in FROZEN.get(contract, []):
        cfg = f.cfg(label, 'as frozen (-0.5 bp)')
        full = c8.results(cfg, k.TRAIN + k.VALID) is not None
        # The test block is shown only once a contest pick for the contract has been tested.
        tested = full and any(k.final_pick(contract, r) and c8.results(k.final_pick(contract, r)[1], k.TEST) for r in k.RULES)
        tested = tested and c8.results(cfg, k.TEST) is not None
        add(contract, 'C8 choice: ' + label.replace(' OBI', ''), 'c8', 'reference' if not full else
            ('reference, all blocks' if tested else 'reference, training + validation'),
            cfg, (k.TRAIN + k.VALID + (k.TEST if tested else [])) if full else k.SCREEN)
json.dump(rows, open(OUT, 'w'))
import collections; print(len(rows), collections.Counter((r['contract'], r['entry']) for r in rows))
