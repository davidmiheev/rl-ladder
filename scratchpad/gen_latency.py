import sys, gc
sys.path.insert(0, "experiments/c1_glft_baseline")
import config
todo = [('ethusdc', d) for d in ['2025-09-03','2025-09-05','2025-09-10','2025-09-12','2025-09-16',
                                 '2025-09-17','2025-09-19','2025-09-21','2025-09-23','2025-09-24']]
todo.append(('xrpusdt', '2025-09-25'))
for sym, d in todo:
    config.use_symbol(sym)
    if config.latency_path(d).exists():
        print(f"{sym} {d} have", flush=True); continue
    config.ensure_latency(d)
    gc.collect()
    print(f"{sym} {d} ok", flush=True)
print("done")
