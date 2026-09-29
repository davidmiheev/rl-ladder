"""C9e -- DAgger with a privileged teacher: learn which quotes to pull.

The oracle test showed that knowing the mid 1-3 s ahead is worth a lot on
ETHUSDC, and that a forecast of the move itself is far too weak to capture
it. So the student here does not forecast the move. It learns one decision
per side and step, **keep or pull this quote**, from a teacher that sees the
future:

* **Teacher (privileged).** For the quote the student actually posts at step
  t, would it be filled within the next second (the opposite touch reaches
  its price), and what would that fill earn, marked 5 s later against the
  quote price plus the maker rebate? A fill that loses is labelled "pull",
  one that earns "keep", each weighted by the money at stake. A quote that
  would not fill costs nothing either way and carries no weight.
* **Student (causal).** A boosted-tree score from market features only (the
  17 of `predict.py`, mirrored so one model serves both sides), plus a
  logistic layer over the student's own state that only the kernel knows
  when it quotes: the quote's distance from the mid, the inventory a fill
  would add to, and that side's markout toxicity. The kernel pulls the side
  (or widens it) when the probability exceeds `dag_thr`.
* **DAgger.** Round 0 labels the states the C8 rule visits; every later round
  rolls out the current student on the training sessions, labels the states
  *it* visits, adds them to the data and refits on everything so far.

Only training sessions are ever rolled out or labelled; the student's action
threshold and size are then chosen on training sessions like every other C9
variant, and reported held out and stress-tested (`c9.py --stage dagger`).

    python experiments/c9_fair_price/dagger.py rollout --round 0 --shard 0/2
    python experiments/c9_fair_price/dagger.py fit --round 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "experiments" / "c8_risk_controls")]

import c8  # noqa: E402
import features  # noqa: E402
import predict  # noqa: E402

ROOT = REPO / "runs" / "c9" / "dagger"
TARGETS = tuple(
    t for t in (("glft", "ethusdc"), ("obi", "ethusdc"))
    if t[1] in os.environ.get("C8_SYMBOLS", t[1]).split(",")
)
FILL_STEPS = 10  # a quote counts as filled if the opposite touch reaches it within 1 s
MARK_STEPS = 50  # and the fill is marked 5 s later
REBATE_BP = 0.5  # the default maker fee is a 0.5 bp rebate
#: Signed features change sign for the ask side (mirror), so one model scores both.
SIGNED = {"qi1", "micro_bp", "qi_2", "qi_10", "obi_z", "ofi_1", "ofi_5", "ofi_20", "ofi_100",
          "tfi_5", "tfi_20", "tfi_100", "ret_1", "ret_5", "ret_20", "ret_100"}
MIRROR = np.array([-1.0 if c in SIGNED else 1.0 for c in predict.FEATURES], np.float32)
#: Two teachers. `markout` (the first): pull a quote whose fill would lose
#: on a 5 s markout. It turned out misaligned -- a quote the opposite touch
#: reaches is mostly one the price runs through, so 58-69% of its labels say
#: "pull" and the student learns to pull nearly everything, while the
#: strategies earn over longer holds than 5 s. `pull:<k>:<q>` is the oracle
#: policy the large-move check showed to pay: pull a side just before a move
#: against it in the top (1-q) of k-step moves (on training sessions).
TEACHER = os.environ.get("C9_TEACHER", "markout")
TAG = "" if TEACHER == "markout" else "-" + TEACHER.replace(":", "")
#: The action rollouts use while the student is being trained; the final
#: threshold and action are chosen on training sessions afterwards.
ROLLOUT_ACTION = {"dag_thr": 0.5, "dag_widen_bp": 0.0} if TEACHER == "markout" else \
    {"dag_thr": 0.02, "dag_widen_bp": 0.0}


def student_sets(name: str, symbol: str, rnd: int, action: dict | None = None) -> dict:
    """Configuration overrides for the student `name` (strategy + teacher tag) after round `rnd`."""
    spec = json.loads((ROOT / f"{name}_{symbol}_r{rnd}.json").read_text())
    return {"fair_src": f"dag:{name}:{rnd}", "dag_mode": 1.0, **spec["weights"],
            **(action or ROLLOUT_ACTION)}


def rollout_config(strategy: str, symbol: str, rnd: int) -> dict:
    import c9

    base = c9.base_choice(strategy, symbol)
    return base if rnd == 0 else c8.on_top(base, student_sets(strategy + TAG, symbol, rnd - 1))


def series(strategy: str, rnd: int, symbol: str, date: str, f: dict) -> np.ndarray:
    """The kernel's input for a student: column 1/2 = market score for bid/ask."""
    import lightgbm as lgb

    x, _ = predict.build(f)
    booster = lgb.Booster(model_file=str(ROOT / f"{strategy}_{symbol}_r{rnd}.txt"))
    out = np.full((len(x), 3), np.nan)
    out[:, 1] = booster.predict(x, raw_score=True, num_threads=1)
    out[:, 2] = booster.predict(x * MIRROR, raw_score=True, num_threads=1)
    return out


def roll_path(strategy: str, symbol: str, rnd: int, date: str) -> Path:
    # Round 0 is the rule itself, the same for every teacher.
    return ROOT / "roll" / f"{strategy}{TAG if rnd else ''}_{symbol}_r{rnd}" / f"{date}.npz"


def rollout(strategy: str, symbol: str, rnd: int, date: str) -> None:
    """Replay one training session with round `rnd`'s policy; keep what the teacher needs."""
    import c9

    cfg = rollout_config(strategy, symbol, rnd)
    out = c9.run_one(cfg, date, keep_out=True)
    path = roll_path(strategy, symbol, rnd, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{date}.tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **{k: out[k].astype(np.float32) for k in
                                ("bid_px", "ask_px", "position", "mid", "tox_buy", "tox_sell")})
    os.replace(tmp, path)
    del out
    gc.collect()


def teacher(symbol: str, date: str, roll: dict) -> list[dict]:
    """Label every step at which a posted quote would be filled: one dict per side."""
    f = features.load(symbol, date)
    tick = float(f["tick"])
    ok = f["mid2"] >= 0
    bid_touch = np.where(ok, (f["mid2"] - f["spread"]) / 2.0 * tick, np.nan)
    ask_touch = np.where(ok, (f["mid2"] + f["spread"]) / 2.0 * tick, np.nan)
    mid = np.where(ok, f["mid2"] * tick / 2.0, np.nan)
    n = min(len(mid), len(roll["bid_px"]))
    lo = np.full(n, np.inf)  # lowest ask over the next FILL_STEPS steps
    hi = np.full(n, -np.inf)  # highest bid
    for k in range(1, FILL_STEPS + 1):
        lo[:-k] = np.fmin(lo[:-k], ask_touch[k:n])
        hi[:-k] = np.fmax(hi[:-k], bid_touch[k:n])
    later = np.full(n, np.nan)
    later[:-(FILL_STEPS + MARK_STEPS)] = mid[FILL_STEPS + MARK_STEPS:n]
    x, _ = predict.build(f)
    import strategies

    norm_pos = roll["position"][:n] * roll["mid"][:n] / strategies.MAX_POSITION_DOLLAR
    out = []
    for side, px, sign in ((0, roll["bid_px"][:n].astype(np.float64), 1.0),
                           (1, roll["ask_px"][:n].astype(np.float64), -1.0)):
        quoted = np.isfinite(px) & np.isfinite(later) & np.isfinite(mid[:n])
        with np.errstate(invalid="ignore"):
            reached = lo <= px if side == 0 else hi >= px
        idx = np.where(quoted & reached)[0]
        pnl_bp = sign * (later[idx] - px[idx]) / px[idx] * 1e4 + REBATE_BP
        out.append({
            "x": x[idx] * (MIRROR if side == 1 else 1.0),
            "dist": np.abs(px[idx] - mid[idx]) / mid[idx] * 1e4,
            "pos": norm_pos[idx] if side == 0 else -norm_pos[idx],
            "tox": (roll["tox_buy"] if side == 0 else roll["tox_sell"])[idx].astype(np.float64),
            "y": (pnl_bp < 0).astype(np.float64),
            "w": np.abs(pnl_bp),
        })
    return out


NEG_STRIDE = 20  # the pull teacher's "keep" steps, subsampled and reweighted


def teacher_pull(symbol: str, date: str, roll: dict) -> list[dict]:
    """The oracle large-move policy's action on every quoted step, per side."""
    import fair
    import strategies

    k, q = (float(x) for x in TEACHER.split(":")[1:])
    f = features.load(symbol, date)
    mid = fair.mid_of(f)
    move = fair.future_move_bp(mid, int(k))
    thr = fair.gate_threshold(q, f"oracle:{int(k)}", symbol)
    n = min(len(mid), len(roll["bid_px"]))
    x, _ = predict.build(f)
    norm_pos = roll["position"][:n] * roll["mid"][:n] / strategies.MAX_POSITION_DOLLAR
    out = []
    for side, px, against in ((0, roll["bid_px"][:n].astype(np.float64), -move[:n]),
                              (1, roll["ask_px"][:n].astype(np.float64), move[:n])):
        quoted = np.isfinite(px) & np.isfinite(against)
        pos_idx = np.where(quoted & (against >= thr))[0]
        neg = quoted & (against < thr)
        neg_idx = np.where(neg)[0][::NEG_STRIDE]
        idx = np.concatenate([pos_idx, neg_idx])
        out.append({
            "x": x[idx] * (MIRROR if side == 1 else 1.0),
            "dist": np.abs(px[idx] - mid[idx]) / mid[idx] * 1e4,
            "pos": norm_pos[idx] if side == 0 else -norm_pos[idx],
            "tox": (roll["tox_buy"] if side == 0 else roll["tox_sell"])[idx].astype(np.float64),
            "y": np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]),
            "w": np.concatenate([np.ones(len(pos_idx)), np.full(len(neg_idx), float(NEG_STRIDE))]),
        })
    return out


def _dataset(strategy: str, symbol: str, rounds: range):
    rows = []
    for rnd in rounds:
        for date in c8.TRAIN:
            path = roll_path(strategy, symbol, rnd, date)
            if not path.exists():
                raise SystemExit(f"missing rollout {path}")
            with np.load(path) as z:
                roll = {k: z[k] for k in z.files}
            label = teacher if TEACHER == "markout" else teacher_pull
            for part in label(symbol, date, roll):
                part["date"] = np.full(len(part["y"]), c8.TRAIN.index(date))
                rows.append(part)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def _logistic(z: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Weighted logistic regression by Newton's method; z includes a constant column."""
    beta = np.zeros(z.shape[1])
    w = w / w.mean()
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(z @ beta)))
        grad = z.T @ (w * (p - y)) / len(y) + l2 * beta
        hess = (z * (w * p * (1 - p))[:, None]).T @ z / len(y) + l2 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def fit(strategy: str, symbol: str, rnd: int) -> dict:
    """Fit the student on the teacher's labels from rounds 0..rnd."""
    import lightgbm as lgb

    data = _dataset(strategy, symbol, range(rnd + 1))
    last = len(c8.TRAIN) - 1  # the last training session is early stopping, never held out
    val = data["date"] == last
    booster = lgb.train(
        {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
         "min_data_in_leaf": 500, "feature_fraction": 0.8, "bagging_fraction": 0.7,
         "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 2},
        lgb.Dataset(data["x"][~val], data["y"][~val], weight=data["w"][~val],
                    feature_name=list(predict.FEATURES)),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(data["x"][val], data["y"][val], weight=data["w"][val])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    name = strategy + TAG
    booster.save_model(str(ROOT / f"{name}_{symbol}_r{rnd}.txt"))
    mkt = booster.predict(data["x"], raw_score=True)
    z = np.column_stack([np.ones(len(mkt)), mkt, data["dist"], data["pos"], data["tox"]])
    beta = _logistic(z, data["y"], data["w"])
    weights = dict(zip(("dag_b", "dag_w_mkt", "dag_w_dist", "dag_w_pos", "dag_w_tox"),
                       (float(b) for b in beta)))
    p = 1.0 / (1.0 + np.exp(-(z @ beta)))
    # What pulling at 0.5 would have saved on the labelled fills, per round's data.
    pulled = p > 0.5
    signed = np.where(data["y"] > 0, data["w"], -data["w"])
    report = {"round": rnd, "rows": int(len(data["y"])), "adverse_share": float(data["y"].mean()),
              "weighted_adverse": float((data["w"] * data["y"]).sum() / data["w"].sum()),
              "trees": booster.best_iteration, "weights": weights,
              "pull_share_at_0.5": float(pulled.mean()),
              "bp_saved_per_fill_at_0.5": float(signed[pulled].sum() / len(signed))}
    report["teacher"] = TEACHER
    if TEACHER != "markout":
        # How well the student ranks the teacher's pulls on the last
        # training session (the early-stopping one): weighted AUC, and
        # precision and recall at the rollout threshold.
        yv, wv, pv = data["y"][val], data["w"][val], p[val]
        order = np.argsort(pv)
        ranks = np.empty(len(pv))
        ranks[order] = np.cumsum(wv[order]) - wv[order] / 2.0
        pos_w, neg_w = wv[yv > 0].sum(), wv[yv == 0].sum()
        auc = ((ranks[yv > 0] * wv[yv > 0]).sum() - pos_w * pos_w / 2.0) / (pos_w * neg_w)
        hit = pv > ROLLOUT_ACTION["dag_thr"]
        report.update({
            "val_auc": float(auc),
            "precision_at_thr": float((wv * yv * hit).sum() / max((wv * hit).sum(), 1e-12)),
            "recall_at_thr": float((wv * yv * hit).sum() / pos_w),
            "pull_share_at_thr": float((wv * hit).sum() / wv.sum()),
        })
    (ROOT / f"{name}_{symbol}_r{rnd}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout")
    r.add_argument("--round", type=int, required=True)
    r.add_argument("--shard", default="0/1")
    fp = sub.add_parser("fit")
    fp.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    if args.cmd == "rollout":
        index, count = (int(x) for x in args.shard.split("/"))
        units = [(s, y, d) for s, y in TARGETS for d in c8.TRAIN
                 if not roll_path(s, y, args.round, d).exists()]
        units.sort(key=lambda u: (u[2], u[0]))
        for strategy, symbol, date in units[index::count]:
            rollout(strategy, symbol, args.round, date)
            print("rolled", strategy, symbol, args.round, date, flush=True)
    else:
        for strategy, symbol in TARGETS:
            fit(strategy, symbol, args.round)


if __name__ == "__main__":
    main()
