"""C1 -- the same measurements across every asset in the dataset.

    python experiments/c1_glft_baseline/run_cross_asset.py --hours 4

C1's result is a negative one, and a negative result on one symbol is a result
about that symbol.  This runs the phase's three load-bearing measurements on
one day of every symbol available (BTCUSDT excluded: a gigabyte a day, and
BTCUSDC covers the same asset), and plots them together:

1. the order-arrival intensity fit, which says whether GLFT's
   `lambda(delta) = A exp(-k delta)` describes the asset at all;
2. the touch spread and the volatility, in the units the model works in;
3. adverse selection -- how far the price moves against a maker filled by
   aggressive flow -- measured with no orders placed, so no fill model is
   involved.

The comparison is in basis points rather than ticks, because ticks are not
comparable across assets: a tick is 0.002 bp on BTCUSDC and 1.4 bp on ENAUSDC.
In bp the maker rebate is a constant 0.5 bp for every asset, which makes the
break-even condition one horizontal line on the plot.

Tick and lot sizes are detected from the data rather than hard-coded, and
written into the run's summary so `config.SYMBOLS` can be checked against them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import config  # noqa: E402
import glft  # noqa: E402

from rl_ladder.run import Run  # noqa: E402

DAY = "2025-08-26"

#: Horizons, in seconds, at which a maker's new position is marked to the mid.
HORIZONS = [0.1, 0.5, 1.0, 2.0, 5.0, 30.0]

#: The rebate, as a fraction of trade value.  Unlike everything else here it is
#: the same number for every asset, which is what makes bp the right unit.
REBATE_BP = -config.MAKER_FEE * 1e4

CANDIDATE_STEPS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def detect_step(values: np.ndarray) -> float:
    """The largest of `CANDIDATE_STEPS` that divides every value."""
    finite = values[np.isfinite(values) & (values > 0)]
    sample = finite[:: max(1, len(finite) // 200_000)]
    best = CANDIDATE_STEPS[0]
    for step in CANDIDATE_STEPS:
        if np.all(np.abs(sample / step - np.round(sample / step)) < 1e-6):
            best = step
    return best


def symbols() -> list[str]:
    """Every symbol with a day file on disk, smallest file first."""
    found = []
    for path in config.DATA.glob(f"*_{DAY}.npz"):
        name = path.name.split("_")[0]
        if name == "btcusdt":
            continue
        found.append((path.stat().st_size, name))
    return [name for _, name in sorted(found)]


#: Every asset quotes the same notional, so the PnL curves are comparable:
#: $4,400 an order (one ETH on these days) against a $88,000 position cap and
#: a $100,000 book, which is the sizing the rest of C1 uses.
ORDER_NOTIONAL = 4_400.0
MAX_NOTIONAL = 88_000.0
BOOK_SIZE = 100_000.0
GRID_NUM = 20


def register(symbol: str) -> tuple[float, float, float]:
    """Measure this symbol's tick size, lot size and price level from the data.

    The price comes from trade prints rather than from the book, because the
    book's outer levels run to absurd prices that would put the ROI bounds --
    and hence the depth vector's memory -- somewhere silly.
    """
    day = np.load(config.DATA / f"{symbol}_{DAY}.npz")["data"]
    tick = detect_step(day["px"])
    lot = detect_step(day["qty"])
    trades = day["px"][(day["ev"] & 0xFF) == 2]
    price = float(np.median(trades)) if len(trades) else float(np.median(day["px"]))
    del day
    config.register_symbol(symbol, tick, lot, roi_ub=4.0 * price)
    config.use_symbol(symbol)
    return tick, lot, price


def measure_symbol(symbol: str, hours: float) -> dict:
    """Everything C1 needs to know about one asset, from one day."""
    tick, lot, trade_price = register(symbol)
    config.ensure_latency(DAY)

    # --- 100 ms pass: spread, volatility, arrival depths -------------------
    hbt = config.build_backtest(DAY)
    arrival_depth, mid_price_chg, mid_price, spread = glft.measure_market(hbt, glft.INTERVAL_NS)
    hbt.close()

    price = float(np.nanmedian(mid_price) * tick) if np.isfinite(mid_price).any() else trade_price
    sigma = float(glft.realised_volatility(mid_price_chg, glft.INTERVAL_NS))
    finite_spread = spread[np.isfinite(spread)]
    depths = arrival_depth[np.isfinite(arrival_depth)]

    window = glft.window_steps(glft.INTERVAL_NS)
    tmp = np.zeros(glft.MAX_BUCKET, np.float64)
    lambda_ = glft.measure_trading_intensity(arrival_depth[:window], tmp).copy() / glft.WINDOW_SECONDS
    ticks = np.arange(len(lambda_)) + 0.5
    fit = ticks[: glft.SHALLOW], lambda_[: glft.SHALLOW]
    y = np.log(fit[1])
    slope, log_a = glft.linear_regression(fit[0], y)
    A, k = float(np.exp(log_a)), float(-slope)
    residual = y - (log_a + slope * fit[0])
    r2 = float(1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2))
    c1, c2 = glft.compute_coeff(0.05, 0.05, 1.0, A, k)
    half_spread = float(c1 + 0.5 * c2 * sigma)

    # --- 2 ms pass: adverse selection, no orders placed --------------------
    interval_ns = 2_000_000
    n_steps = int(hours * 3600 * 1e9 / interval_ns)
    hbt = config.build_backtest(DAY)
    mids = np.full(n_steps + 1, np.nan)
    flow = np.zeros(n_steps + 1)
    qty = np.zeros(n_steps + 1)
    n = glft.measure_flow_continuation(hbt, interval_ns, n_steps, mids, flow, qty)
    hbt.close()

    felt = {}
    for horizon in HORIZONS:
        ahead = int(horizon * 1e9 / interval_ns)
        future = np.full(n, np.nan)
        future[: n - ahead] = mids[ahead:n] - mids[: n - ahead]
        both = np.concatenate([future[flow[:n] < 0], -future[flow[:n] > 0]])
        felt[horizon] = float(np.nanmean(both))

    tick_bp = tick / price * 1e4
    return {
        "symbol": symbol,
        "tick_size": tick,
        "lot_size": lot,
        "price": price,
        "tick_bp": tick_bp,
        "sigma_ticks": sigma,
        "sigma_bp": sigma * tick_bp,
        "median_spread_tick": float(np.median(finite_spread)),
        "pct_spread_1_tick": float((finite_spread == 1).mean() * 100),
        "steps_with_trade": int(len(depths)),
        "A": A,
        "k": k,
        "r2": r2,
        "c1": float(c1),
        "c2": float(c2),
        "glft_half_spread_tick": half_spread,
        "rebate_ticks": REBATE_BP / tick_bp,
        "income_bp": 0.5 * float(np.median(finite_spread)) * tick_bp + REBATE_BP,
        "adverse_1s_ticks": felt[1.0],
        "adverse_1s_bp": felt[1.0] * tick_bp,
        "felt_ticks": felt,
        "felt_bp": {h: v * tick_bp for h, v in felt.items()},
        "pct_steps_aggressive": float((qty[:n] > 0).mean() * 100),
        "lambda": lambda_.tolist(),
    }


def run_pnl(symbol: str, price: float) -> dict:
    """Replay the day with the two quoters and return their equity curves.

    Both are sized by notional rather than by unit count, so a BTCUSDC order
    and an ENAUSDC order are the same amount of money and the curves can share
    an axis.
    """
    from hftbacktest import Recorder  # noqa: PLC0415

    config.use_symbol(symbol)
    qty = max(round(ORDER_NOTIONAL / price / config.LOT_SIZE), 1) * config.LOT_SIZE
    cap = max(round(MAX_NOTIONAL / price / config.LOT_SIZE), 1) * config.LOT_SIZE
    interval_ns = glft.INTERVAL_NS

    out = {"order_qty": qty, "max_position": cap}
    for name in ("glft-grid", "touch"):
        hbt = config.build_backtest(DAY)
        recorder = Recorder(1, 1_000_000)
        if name == "glft-grid":
            glft.gridtrading_glft_mm(
                hbt, recorder.recorder, 0.05, 1.0, 1.0, 0.05, qty, cap, GRID_NUM, interval_ns, 1
            )
        else:
            glft.touch_quoter(
                hbt, recorder.recorder, qty, cap, interval_ns, 1,
                np.zeros((1, 3)), np.zeros(1), 0,
            )
        hbt.close()
        record = recorder.get(0)
        equity = record["balance"] + record["position"] * record["price"] - record["fee"]
        keep = np.isfinite(equity)
        equity = equity[keep]
        stamps = record["timestamp"][keep]
        stride = max(1, len(equity) // 1440)
        out[name] = {
            "hours": ((stamps[::stride] - stamps[0]) / 3.6e12).tolist(),
            "equity_pct": (equity[::stride] / BOOK_SIZE * 100).tolist(),
            "final_pct": float(equity[-1] / BOOK_SIZE * 100),
            "fees_pct": float(-record["fee"][keep][-1] / BOOK_SIZE * 100),
            "trades": int(record["num_trades"][keep][-1]),
            "trading_value": float(record["trading_value"][keep][-1]),
        }
        del record
    return out


def plot_pnl(rows: list[dict], run: Run) -> None:
    """Equity curves, and where the day's money actually came from."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415

    order = sorted(rows, key=lambda r: r["sigma_bp"])
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[: len(order)]

    fig = plt.figure(figsize=(13, 9.5))
    grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.24)

    for col, strategy in enumerate(("glft-grid", "touch")):
        ax = fig.add_subplot(grid[0, col])
        for row, colour in zip(order, colours, strict=True):
            pnl = row["pnl"][strategy]
            ax.plot(pnl["hours"], pnl["equity_pct"], color=colour, lw=1.2, label=row["symbol"])
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xlabel("hours into 2025-08-26")
        ax.set_ylabel("equity, % of a 100k book")
        ax.set_title(f"({'ab'[col]}) {strategy}: equity through the day")
        if col == 0:
            ax.legend(fontsize=6.5, ncol=2)

    # (c) where the money went, per asset: gross trading against rebates.
    ax = fig.add_subplot(grid[1, 0])
    x = np.arange(len(order))
    gross = [r["pnl"]["glft-grid"]["final_pct"] - r["pnl"]["glft-grid"]["fees_pct"] for r in order]
    fees = [r["pnl"]["glft-grid"]["fees_pct"] for r in order]
    net = [r["pnl"]["glft-grid"]["final_pct"] for r in order]
    ax.bar(x - 0.22, gross, 0.44, label="gross trading", color="#c44e52")
    ax.bar(x + 0.22, fees, 0.44, label="rebates earned", color="#55a868")
    ax.plot(x, net, "ko", ms=5, label="net")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["symbol"] for r in order], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of book, one day")
    ax.set_title("(c) glft-grid: the rebate is the whole of the income")
    ax.legend(fontsize=7)

    # (d) the backtest's realised cost against the cost measured without it.
    # They land in the same range and do NOT rank the assets the same way
    # (r is about 0.1), which is the point: the market-wide 1 s number says
    # how expensive passive fills are in general, not which asset a particular
    # quoter loses least on.  That depends on where it quotes and how long it
    # holds, and only the backtest knows.
    ax = fig.add_subplot(grid[1, 1])
    xs, ys = [], []
    for row, colour in zip(order, colours, strict=True):
        pnl = row["pnl"]["glft-grid"]
        if pnl["trading_value"] <= 0:
            continue
        realised = -(pnl["final_pct"] - pnl["fees_pct"]) / 100 * BOOK_SIZE / pnl["trading_value"] * 1e4
        xs.append(-row["adverse_1s_bp"])
        ys.append(realised)
        ax.scatter(xs[-1], realised, color=colour, s=45, zorder=3)
        ax.annotate(row["symbol"], (xs[-1], realised), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    xa, ya = np.array(xs), np.array(ys)
    lim = max(xa.max(), ya.max()) * 1.15
    slope, intercept = np.polyfit(xa, ya, 1)
    line = np.linspace(0, lim, 20)
    corr = float(np.corrcoef(xa, ya)[0, 1])
    ax.plot(line, slope * line + intercept, "k--", lw=1, label=f"fit, r = {corr:.2f} (no ranking)")
    ax.plot(line, line, color="0.7", ls=":", lw=1, label="equal")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("adverse selection at 1 s, measured with no orders (bp)")
    ax.set_ylabel("gross loss per unit of trade value, from the backtest (bp)")
    ax.set_title("(d) same range, different ranking", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Passive market making PnL across Binance USDT/USDC-margined perpetuals, {DAY}", y=0.985)
    path = run.path("cross-asset-pnl.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot(rows: list[dict], run: Run) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    order = sorted(rows, key=lambda r: r["sigma_bp"])
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[: len(order)]

    fig = plt.figure(figsize=(13, 9.5))
    grid = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26)

    # (a) intensity fits, one line per asset, on a shared axis.
    ax = fig.add_subplot(grid[0, 0])
    for row, colour in zip(order, colours, strict=True):
        lam = np.array(row["lambda"])[: glft.SHALLOW]
        x = np.arange(len(lam)) + 0.5
        ax.plot(x, lam, color=colour, lw=1.2, label=f"{row['symbol']} (k={row['k']:.3f})")
        ax.plot(x, row["A"] * np.exp(-row["k"] * x), color=colour, ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta$ (half-ticks from the mid)")
    ax.set_ylabel(r"$\lambda(\delta)$ per second")
    ax.set_title("(a) order-arrival intensity, measured (solid) and fitted (dotted)")
    ax.legend(fontsize=6.5, ncol=2)

    # (b) the economics: income against cost, both in bp, break-even on y = x.
    ax = fig.add_subplot(grid[0, 1])
    for row, colour in zip(order, colours, strict=True):
        ax.scatter(-row["adverse_1s_bp"], row["income_bp"], color=colour, s=45, zorder=3)
        ax.annotate(row["symbol"], (-row["adverse_1s_bp"], row["income_bp"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    lim = max(1.0, max(-r["adverse_1s_bp"] for r in order) * 1.15)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="break even")
    ax.axhline(REBATE_BP, color="0.6", lw=0.9, ls=":", label=f"rebate alone ({REBATE_BP:g} bp)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("adverse selection at 1 s (bp per fill)")
    ax.set_ylabel("income per fill: half spread + rebate (bp)")
    ax.set_title("(b) income per fill against adverse selection at 1 s")
    ax.legend(fontsize=7, loc="upper left")

    # (c) adverse selection by horizon, in bp so assets are comparable.
    ax = fig.add_subplot(grid[1, 0])
    for row, colour in zip(order, colours, strict=True):
        ax.plot(HORIZONS, [row["felt_bp"][h] for h in HORIZONS], marker="o", ms=3,
                color=colour, lw=1.2, label=row["symbol"])
    ax.axhline(-REBATE_BP, color="0.4", ls="--", lw=1)
    ax.annotate("the rebate covers this much", (0.12, -REBATE_BP), fontsize=7,
                va="bottom", color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel("seconds after the fill (log)")
    ax.set_ylabel("mid move against the position (bp)")
    ax.set_title("(c) adverse selection, measured with no orders placed")
    ax.legend(fontsize=6.5, ncol=2)

    # (d) cost against volatility -- the relationship the whole phase turns on.
    ax = fig.add_subplot(grid[1, 1])
    x = np.array([r["sigma_bp"] for r in order])
    y = np.array([-r["adverse_1s_bp"] for r in order])
    for row, colour in zip(order, colours, strict=True):
        ax.scatter(row["sigma_bp"], -row["adverse_1s_bp"], color=colour, s=45, zorder=3)
        ax.annotate(row["symbol"], (row["sigma_bp"], -row["adverse_1s_bp"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(0, x.max() * 1.1, 20)
    corr = float(np.corrcoef(x, y)[0, 1])
    ax.plot(xs, slope * xs + intercept, "k--", lw=1,
            label=f"{slope:.2f}·σ {intercept:+.2f},  r={corr:.2f}")
    ax.axhline(REBATE_BP + 0.5 * np.median([r["median_spread_tick"] * r["tick_bp"] for r in order]),
               color="0.6", ls=":", lw=1, label="typical income")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(r"$\sigma$ (bp per $\sqrt{s}$)")
    ax.set_ylabel("adverse selection at 1 s (bp per fill)")
    ax.set_title("(d) the cost is volatility, in every asset")
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Passive market making across Binance USDT/USDC-margined perpetuals, {DAY}", y=0.985)
    path = run.path("cross-asset.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument(
        "--replot",
        metavar="RUNDIR",
        help="redraw the figures from a finished run instead of re-measuring",
    )
    args = parser.parse_args()

    if args.replot:
        import json  # noqa: PLC0415

        directory = Path(args.replot)
        rows = json.loads((directory / "summary.json").read_text())["rows"]
        for row in rows:  # JSON turns the horizon keys into strings
            row["felt_bp"] = {float(h): v for h, v in row["felt_bp"].items()}
        run = Run(phase="c1", name="cross-asset-replot", config={"from": str(directory)},
                  root=directory.parent)
        run.dir = directory
        if "lambda" in rows[0]:
            plot(rows, run)
        plot_pnl(rows, run)
        return

    names = symbols()
    print(f"{len(names)} symbols: {', '.join(names)}\n")
    run = Run(phase="c1", name="cross-asset", config={"day": DAY, "hours": args.hours,
                                                      "symbols": names})

    header = (f"{'symbol':>9} {'tick':>9} {'price':>10} {'tick bp':>8} {'sigma bp':>9} "
              f"{'1-tick':>7} {'k':>7} {'R2':>6} {'half':>7} {'income':>7} {'adverse':>8} {'ratio':>6}")
    print(header)
    rows = []
    for i, symbol in enumerate(names):
        row = measure_symbol(symbol, args.hours)
        row["pnl"] = run_pnl(symbol, row["price"])
        rows.append(row)
        run.log(step=i, **{key: value for key, value in row.items() if key not in ("lambda", "pnl")})
        print(
            f"{row['symbol']:>9} {row['tick_size']:9g} {row['price']:10.4f} {row['tick_bp']:8.4f} "
            f"{row['sigma_bp']:9.3f} {row['pct_spread_1_tick']:6.1f}% {row['k']:7.4f} "
            f"{row['r2']:6.3f} {row['glft_half_spread_tick']:7.1f} {row['income_bp']:7.3f} "
            f"{-row['adverse_1s_bp']:8.3f} {-row['adverse_1s_bp'] / row['income_bp']:6.2f}",
            flush=True,
        )

    print("\n'income' is half the touch spread plus the rebate, in bp; 'adverse' is the mid")
    print("move against the position 1 s after a fill; 'ratio' is cost over income, and")
    print("anything above 1.0 is a market a passive maker cannot make without an edge.")

    print(f"\n{'symbol':>9} {'glft-grid %':>12} {'  of which rebate':>18} {'trades':>8} "
          f"{'touch %':>9}")
    for row in sorted(rows, key=lambda r: r["sigma_bp"]):
        g, t = row["pnl"]["glft-grid"], row["pnl"]["touch"]
        print(f"{row['symbol']:>9} {g['final_pct']:12.3f} {g['fees_pct']:18.3f} "
              f"{g['trades']:8d} {t['final_pct']:9.3f}")

    plot(rows, run)
    plot_pnl(rows, run)
    run.finish({"rows": rows})
    print(f"run dir: {run.dir}")


if __name__ == "__main__":
    main()
