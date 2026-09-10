# C1 — Calibrated GLFT: task, goal, and how it was solved

Part C's benchmark: the closed-form market maker every RL variant has to beat.
Reproducing it on 2025 ETHUSDC turns out to be a negative result with a precise
cause, and the cause is what decides how C2 builds its environment. Lab
notebook: [`experiments/c1_glft_baseline/README.md`](../../experiments/c1_glft_baseline/README.md).

---

## 1. The task

[`ladder.md`](../ladder.md#c1) states three items:

1. Calibrate $A$, $k$ (order-arrival intensity) and $\sigma$ (volatility) from
   market data; derive $c_1$, $c_2$ and reproduce the GLFT half-spread/skew
   formulas.
2. Run the calibrated GLFT grid market maker in hftbacktest on the sample data;
   reproduce the reported Sharpe/Sortino/MaxPositionValue stats.
3. Reproduce the simplified non-parametric HF-grid variant (percentile-based
   intensity, micro-price) and compare against the parametric GLFT baseline.

## 2. The goal: a benchmark that is not a straw man

Every later Part C phase makes a claim of the form "the learned policy beats the
calibrated formula". That claim is only worth something if the formula was
calibrated properly and run under the same fill assumptions. So C1's real job is
not to produce a number — it is to produce a number that can be *trusted*, and
to expose every assumption underneath it, because those assumptions are shared
by everything built on top.

The Guéant–Lehalle–Fernandez-Tapia model is the natural choice. It is an
asymptotic (no terminal time) refinement of Avellaneda–Stoikov, which makes it
appropriate for a perpetual future, and it gives quote depths in closed form
from three measurable quantities rather than from a solved HJB equation.

## 3. The model

### Order-arrival intensity

The model assumes market orders arrive at a quote posted $\delta$ ticks from the
fair price at rate

$$\lambda(\delta) = A e^{-k\delta}.$$

Both parameters are estimated by regression: taking logs makes it a straight
line,

$$\log \lambda(\delta) = \log A - k\delta,$$

so one least-squares fit over the empirical arrival-depth histogram gives
$\hat{A} = e^{\widehat{\log A}}$ and $\hat{k} = -\widehat{\text{slope}}$.

$\lambda$ itself is measured by recording, in each 100 ms step, how deep the
deepest market order reached relative to the previous step's mid-price, and
counting the steps in which a trade reached at least each depth. Buy trades are
measured above the mid, sell trades below.

### The optimal quotes

Equations (4.6) and (4.7) of [Guéant, Lehalle & Fernandez-Tapia](https://huggingface.co/papers/1605.01862)
give the asymptotically optimal bid and ask depths for a maker holding
inventory $q$, quoting size $\Delta$, with risk aversion $\gamma$ and a
liquidity-penalty parameter $\xi$:

$$\delta^{b*}_{\text{approx}}(q) = \frac{1}{\xi\Delta}\log\!\left(1 + \frac{\xi\Delta}{k}\right)
+ \frac{2q + \Delta}{2}\sqrt{\frac{\gamma\sigma^2}{2A\Delta k}\left(1 + \frac{\xi\Delta}{k}\right)^{\frac{k}{\xi\Delta}+1}}$$

$$\delta^{a*}_{\text{approx}}(q) = \frac{1}{\xi\Delta}\log\!\left(1 + \frac{\xi\Delta}{k}\right)
- \frac{2q - \Delta}{2}\sqrt{\frac{\gamma\sigma^2}{2A\Delta k}\left(1 + \frac{\xi\Delta}{k}\right)^{\frac{k}{\xi\Delta}+1}}$$

Pulling $\sigma$ out of the square root defines two coefficients that depend
only on the calibration, not on the state:

$$c_1 = \frac{1}{\xi\Delta}\log\!\left(1 + \frac{\xi\Delta}{k}\right),
\qquad
c_2 = \sqrt{\frac{\gamma}{2A\Delta k}\left(1 + \frac{\xi\Delta}{k}\right)^{\frac{k}{\xi\Delta}+1}}$$

and the quotes collapse to a half spread plus an inventory skew:

$$\delta^{b*}(q) = \underbrace{c_1 + \tfrac{\Delta}{2}\sigma c_2}_{\text{half spread}} + \underbrace{\sigma c_2}_{\text{skew}}\, q,
\qquad
\delta^{a*}(q) = \left(c_1 + \tfrac{\Delta}{2}\sigma c_2\right) - \sigma c_2\, q.$$

This rewrite is an algebraic identity, which is worth stating because it is
checkable: `run_calibrate.py` verifies it to $5 \times 10^{-13}$ over two
thousand random parameter draws before reporting anything (§6).

### The spread floor, and why it is the whole story

The model has one free parameter that no data pins down. $\gamma$ is "arbitrarily
chosen" in the paper; the notebook sets $\gamma = \xi = 0.05$ and $\Delta = 1$.
So the first question to ask of a calibration is: **how tight can these quotes
possibly be?**

Write $u = \xi\Delta$. The half spread is

$$H(u) = \frac{\log(1 + u/k)}{u} + \frac{\Delta\sigma}{2}\sqrt{\frac{\gamma}{2A\Delta k}\left(1+\frac{u}{k}\right)^{k/u+1}}.$$

The second term is non-negative, so $H(u) \ge c_1(u) = \log(1+u/k)/u$. And
$c_1$ is strictly decreasing in $u$ with

$$\lim_{u \to 0^+} c_1(u) = \frac{1}{k}, \qquad \lim_{u \to \infty} c_1(u) = 0,$$

the first limit by $\log(1+x) = x - x^2/2 + O(x^3)$. So shrinking $\gamma$ lowers
$c_1$'s contribution towards its supremum $1/k$ rather than towards zero — and
meanwhile the second term behaves like $C\sqrt{\gamma}$ as $\gamma \to 0$
(with $\xi = \gamma$), since $c_2 \propto \sqrt{\gamma}$ and the power factor
tends to $e$. Expanding for small $\gamma$,

$$H(\gamma) = \frac{1}{k} + C\sqrt{\gamma} - \frac{\Delta\gamma}{2k^2} + O(\gamma^{3/2}),$$

whose $\sqrt{\gamma}$ term dominates, so $H$ is increasing near $0$ and

$$\boxed{\ \inf_{\gamma > 0} H(\gamma) = \frac{1}{k}\ }$$

approached only in the risk-neutral limit. **No choice of risk aversion lets a
GLFT quoter post closer to the fair price than $1/k$.** Everything C1 finds
follows from putting a measured $k$ into that bound.

### Volatility, in the model's units

$\sigma$ is a mid-price volatility in ticks per $\sqrt{\text{second}}$. Sampling
the mid every $\tau$ seconds and taking the standard deviation of the changes
gives a per-step figure; under a random-walk mid the per-second one is

$$\hat\sigma = \operatorname{sd}\!\left(m_{t} - m_{t-1}\right) \cdot \tau^{-1/2}.$$

The units matter more than they look. $\sigma$ in *ticks* scales with the price
divided by the tick size, so an asset whose price triples without its tick size
changing has three times the volatility in the only units the model can see.

## 4. What was done

**Data.** Binance USDⓈ-M ETHUSDC perpetual, from the private Hugging Face
dataset `cardinalai/backtest_data`, already normalised into hftbacktest's
eight-field event format. Ten days: five training
(2025-08-26, -27, -28, 09-01, 09-08) and five held out
(2025-09-15, -18, -20, -22, -25). The split rule is chronological and is fixed
in [`config.py`](../../experiments/c1_glft_baseline/config.py) so C3–C5 cannot
re-pick it; see [`docs/data.md`](../data.md).

**Order latency** is not in the dataset, so it is synthesised from the feed
latency the data itself carries (`local_ts - exch_ts`) at hftbacktest's own
multipliers — entry $4\times$, response $3\times$ — giving a round trip of about
20 ms on these days. Our generator is checked bit-for-bit against hftbacktest's
packaged one (§6); ours exists only because theirs copies a whole 60-million-row
day through polars and does not fit in memory here.

**Five quoters**, in [`glft.py`](../../experiments/c1_glft_baseline/glft.py),
transcribed from hftbacktest's notebooks rather than reinvented:

| name | what it is |
| --- | --- |
| `touch` | no model: one lot at the best bid and the best ask. The floor. |
| `glft` | equations (4.6)/(4.7) verbatim, one quote a side, $\gamma=\xi=0.05$, $\Delta=1$ |
| `glft-adj` | the same with the notebook's `adj2 = 0.05` damping the skew |
| `glft-grid` | `glft-adj` laid out as a 20-deep grid — the configuration the published Sharpe comes from |
| `simplified` | item 3: percentile half spread, micro-price fair value, same grid |

Known quirks in the source are **reproduced, not fixed**, and flagged in
comments. The largest is that the intensity histogram indexes buckets at
`round(depth/0.5) - 1`, so its depth axis is in half-ticks while the quoter
spends the resulting half spread as if it were ticks; hftbacktest's own note
says the example is left unchanged to preserve published results, and C1 does
the same, reporting the fit both ways so the distortion is on the record rather
than folded into a fudge factor.

## 5. Results

### Item 1 — the calibration

On the calibration day, 2025-08-26, over 863,995 steps of 100 ms with 249,144 of
them containing a trade:

| quantity | value |
| --- | --- |
| touch spread | **1 tick in 98.4% of steps** (median 1, p95 1) |
| $A$, $k$ (full range) | 1.810, 0.0135 ($R^2 = 0.984$) |
| $A$, $k$ (first 70 buckets) | 2.284, 0.0136 ($R^2 = 0.981$) |
| $A$, $k$ (buckets at their true tick depth) | 2.299, 0.0272 |
| $\sigma$, first 10-min window | 74.2 ticks$/\sqrt{s}$ |
| $\sigma$, whole day | 47.1 ticks$/\sqrt{s}$ |
| $c_1$, $c_2$ | 30.84, 2.392 |
| half spread | 119.6 ticks |
| skew | 177.5 ticks per unit of inventory |
| inventory that cancels the whole half spread | 0.67 units |
| trading steps reaching the half spread | 0.81% |

Rolling recalibration every 5 s on a trailing 10-minute window, 17,160 windows
over the day (median [p5, p95]): $A$ 1.09 [0.62, 2.74], $k$ 0.0196 [0.0124,
0.0261], $\sigma$ 38.2 [25.7, 84.5], half spread 75.7 [55.2, 131.2], skew 100.7
[67.4, 197.6].

Three things in that table decide everything that follows.

**The asset has no room inside its spread.** ETHUSDC's tick is tiny in relative
terms — 0.01 on a ~4400 price, 0.023 bp — and yet the touch spread is one tick
98.4% of the time. GLFT's derivation treats quote depth as a continuous choice;
here the only choices are *at the touch* or *away from it*.

**The fitted $k$ puts the model's own floor 25 ticks out.** With $k = 0.0196$
per half-tick bucket, $1/k \approx 51$ bucket units $\approx 25.5$ ticks. By the
bound in §3 that is the tightest quote the model can produce *at any risk
aversion*, on a market whose spread is 1 tick. A numerical sweep confirms it:
$\gamma$ from 0.001 to 5 moves the half spread over 54.6 → 3345.6, with the
minimum at the risk-neutral end, exactly as the expansion predicts.

**The skew is stronger than the spread.** 0.67 units of inventory cancels the
entire half spread, so the quoter refuses to hold inventory at all — which is
the same pathology the source notebook reports (two units there) and the reason
`adj2` exists.

For comparison, the published ETHUSDT 2022-10-03 calibration is $A = 2.986$,
$k = 0.0424$, $\sigma = 10.7$, half spread 20.47, skew 9.76, with 1.86% of
trades reaching the quote. Our $k$ is a third of theirs and our $\sigma$ four
times theirs, and both differences push the quote further out.

### Item 2 — the backtest

_(filled in below from the final runs)_

### Item 3 — non-parametric intensity and the micro-price

_(filled in below from the final runs)_

### Why it loses: the diagnosis

_(filled in below)_

## 6. Checks with known answers

Calibration code is impossible to eyeball, so four things are verified against
answers known in advance before any number above is reported.
`run_calibrate.py --check-only` runs the first three in about a second.

| check | how | result |
| --- | --- | --- |
| $c_1$/$c_2$ reproduce (4.6)/(4.7) | evaluate both forms at 2000 random $(\gamma,\Delta,A,k,\sigma,q)$ | max abs error $4.5\times10^{-13}$ |
| the intensity histogram | against a plain-Python loop that is obviously correct and obviously too slow | 0 mismatches |
| the hand-rolled OLS | against `numpy.polyfit` on data with a known slope | max abs error $4.1\times10^{-13}$ |
| the order-latency generator | against hftbacktest's own packaged helper (`--verify-latency`) | identical, field by field |

The first of these is the one that earns its keep: the paper's form and the
notebook's rewrite are the same expression, so any transcription slip in either
shows up as a disagreement rather than as a plausible-looking number.

## 7. What this sets up

_(filled in below)_

## 8. Where these items come from

| item | primary source | why it was picked |
| --- | --- | --- |
| 1 — calibrate $A$, $k$, $\sigma$; derive $c_1$, $c_2$ | [Guéant, Lehalle & Fernandez-Tapia 2016](https://huggingface.co/papers/1605.01862), equations (4.6)–(4.7); the intensity calibration follows [this quant.stackexchange answer](https://quant.stackexchange.com/questions/36073/how-does-one-calibrate-lambda-in-a-avellaneda-stoikov-market-making-problem), as hftbacktest's notebook does | The closed form is what makes GLFT usable as a benchmark at all: three measured numbers in, quotes out, nothing to solve at run time. Calibrating it by hand is also the only way to discover that the fit does not hold on this asset. |
| 2 — run it and reproduce the published stats | [hftbacktest's GLFT + grid-trading notebook](https://github.com/nkaz001/hftbacktest/blob/master/examples/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.ipynb) | Reproducing someone else's published number is the cheapest available test of a data pipeline and a fill model. When it fails, the failure is informative in a way a fresh result never is. |
| 3 — the simplified non-parametric variant | [hftbacktest's "High-Frequency Grid Trading — Simplified from GLFT"](https://github.com/nkaz001/hftbacktest/blob/master/examples/High-Frequency%20Grid%20Trading%20-%20Simplified%20from%20GLFT.ipynb), whose opening paragraph proposes percentile estimates and the micro-price for exactly the case where the parametric form does not fit | It isolates which part of the baseline is doing the work: the functional form, or the microstructure it reads. On a large-tick asset the source itself predicts the parametric form will fail, so this is the pre-registered alternative rather than a post-hoc one. |

Supporting sources: the order-latency model follows hftbacktest's
[Order Latency Data](https://hftbacktest.readthedocs.io/en/latest/tutorials/Order%20Latency%20Data.html)
tutorial; the fill and queue-position semantics that the diagnosis in §5 turns
on are documented in
[Order Fill](https://hftbacktest.readthedocs.io/en/latest/order_fill.html);
the framing of the half spread and skew matches
[Stochastic Control Theory and High Frequency Trading](https://ieor.columbia.edu/files/seasdepts/industrial-engineering-operations-research/pdf-files/Borden_D_FESeminar_Sp10.pdf)
and BitMEX's
[How to Market Make Bitcoin Derivatives, Lesson 2](https://blog.bitmex.com/how-to-market-make-bitcoin-derivatives-lesson-2/),
both cited by the notebook.

The spread-floor bound in §3, the edge-versus-exposure accounting in §5 and the
forward links in §7 are this repo's, not claims from the sources.
