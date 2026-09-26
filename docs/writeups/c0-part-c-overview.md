# Part C in one page — market making on hftbacktest: what made money, what didn't

The question behind all of Part C: can reinforcement learning improve a
published market-making strategy, measured on days it never saw? This page is
the summary; each phase has its own writeup, linked in §4.

> **At a glance**
>
> - **Best held-out result:** OBI on XRPUSDC with a markout-driven spread,
>   **+4.91%** over eleven held-out sessions, **+1.38% without the crash day**,
>   positive on **9 of 11** sessions at ~500 fills a day.
> - **Everything on ETHUSDC still loses held out.** The best, the same rule on
>   OBI, is at −0.62%, against −6.40% for the widened baseline and −7.14% for
>   the published parameters.
> - **The two things that helped were both hand-written rules:** a wider,
>   per-asset spread floor (C6) and widening after toxic fills (C7).
> - **Learning did not beat them.** TD3 on C6's widened strategies (two recipes,
>   five seeds), GRPO- and PPO-style parameter search, extra order-book and
>   BTCUSDT inputs, fill targets and a reward that discounts inventory PnL all
>   lose to the best fixed rule held out.

---

## 1. Setup, in five lines

- **Data:** Binance USDⓈ-M perpetual L2 books and trades, 2025-08-26 to
  2025-09-25, replayed tick by tick in hftbacktest with queue position and
  synthesised order latency.
- **Split:** everything is chosen on sessions before 2025-09-15 and reported on
  the eleven sessions from 09-15 to 09-25. Nothing is chosen on held-out data.
- **Deep assets:** ETHUSDC and XRPUSDC, 11 held-out sessions each. Seven more
  assets with 3 held-out sessions each.
- **Strategies:** the five market makers hftbacktest publishes — GLFT, a
  volatility grid (`hfgrid`), order-book imbalance (`obi`), and two pair
  strategies (`apt`, `basis`) on XRP.
- **Score:** cumulative held-out return in % of a 100,000 book; fills and
  per-session consistency beside it, so "stopped trading" never passes for
  "made money".

## 2. The track record

![track-record](../assets/partc/track-record.png)

*Figure — cumulative return over the eleven held-out sessions: the published
parameters (dotted), C6's widened spread floor (blue), and C7's markout rule on
top of it (orange). Session 8 is 2025-09-22, when the market crashed.*

Two things are visible at once. The widened floor helps on every panel but
does not stop the crash-day loss. The markout rule does: on 09-22 the widened
strategies lose 3–4% while the markout versions lose about 1% or gain —
plausibly because they had already widened as their fills turned toxic. Part of XRP's gain that
day is holding the right inventory through the move, which is why the result
is also quoted with 09-22 removed.

## 3. Every method, held out

| asset | strategy | published | C6 widened | C6 RL v2 | C7 markout | C4 grid (20 d) | C4 GRPO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETH | glft | −15.13 | −7.07 | −6.30 | −5.36 | −3.01 | **−1.89** |
| ETH | hfgrid | −1.98 | −2.11 | −2.20 | −1.99 | **−1.86** | – |
| ETH | obi | −7.14 | −6.40 | −8.00 | **−0.62** | **−0.62** | −2.91 |
| XRP | glft | −13.17 | −1.89 | −0.86 | **−0.55** | −1.22 | – |
| XRP | hfgrid | −1.58 | −1.15 | −1.48 | **+1.96** | +0.59 † | – |
| XRP | obi | −6.72 | −2.91 | −2.15 | **+4.91** | +3.64 | – |
| XRP | apt | −3.39 | +2.00 | −1.66 | – | +7.26 † | – |
| XRP | basis | −4.59 | +0.36 | −2.35 | – | +1.69 † | – |

*Held-out cumulative return, % of book, eleven sessions. RL and GRPO are
medians over seeds. † trades a handful of fills or none on a median session;
the number is one crash session, not market making.*

## 4. What each phase contributed

| phase | question | answer, held out |
| --- | --- | --- |
| [C1](c1-glft-baseline.md) | Does calibrated GLFT make money here? | No: it quotes inside the cost of adverse selection and loses 0.8 bp per fill. |
| [C2](c2-env.md) | Can the problem be posed for RL without a degenerate answer? | Yes, once the "quote so far out nothing fills" corner is ruled out. |
| [C3](c3-continuous-control.md) | Do DDPG/TD3/SAC beat the fixed formula? | They lose less than it, never make money; seed spread is large. |
| [C6](c6-strategy-zoo.md) | Is that just GLFT being a bad baseline? | No. On five strategies and nine assets, a per-asset wider spread is the largest single gain (+2.25 points median); TD3 on top of it does not beat it. |
| [C7](c7-signals.md) | What information is the policy missing? | Not the deep book or BTCUSDT. The toxicity of its own recent fills: a markout rule beats C6 in 6 of 6 pairs and 13 of 18 untuned. |
| [C4](c4-ppo-grpo.md) | Can search (grid, GRPO, PPO) tune the fixed rules further? | The grid refines them on ETH; learned search, fill targets and an inventory-discounted reward do not. |

## 5. Lessons that held up

1. **Measure the cost of a fill before tuning anything.** Every positive
   result in Part C comes from quoting wider when fills are expensive.
2. **Fixed rules first.** Each learned method was beaten by a hand-written rule
   that encoded the right quantity; RL is worth trying on top of the best rule,
   not instead of it.
3. **Seeds and sessions, not single runs.** Single-seed RL rows were
   systematically kinder than five-seed rows, and four training sessions
   chose XRP parameters that twenty did not.
4. **Report the crash day separately.** One session (2025-09-22) can make a
   strategy that barely trades look like the best in the table.
5. **Fills are not free.** Both forcing volume and rewarding it cost money on
   this venue, at every level tried.

## 6. What next

- A **circuit breaker** (pause quoting on volatility or one-sided-flow spikes)
  and a **daily loss limit**, as grid dimensions on top of the markout rule.
- The **breadth assets** on eleven held-out sessions, not three, starting with
  OBI, where the markout rule already generalised to six of seven.
- **RL on top of the markout rule** — its average in the state, the rule as the
  residual baseline — as the one learning experiment still worth its compute.
