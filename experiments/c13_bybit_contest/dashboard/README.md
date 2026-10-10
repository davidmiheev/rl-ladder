# C13 dashboard ("OBI Before and Now")

Source of the private claude.ai dashboard that tracks C10-C13 per session.

| file | what |
| --- | --- |
| `index.html` | the page (the dashboard's `files/index.html`) |
| `build.py` | builds the `contest` dataset (one row per contract, entry and session: return, fills, turnover) from the result cache in `runs/` |
| `sessions.json` | the `sessions` dataset (C10/C12 forward-test sessions), as uploaded |

`contest.json` is not committed: it is about 2 MB and `build.py` regenerates it
(`python experiments/c13_bybit_contest/dashboard/build.py contest.json` from
the repo root). The dashboard reads both datasets as attached JSON files.
