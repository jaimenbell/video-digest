# Premise check: pc-2026-09-11-grid-drawdown-writer

**Task id:** 4ad981f6c279476d8f7933800209202f
**Repo this session was caged to:** video-digest (`C:\Users\jaime\projects\video-digest`)
**Branch:** lane/g3-audit

## Claim under test

Task title: *"grid-trading-bot state.json reports drawdown_pct=0.019 while fleet_health
computes 1.3176 from pnl_total -26.35 on starting_capital 2000 (fleet-health-latest.json
state_integrity, 2026-09-11); the state writer under-reports drawdown."*

Source spec (`C:/Users/jaime/.claude/staged/lane-report-w11-fleet-health-errors-2026-09-11.md`,
"Next-wave item" section, lines 180-186) logs this only as a flagged discrepancy for a future
lane, not investigated there — this session investigates it.

## Scope note

`grid-trading-bot` is not one of the three repos CLAUDE.md names as categorically off limits
to this cage (`overnight-waves`, `alphahive`, `shared`), so its source and `state.json` were
readable. `shared` (where `fleet_health.py` / `health.py` live, per the lane report's own grep
of `errors_24h` consumers) **is** categorically off limits — per the task's ambiguity rule, no
attempt was made to read `fleet_health.py` or the `fleet-health-latest.json` artifact even
read-only. Only `Read`/`Grep`/`Glob` were used against `grid-trading-bot`; nothing was written
there.

## Result: PREMISE REFUTED — not a state-writer bug, a unit/definition mismatch

`grid-trading-bot\state_writer.py:190-234` documents and implements `drawdown_pct` as
**peak-to-current drawdown, expressed as a fraction** (docstring, line 204: `"peak-to-current
drawdown as a fraction (e.g. 0.05 = 5%)"`). The formula (lines 226-234):

```python
current_equity = _STARTING_CAPITAL + total_pnl
if current_equity > _peak_equity:
    _peak_equity = current_equity
drawdown_pct = (
    max(0.0, (_peak_equity - current_equity) / _peak_equity)
    if _peak_equity > 0
    else 0.0
)
_state["drawdown_pct"] = round(drawdown_pct, 6)
```

This is drawdown from the bot's own **peak equity** (persisted in `peak_equity_usd` so it
survives restarts), not a loss-from-starting-capital figure.

**Verified against the live `state.json`** (`grid-trading-bot\state.json`, read this session,
timestamped `2026-09-12T13:59:51Z` — one day after the claim's 2026-09-11 snapshot, so absolute
numbers have drifted but the formula is verifiable on current data):

```
pnl_total        = -25.3409
unrealized_pnl   = -0.3776
total_pnl        = -25.7185   (= pnl_total + unrealized_pnl)
starting_capital = 2000.0
peak_equity_usd  = 2011.2993
drawdown_pct     = 0.018405
```

Recomputing by hand: `current_equity = 2000.0 + (-25.7185) = 1974.2815`;
`(2011.2993 - 1974.2815) / 2011.2993 = 0.0184049...` → rounds to `0.018405`, an **exact match**
to the persisted field. The writer computes exactly what it documents; there is no
under-reporting bug in `state_writer.py`.

The claim's framing ("state writer under-reports drawdown," 0.019 vs. computed 1.3176)
appears to come from comparing two **different metrics**, not a writer defect:
- `state.json`'s `drawdown_pct` is peak-equity-based, and is a **fraction** (0.019 fractional
  = 1.9%, not 0.019%).
- fleet_health's "computed 1.3176" is consistent with a starting-capital-basis calculation
  (`pnl_total / starting_capital`): `-26.35 / 2000 = -0.013175` → `1.3175%`, matching the
  claimed 1.3176 to four significant figures.

On today's numbers, the peak-based metric (1.8405%) is **larger**, not smaller, than the
equivalent starting-capital-basis metric (`|total_pnl| / starting_capital = 25.7185/2000 =
1.286%`) — because `peak_equity_usd` (2011.30) is above `starting_capital` (2000), so drawdown
from peak is necessarily ≥ loss from start whenever the bot was ever ahead. This directly
contradicts the "under-reports" direction of the premise: state.json's own number is not
smaller than the alternative basis, it is comparable or larger once the fraction-vs-percent
scale is read correctly.

## What this check could not see

- `fleet_health.py` / `health.py` (in `shared`, categorically off limits to this cage) were not
  read, so the exact formula fleet_health used to arrive at "1.3176" is inferred from the
  claim's own numbers (`pnl_total -26.35`, `starting_capital 2000`), not confirmed from
  fleet_health's source.
- `fleet-health-latest.json` (the state_integrity artifact named in the claim) was not located
  or read — unknown filesystem location, and if it lives under `shared` or `overnight-waves` it
  would be off limits regardless.
- No shell access in this cage, so no test run (`pytest grid-trading-bot/tests`) was performed
  to confirm `drawdown_pct` semantics are covered by regression tests; `tests/test_mcp_tools.py`
  and `tests/test_pnl_semantics.py` reference `drawdown_pct` as a fixture field (confirmed via
  Grep) but were not executed.
- The exact `state.json` snapshot from 2026-09-11 (the one the claim quotes: `pnl_total -26.35`,
  `drawdown_pct=0.019`) is gone — `state.json` is live-overwritten on every bot cycle
  (`cycle_count=9233` as of this read) — so the claim's exact figures were not independently
  reproduced, only the current live figures, which follow the same documented formula exactly.

## Verdict

**REFUTED** (as a state-writer bug). `grid-trading-bot\state_writer.py:204,226-234` computes
`drawdown_pct` correctly per its own documented definition (peak-to-current, as a fraction);
verified by exact recomputation against the live `state.json`. The discrepancy the task
describes is best explained as a **unit/definition mismatch** between two legitimate but
different drawdown metrics (peak-equity basis vs. starting-capital basis, plus a
fraction-vs-percent scale reading), not an under-reporting defect in the writer. This is
consistent with the same "unit/definition mismatch" characterization already recorded
elsewhere for this fleet-health state_integrity gap (see handoff note "N10 state_integrity gap
= PARKED-skip + unit/definition mismatch," `~/.claude/handoffs/handoff.md` v145 — not
independently re-derived from that source in this session, only cross-referenced).
Recommend the operator close this as "premise refuted, definition mismatch" rather than routing
a fix lane at `grid-trading-bot`'s state writer. If the operator wants `fleet_health`'s
consumer-side interpretation reconciled (e.g. read `peak_equity_usd`/`starting_capital`
directly and compute both bases explicitly, or label the two metrics distinctly), that requires
a session with access to `shared`, which this cage does not have.
