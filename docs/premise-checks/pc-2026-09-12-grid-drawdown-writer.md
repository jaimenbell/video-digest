# Premise check: pc-2026-09-12-grid-drawdown-writer (v2)

**Task id:** d136e57339d54dbcaa20a21fce672a17
**Repo this session was caged to:** video-digest (`C:\Users\jaime\projects\video-digest`)
**Branch:** lane/g3-audit

## Claim under test

Task title: *"premise-check v2: pc-2026-09-12-grid-drawdown-writer - grid-trading-bot
state.json reports drawdown_pct=0.019 while fleet_health computes 1.3176 from pnl_total
-26.35 on starting_capital 2000 (fleet-health-latest.json state_integrity, 2026-09-11); the
state writer under-reports drawdown."*

This is the same claim, verbatim, as `pc-2026-09-11-grid-drawdown-writer.md` (task
`4ad981f6c279476d8f7933800209202f`), re-minted as a fresh task-board item under a new task
id. That prior check already concluded REFUTED (unit/definition mismatch, not a writer bug).
This session independently re-read the current source and live state fresh rather than
trusting the prior file's conclusion — and found the underlying code has materially changed
since that prior check.

## Scope note

`grid-trading-bot` is not one of the three repos CLAUDE.md names as categorically off limits
to this cage (`overnight-waves`, `alphahive`, `shared`), and this session's brief explicitly
grants read access to it by name. `shared` (where `fleet_health.py`/`health.py`/`criteria.py`
live, per the prior check's own note) **is** categorically off limits — per the task's
ambiguity rule, no attempt was made to read anything under `shared`, even read-only. Only
`Read`/`Grep`/`Glob` were used against `grid-trading-bot`; nothing was written there. Files
matching secret-like names were not opened (none were encountered).

## Result: PREMISE REFUTED — and the writer has since shipped a migration that resolves the ambiguity

`grid-trading-bot\state_writer.py` now differs from the 09-11/09-12 version the prior check
read. It carries a dated review-finding comment ("F1 (review finding, 2026-09-13)",
`state_writer.py:90`) and a schema-version bump (`_STATE_SCHEMA_VERSION = 1`,
`state_writer.py:58`) implementing a documented migration ("spec:
state-integrity-drawdown-spec.md sec 5.4", `state_writer.py:54-57`):

- `drawdown_pct` (`state_writer.py:80,247-251,300-301`) — **FROZEN legacy field**: peak-to-
  current drawdown as a **fraction**, unchanged semantics and units, kept only for backward
  compatibility with alphahive consumers named in `tests/test_drawdown_metrics.py:11-19`
  (`agents/apothecary_agent.py:191`, `agents/portfolio_monitor.py:30`,
  `memory/memory_agent.py:151` — read about, not opened, since they are outside this cage's
  read grant).
- `drawdown_pct_peak` (`state_writer.py:252-253,303`) — the **same** peak-to-current formula,
  now published in **PERCENT** (`round(drawdown_pct * 100.0, 6)`).
- `loss_from_start_pct` (`state_writer.py:254-256,304-312`) — a **new, distinct** metric:
  `max(0.0, -total_pnl / starting_capital) * 100.0`, i.e. exactly the starting-capital-basis
  percent figure the task's claim says fleet_health computed as "1.3176".

Live `state.json` (read this session, `last_activity_at: 2026-09-14T14:00:21Z`,
`cycle_count: 469`):
```
pnl_total            = -26.2558
unrealized_pnl       = -0.4301
total_pnl            = -26.6859
starting_capital     = 2000.0
peak_equity_usd      = 2011.2993
drawdown_pct         = 0.018886   (frozen legacy fraction)
drawdown_pct_peak    = 1.88859    (same metric, percent)
loss_from_start_pct  = 1.334294   (NEW: starting-capital basis, percent)
```
Hand-recomputation: `current_equity = 2000.0 + (-26.6859) = 1973.3141`;
`(2011.2993 - 1973.3141) / 2011.2993 = 0.0188861...` → matches `drawdown_pct` (0.018886) and
`drawdown_pct_peak` (1.8886, vs. published 1.88859 — sub-0.0001 rounding-order difference,
consistent with the two fields being rounded independently at 6 decimals). Separately,
`-total_pnl / starting_capital * 100 = 26.6859 / 2000 * 100 = 1.334295` → matches
`loss_from_start_pct` (1.334294, same sub-0.000001 rounding-order artifact). Both formulas
verified exactly against live data; the writer computes precisely what it now documents.

This both reconfirms the prior check's conclusion (the writer was never wrong about the
metric it computed) and **closes the underlying ambiguity the claim's framing traded on**: as
of this migration, `state.json` no longer forces a consumer to choose one basis — it
publishes the peak-relative metric (legacy `drawdown_pct` + its percent twin
`drawdown_pct_peak`) and the start-relative metric (`loss_from_start_pct`) side by side, under
unambiguous names. A consumer reading `loss_from_start_pct` would get ~1.33, matching the
claim's "fleet_health computes 1.3176" to within the capital drift between the two snapshots
(`pnl_total` moved from -26.35 at claim-time to -26.2558 now).

`tests/test_drawdown_metrics.py` (339 lines, read in full) is a purpose-built contract-test
file for exactly this migration: `test_legacy_drawdown_pct_is_frozen_as_a_fraction` (positive
control — fires if `drawdown_pct` is ever redefined), `test_peak_and_start_metrics_discriminate_once_peak_exceeds_start`
(proves the two metrics are genuinely different numbers, not one aliased twice), and a
`TestRestartShape` class pinning a second, related review finding (F1: the migrated fields
must be **absent**, not a fake `0.0`, until the first `push_state()` after a restart, so a
consumer never sees a real restored peak beside a fabricated zero drawdown — "ABSENT IS
HONEST; ZERO IS A LIE", `state_writer.py:106`). This test file was read, not executed (no
shell in this cage).

## What this check could not see

- No shell access in this cage, so `tests/test_drawdown_metrics.py` was read but not run;
  whether it currently passes against the live `state_writer.py` was not confirmed by
  execution, only by manual formula tracing against live `state.json` output (above).
- `shared` (where `fleet_health.py`/`health.py`/`criteria.py` — the actual consumer of these
  fields — live) is categorically off limits to this cage. Whether `criteria.py` has been
  updated to read `loss_from_start_pct` (rather than recomputing a peak-relative figure from
  `peak_equity_usd` and comparing it against a starting-capital basis, the mismatch the prior
  check's file and this file's own review-finding comments describe) was **not verified** —
  only inferred from `state_writer.py`'s own comments about `criteria.py`'s past behavior
  (`state_writer.py:96-102`, `test_drawdown_metrics.py:294`).
- No `git log`/`git blame` (no shell) — the "2026-09-13" date and "F1 (review finding)"
  attribution are taken from source comments, not independently confirmed against commit
  history.
- The `state-integrity-drawdown-spec.md` file itself was not found inside `grid-trading-bot`
  (`Glob` returned no matches) — it is referenced as living under `shared/docs/specs/`
  (`tests/test_drawdown_metrics.py:5`), which is off limits, so its full text (sections 3.1,
  3.2, 4, 5.4, 6.3, cited by source comments) was not read directly, only as cited.
- The exact `fleet-health-latest.json` state_integrity artifact the claim quotes
  (`pnl_total -26.35`, computed `1.3176`) was not located or read (unknown location; if under
  `shared` or `overnight-waves` it would be off limits regardless) — this check verified the
  writer's current formulas and current live output, not the exact historical snapshot the
  claim's numbers came from.

## Verdict

**REFUTED**, and materially different from the prior (09-11) check's basis for saying so: it
is no longer just that the writer computed its one existing metric correctly — the writer has
since (per its own dated comments, ~2026-09-13) shipped a schema migration
(`state_schema_version: 1`) that adds `loss_from_start_pct`, the exact starting-capital-basis
percent metric this claim's "fleet_health computes 1.3176" describes, published alongside the
unchanged legacy `drawdown_pct` fraction and its new percent twin `drawdown_pct_peak`. Verified
by hand-recomputation against live `state.json` (both formulas match to 5+ significant
figures) and by reading the purpose-built `tests/test_drawdown_metrics.py` contract suite,
which exists specifically to pin this migration and its restart-shape edge case. Recommend the
operator close this as "premise stale/refuted — writer already migrated (schema v1,
loss_from_start_pct added ~2026-09-13), superseded by
`pc-2026-09-11-grid-drawdown-writer`" rather than routing a fix lane at `grid-trading-bot`'s
state writer. If any gap remains, it is consumer-side (whether `fleet_health`/`criteria.py` in
`shared` reads the new field yet), which requires a session with access to `shared` that this
cage does not have.
