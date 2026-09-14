# Premise check: pc-2026-09-11-busmcp-mock-parity

**Task id:** 64cbc0d82d2a4e65a9f8493a5e698524
**Repo this session was caged to:** video-digest (`C:\Users\jaime\projects\video-digest`)
**Branch:** lane/g3-audit

## Claim under test

Task title: *"bus-mcp board tests use respx mocks and no test compares the mock's row
shape to the live bus board (which now carries `origin_thread_id` as its 15th key since
alphahive master@265106c1)."*

The staged spec (`C:/Users/jaime/.claude/staged/review-w1-board-origin-thread-opus-2026-09-11.md`,
finding #18, line 39) is the source of this claim: it flags `bus-mcp/tests/test_board.py`'s
respx mocks as "a test double that structurally cannot detect a real board-shape change" and
explicitly logs it as "not this lane's repo; log it, don't block."

## Scope note

`bus-mcp` is not one of the three repos CLAUDE.md names as categorically off limits to this
cage (`overnight-waves`, `alphahive`, `shared`). Verifying this premise requires reading
`bus-mcp`'s own test tree, which is the repo the claim is actually about — so, unlike the
`overnight-waves` premise-check on the same date (`pc-2026-09-11-conftest-depth.md`), this one
was checkable. Only `Read`/`Grep`/`Glob` were used against `bus-mcp`; nothing was written
there.

## Result: PREMISE REFUTED — the gap already closed

The claim was true as of the review that raised it, but is **no longer true**. `bus-mcp` now
has a dedicated parity test file that does exactly what the claim says is missing:

- `bus-mcp/tests/test_board_parity.py` (178 lines) — `TestLiveBoardRowParity` loads a
  committed real-row fixture and asserts (a) no `config.BOARD_SENSITIVE_COLUMNS` member
  leaks onto it, and (b) the fixture's key set matches `EXPECTED_BOARD_KEYS`
  (`test_board_parity.py:44-62`), a literal that **includes `origin_thread_id`**
  (`test_board_parity.py:60`) — the exact 15th key this premise names.
- `bus-mcp/tests/fixtures/board_row_live.json` — one real row "captured verbatim from the
  live coordination-bus board" (`_captured_from`, line 3) at `_captured_at:
  2026-09-12T01:41:29Z`, with `origin_thread_id: 3` present in the `row` object (line 20).
  The fixture is a real observed row, not a hand-typed shape guess.
- `TestParityChecksHavePositiveControl` (`test_board_parity.py:124-151`) is a positive
  control: it mutates an in-memory copy of the fixture to inject an unexpected column, delete
  `origin_thread_id`, and inject a sensitive-column leak, and asserts the detector fires on
  each — plus stays silent on the unmutated real row. The file's own docstring
  (`test_board_parity.py:1-29`) states the committed fixture was also empirically mutated on
  disk and reverted, to prove the same drift would be caught for real, not just in the
  in-memory positive-control copies.

This directly contradicts "no test compares the mock's row shape to the live bus board":
`test_board_parity.py` is exactly that comparison, and it is keyed on the real 15-column shape
including `origin_thread_id`.

What the premise still gets right, narrowly: `bus-mcp/tests/test_board.py` itself (445 lines,
confirmed via `Read`) does still mock `/tasks/board` with small hand-rolled JSON bodies (e.g.
`{"task_id": "a", "status": "pending"}` at `test_board.py:132`), and nothing in that file
cross-checks those mock shapes against the live board. But that file is no longer the whole
story — `test_board_parity.py` is a separate, already-existing file that covers the shape-drift
risk the original claim was worried about. `grep -rn "origin_thread_id" bus-mcp/` found matches
in exactly three files: `bus_mcp/config.py`, `tests/fixtures/board_row_live.json`, and
`tests/test_board_parity.py` — i.e. the awareness of this column is already wired through
config and the parity test, not absent.

## What this check could not see

- No shell access in this cage, so no `git log`/`git status` was run against `bus-mcp` to
  confirm whether `test_board_parity.py` is committed, staged, or still working-tree-only, or
  which commit/lane authored it. The file's own docstring and the fixture's `_captured_at`
  timestamp (2026-09-12T01:41, after the review that raised this premise) are the only dating
  evidence available from inside this cage.
- Did not execute `pytest` against `bus-mcp` (no shell), so the parity tests are confirmed to
  exist and to assert the right things by reading them, not confirmed to currently pass.
- Did not read `alphahive`'s `coordination_bus.py` directly (categorically off limits to this
  cage); the "15th key / `origin_thread_id`" fact is taken from the already-provided staged
  review document, not independently re-derived from `alphahive` in this session.

## Verdict

**REFUTED.** The premise-check task names a gap that has already been closed by
`bus-mcp/tests/test_board_parity.py` + `tests/fixtures/board_row_live.json`, both present on
disk and both keyed on `origin_thread_id`. Recommend the task settle as "premise stale, no
action needed" rather than routing a fix lane at `bus-mcp` — the fix this task would have asked
for already exists. If provenance (committed vs. uncommitted, which lane) matters to the
operator, that needs a session with shell access to `bus-mcp`, which this cage does not have.
