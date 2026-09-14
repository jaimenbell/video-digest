# Premise check: pc-2026-09-12-busmcp-mock-parity

**Task id:** 026125376c9448bf83754e6eb2182c00
**Repo this session was caged to:** video-digest (`C:\Users\jaime\projects\video-digest`)
**Branch:** lane/g3-audit

## Claim under test

Task title: *"bus-mcp board tests use respx mocks and no test compares the mock's row
shape to the live bus board (which now carries `origin_thread_id` as its 15th key since
alphahive master@265106c1)."*

This is the same claim, verbatim, as `pc-2026-09-11-busmcp-mock-parity.md` (task
`64cbc0d82d2a4e65a9f8493a5e698524`), re-minted a day later as a fresh task-board item under a
new task id. This check independently re-verified the same claim from inside this session
rather than trusting the prior file's conclusion.

## Scope note

`bus-mcp` is not one of the three repos CLAUDE.md names as categorically off limits to this
cage (`overnight-waves`, `alphahive`, `shared`), and this session's brief explicitly grants
read access to it by name. Only `Read`/`Grep`/`Glob` were used against `bus-mcp`; nothing was
written there.

## Result: PREMISE STILL REFUTED — the gap remains closed, re-confirmed independently

Re-read the same files fresh this session:

- `bus-mcp/tests/test_board.py:1-200` — confirmed `TestListTasksBoard` still mocks
  `/tasks/board` with small hand-rolled JSON bodies (e.g. `{"task_id": "a", "status":
  "pending"}` at `test_board.py:132`), and this file alone does not cross-check those mock
  shapes against the live board.
- `bus-mcp/tests/test_board_parity.py` (178 lines, unchanged from the prior check) — still
  present, still does the comparison the claim says is missing: `TestLiveBoardRowParity`
  loads a committed real-row fixture and asserts (a) no `config.BOARD_SENSITIVE_COLUMNS`
  member leaks onto it, and (b) the fixture's key set matches `EXPECTED_BOARD_KEYS`
  (`test_board_parity.py:44-62`), which includes `origin_thread_id` (`:60`) — the exact 15th
  key this premise names.
- `bus-mcp/tests/fixtures/board_row_live.json` — unchanged, still `_captured_at:
  2026-09-12T01:41:29Z`, `origin_thread_id: 3` present in the `row` object (line 20).
- `bus_mcp/config.py:54-66` — `BOARD_SENSITIVE_COLUMNS` / `BOARD_SENSITIVE_COLUMNS_TEXT`
  confirmed present and referenced by `test_board_parity.py`'s leak-detector test.
- `grep -rn "origin_thread_id|approval_ref|_BOARD_COLUMNS|BaseModel|TypedDict"` across
  `bus-mcp/` still returns exactly the same three files as the prior check:
  `bus_mcp/config.py`, `tests/fixtures/board_row_live.json`, `tests/test_board_parity.py`.
- No `pyproject.toml`/`pytest.ini` collection exclusion of `test_board_parity.py` was found
  (`tests/conftest.py` exists but was not seen to skip or ignore it), so the parity file is
  collected by default `pytest` discovery, not orphaned.

This directly contradicts "no test compares the mock's row shape to the live bus board":
`test_board_parity.py` is exactly that comparison, keyed on the real 15-column shape
including `origin_thread_id`, and nothing observed this session shows it was removed,
disabled, or bypassed since the prior check one day earlier.

## What this check could not see

- No shell access in this cage, so no `git log`/`git status` was run against `bus-mcp` to
  confirm whether `test_board_parity.py` is committed, or to confirm it currently passes
  under `pytest`. Existence and assertion content were verified by reading the file, not by
  executing it.
- Did not read `alphahive`'s `coordination_bus.py` directly (categorically off limits to this
  cage); the "15th key / `origin_thread_id`" fact is taken from the staged review document
  referenced by the task, not independently re-derived from `alphahive` in this session.
- This is a same-day-adjacent re-check of an already-refuted premise; if the task board minted
  it because provenance (committed vs. working-tree-only) was still open, that gap is not
  closed by this check either — it still needs a session with shell access to `bus-mcp`.

## Verdict

**REFUTED (duplicate of `pc-2026-09-11-busmcp-mock-parity`, independently re-confirmed).** The
premise-check task names a gap that was already closed before the prior task id was even
minted, and remains closed as of this session. Recommend the task settle as "premise stale,
duplicate of `64cbc0d82d2a4e65a9f8493a5e698524`, no action needed" rather than routing a fix
lane at `bus-mcp`.
