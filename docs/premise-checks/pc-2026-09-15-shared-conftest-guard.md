VERDICT: CONFIRMED

# premise-check: shared-conftest-guard-2026-09-15

**Task id:** 62e97f49232a4a3ba9bcf4f0ff977f63
**Repo this session was caged to:** video-digest (`C:\Users\jaime\projects\video-digest`)
**Branch:** lane/g3-audit
**Subject (read-only grant):** `shared/conftest.py`, `shared/tests/test_worktree_resolution_guard.py`

## Claim under test

Memory note `shared-bare-import-blast-radius-measured` fix-queue item (2): "STILL OPEN: conftest
resolution guard failing on out-of-checkout modules, both controls in one commit" — versus the same
note's claim that the guard's "both-directions demo PASSED 2026-08-31". Premise tested: *item (2) is
still open — the guard, or one of its two controls, is not committed.*

## 1. Does `shared/conftest.py` contain a worktree-resolution guard?

Yes. `pytest_collection_finish(session)` — `shared/conftest.py:419-457`. It resolves this
worktree's root (`_worktree`, line 14), calls `_main_checkout_root(root)` (`conftest.py:250-271`) to
find the live checkout via the worktree's `.git` gitfile, and if this run *is* a worktree, calls
`_out_of_checkout_modules(root, live)` (`conftest.py:274-295`) to find any `sys.modules` entries whose
`__file__` resolved under the live checkout rather than this worktree (excluding `.venv`/
`site-packages`). If any offenders are found it raises `pytest.UsageError` (exit code 4) naming each
leaked module (`conftest.py:448-456`).

## 2. Does the test file contain both controls?

Present, but not as the spec defines a control:

- `TestOutOfCheckoutModules.test_fires_on_live_resolved_module` —
  `shared/tests/test_worktree_resolution_guard.py:47-58`. Arranges a leaked module, calls
  `root_conftest._out_of_checkout_modules(wt, live)` directly, and asserts
  `"entry_halt_watch" in offenders`.
- `TestOutOfCheckoutModules.test_silent_on_worktree_resolved_module` —
  `shared/tests/test_worktree_resolution_guard.py:60-71`. Arranges a clean resolution, calls the
  same helper directly, and asserts `"entry_halt_watch" not in offenders`.

Both tests exercise `_out_of_checkout_modules` — the guard's **helper** — and assert on its return
value. Neither test calls `pytest_collection_finish` (the guard itself) or asserts that it
raises/does-not-raise `pytest.UsageError`. Per the spec's own definition ("A test that only checks
the guard's helper returns a value is NOT a control — a control exercises the fire/no-fire
behaviour"), neither qualifies as the FIRES or STAYS-SILENT control.

The only fire/no-fire evidence for the guard itself is documented as a *manual, uncommitted* demo in
the conftest docstring at `shared/conftest.py:435-438`: "Positive control: a throwaway worktree with
this conftest fires on the three known-leaking modules. Negative control: the live full suite runs
silent." That is a description of a one-off demonstration, not automated test code in
`test_worktree_resolution_guard.py`.

## 3. Verdict

**CONFIRMED** — item (2) is still open. The guard (`pytest_collection_finish`) is committed in
`conftest.py`, but `test_worktree_resolution_guard.py` commits only helper-level tests
(`_out_of_checkout_modules` in isolation); it does not commit a FIRES control or a STAYS-SILENT
control that exercises `pytest_collection_finish`'s actual raise/no-raise behavior. The "both-directions
demo PASSED 2026-08-31" claim in the memory note refers to the manual demo described in the conftest
docstring, not to committed automated controls — so statement A (item (2) open) and statement B (demo
passed) are both accurate and do not actually contradict each other once "committed control" is read
strictly per this spec's definition.

## What this check could not see

This check read only the two granted files (`shared/conftest.py`,
`shared/tests/test_worktree_resolution_guard.py`) and did not execute pytest. It could not see: (a)
whether an integration-style FIRES/STAYS-SILENT control exists anywhere else in the `shared` repo
outside this one test file; (b) whether the manual throwaway-worktree demo described in the conftest
docstring was ever automated in a later, unreviewed commit; (c) the actual runtime behavior of
`pytest_collection_finish` (raise/no-raise) under either condition, since running pytest against
`shared` was out of scope for this cage; (d) any git history or commit metadata for either file, since
only their current on-disk contents were read.
