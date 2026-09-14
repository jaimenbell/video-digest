# Premise check: pc-2026-09-11-conftest-depth

**Task id:** 18d00760328945c9ac7dd71a0a25d4cb
**Repo this session was caged to:** video-digest (`C:\Users\jaime\projects\video-digest`)
**Branch:** lane/g3-audit

## Claim under test

From the task title and the staged spec (`C:/Users/jaime/.claude/staged/converge-w6-ack-parity-2026-09-11.md`, line 12):

> overnight-waves `conftest.py` computes `_PROJECTS_ROOT = REPO_ROOT.parent`, which is 3 levels
> short inside `.claude/worktrees/<x>`; every worktree suite needs
> `PYTHONPATH=C:\Users\jaime\projects`. One-line fix candidate for the next wave (walk up until
> a dir containing `shared/` is found).

The spec records this as a defect *surfaced but not fixed*, flagged as a candidate for "the next
wave" (i.e. a future task against the `overnight-waves` repo).

## Result: BLOCKED — premise cannot be checked from this cage

This session is caged to the `video-digest` repo, writable only inside `docs/premise-checks`.
CLAUDE.md's cage rules for this task class state: *"Never edit the cage. `overnight-waves`,
`alphahive` and `shared` are off limits categorically; you are not permitted to work in them at
all."* Verifying the claim requires reading `overnight-waves/conftest.py` (and possibly
`overnight-waves/.claude/worktrees/*`) — that repo is categorically off limits to this session, so
per the task's ambiguity rule ("if you cannot tell whether an action falls inside the allowed
column ... you are in the forbidden column") no attempt was made to read it, even read-only.

Confirmed from inside the allowed repo only:
- `video-digest` has no `conftest.py` anywhere in the tree (`Glob: conftest.py` → no matches).
- `video-digest` has no `docs/` directory prior to this write (`Glob: docs/**` → no matches before
  this file was created).

Both checks show the claim is entirely about `overnight-waves` and has no footprint in
`video-digest` — there is nothing in this repo that this session could use to independently
corroborate or refute the `_PROJECTS_ROOT` claim.

## Verdict

INDETERMINATE / BLOCKED. This premise names an artifact (`overnight-waves/conftest.py`) that this
caged session has no authorized access to. Recommend the operator route this premise-check to a
session caged to `overnight-waves` (read-only is sufficient — no fix was requested here, only
verification of the premise).
