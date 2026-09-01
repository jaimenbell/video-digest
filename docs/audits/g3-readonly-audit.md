# G3 supervised read-only audit — docs-vs-reality consistency

Class-1 READ-ONLY audit. You are the caged Task-Board worker session. This file is
the task authority (the frame that loaded it explains the cage and the findings
contract).

## The audit

Check whether this repo's documentation matches its code reality. Specifically:

1. Every command, script entry point, or invocation documented in `README.md`
   (and any other top-level `*.md`): does the referenced module/file/entry point
   actually exist in the repo, and do its documented flags match the code?
2. Every dependency named in documentation prose: is it present in
   `pyproject.toml`?
3. Any config keys, environment variables, or file paths that documentation
   claims exist: verify each has a real referent in the code (cite file:line).

## Rules for this audit

- READ-ONLY: write NOTHING inside this repository. Your findings go ONLY to the
  `findings.json` artifact path the session frame gave you (it is outside the
  repo by design).
- Every claim needs evidence (`file:line` or the exact command a human could
  run). A stale-doc claim must cite both the doc line and the code reality.
- An empty findings array is a legitimate, correct answer if the docs are clean.
  Do not invent findings.
- State blind spots: anything you could not check, say so in a final findings
  entry with `status: "unchecked"`.
