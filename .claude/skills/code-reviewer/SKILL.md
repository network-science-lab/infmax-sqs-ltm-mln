---
name: code-reviewer
description: >
  Code reviewer for infmax-sq-ltm-mln. Use this skill when the user wants to review a GitHub PR —
  researching the changes in depth, discussing findings interactively, building up
  REVIEW_REMARKS.md collaboratively, and posting a formal GitHub review with anchored
  inline comments and a summary. Trigger on: /code-reviewer, /code-reviewer <N>, "review PR",
  "review this PR", "let's review", "look at PR #N", "check this PR".
---

## What this skill does

Three phases: **research -> discuss -> publish**. Do not skip ahead — the user drives the pace.

---

## Phase 1 — Research

Given a PR number, gather everything before presenting findings.

```bash
gh pr view <N> --json title,body,author,baseRefName,headRefName,files
gh pr diff <N>
gh repo view --json nameWithOwner   # capture owner/repo for later
```

Then read the full current content of every changed file — not just the diff lines.
Surrounding context often reveals the real issue.

Also read `README.md` to understand the project's goals and architecture.

### What to focus on

**LTM and sequential seeding correctness** — this is the primary lens:
- Threshold computation: are per-layer or per-actor thresholds applied correctly?
- Activation conditions: is the fraction of active neighbours computed over the right
  neighbourhood (layer-specific vs actorwise)?
- Propagation termination: does the simulation halt correctly when no new activations occur?
- Sequential seeding order: is the seed selection sequence consistent with the defined
  strategy? Are seeds introduced one at a time or in batches as intended?
- Budget constraints: is the seeding budget respected actorwise and layerwise?
  Are `get_seeding_budget_for_network` results used correctly?
- Initial state construction: are `NetworkUpdateBuffer` entries built for the right
  `(actor_id, layer_name, state)` tuples?
- `preselected_actors` ordering: does the ranking list correctly prepend pre-selected
  actors before the remainder of the actor set?
- Ranking completeness: does the ranking cover all actors in the network, with no
  duplicates and no omissions?

**Multilayer network handling**:
- Actor vs node disambiguation: operations that should be actorwise must not collapse
  to single-layer node logic
- Layer iteration: are all layers processed when required, or is a layer silently skipped?
- Inter-layer coupling: if the model aggregates influence across layers, is the
  aggregation rule (AND/OR/weighted) applied as specified?
- `network_diffusion` API contracts: are `MultilayerNetwork`, `MLNetworkActor`, and
  `MLTModel` subclasses used according to the library's expected interface?

**Simulation and reproducibility**:
- Monte Carlo estimation: are enough repetitions used, and is influence spread averaged
  correctly (not summed, not max-pooled)?
- Random seed management: is `random.seed` / `np.random.seed` set before each
  independent trial, not once globally?
- Convergence: does the simulator detect fixed points correctly and not iterate past them?
- Result serialization: are raw results written atomically so a crashed run does not
  produce a partial file that looks complete?

**Architecture and modularity**:
- Is logic in the right layer? Seed selection belongs in seed selectors, LTM propagation
  in model classes, I/O in loaders/result handlers — not mixed
- Duplication: same ranking or threshold logic copy-pasted across selectors or runners
- Interface contracts: does a new seed selector honour the `preselected_actors` protocol
  expected by the model's `determine_initial_states`?
- Invariant violations: does the change silently break an assumption held by
  `network_diffusion` (e.g. actor ordering, state name strings)?
- Pydantic config usage: are new config fields validated at the boundary, not deep inside
  simulation logic?

**Project fit**:
- Does the change align with the sequential seeding investigation described in `README.md`?
- Are new components consistent in style and structure with existing ones?
- Are unit tests present for new stable components (loaders, data structures, utilities)?
  Simulation runners and experimental scripts are exempt.

Do **not** flag style issues that pre-commit already enforces (line length, formatting,
import order, type annotations, etc.). Focus on what the code *does*, not how it looks.

---

## Phase 2 — Discuss

Present your findings concisely — **the user's opinion on every remark is final**.
If they dismiss something, drop it without argument. If they raise something you missed,
take it seriously.

Group findings by severity:

- **Must fix** — bugs, broken contracts, silent correctness failures
- **Should fix** — architectural smells, modularity violations, missing tests for stable code
- **Consider** — suggestions, alternative approaches

Use bullet points. Do not write essays. A well-placed dry observation is fine; anything
that reads as a dig at the author is not.

As each remark is agreed upon, write it to `REVIEW_REMARKS.md`:

```markdown
# PR <N> Review Remarks

## <file_path>

### Line <N> — <short imperative title>
\```python
<the offending snippet>
\```
<What is wrong, why it matters, concrete fix. 2–5 sentences. British English.>

---
```

The user may also dictate remarks directly — write those verbatim.

---

## Phase 3 — Publish to GitHub

When the user says `REVIEW_REMARKS.md` is ready, submit a single formal GitHub review.

Build the payload from `REVIEW_REMARKS.md`:

- **Summary body**: 2–3 sentences covering the most important points. Plain prose, no headers,
  no bullet list.
- **Inline comments**: one per remark, anchored to the stated line number.
- **Event**: ask the user — `COMMENT`, `APPROVE`, or `REQUEST_CHANGES`.

```bash
gh api repos/{owner}/{repo}/pulls/{N}/reviews \
  --method POST \
  --input - <<'EOF'
{
  "body": "<summary paragraph>",
  "event": "COMMENT",
  "comments": [
    {"path": "src/foo.py", "line": 42, "side": "RIGHT", "body": "<remark text>"},
    ...
  ]
}
EOF
```

Use RIGHT side and the new-file line number for each comment. For remarks on deleted lines
with no corresponding new line, anchor to the last line of the enclosing hunk.

Print the review URL once posted.
