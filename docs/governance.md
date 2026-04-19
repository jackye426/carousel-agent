# Governance defaults

These flags control optional review loops on top of the Writer. **Defaults favour shipping speed** while keeping ideation oversight on.

| Flag | CLI | Default | When to change |
|------|-----|---------|----------------|
| **Ideation governance** | `--judge-govern` / `--no-judge-govern` | **On** | Turn **off** only for quick drafts or debugging (shortlist + carousel review can veto or force rewrites). |
| **Hook language policy** | `--hook-policy` / `--no-hook-policy` | **Off** | Turn **on** when banned phrases or brand language must be enforced; uses `MODEL_HOOK_JUDGE` and `hook_policy` in code. See `carousel_agents/hook_policy.py`. |
| **Editor QA** | `--qa` / `--no-qa` on `run` | **Off** | Turn **on** for publish-ready passes (`MODEL_EDITOR`). |

**`run-hooks-cta`** and **`redraft-slides`** expose the same concepts where relevant (hook policy, judge rounds).

For the engineering checklist (schema version, tests, paths, observability), see **`TODO.md`**. Strategy (personas, performance) is in **`docs/audience-and-personas-plan.md`**.
