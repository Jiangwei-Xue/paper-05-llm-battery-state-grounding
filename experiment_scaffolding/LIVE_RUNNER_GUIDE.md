# Guarded live runner guide

The live runner sources are included to make the hosted-call method auditable
and to permit prospective replication. They are not invoked by the verified
offline reproduction command. A live run needs network access, may incur cost,
and cannot guarantee the same response from a hosted rolling service.

## Safety and provenance contract

- No credential value is included in this package.
- Credentials are read only from `DEEPSEEK_API_KEY`, `QWEN_API_KEY`, or
  `DASHSCOPE_API_KEY`.
- The default runner mode is preflight or plan-only.
- Provider traffic requires an explicit execution flag and the runner's
  authorization flag.
- Frozen plans, model specifications, prompt templates, response schemas,
  retry limits, concurrency, and branch order are checked before a call.
- Requests, responses, timestamps, returned model identity, HTTP status,
  retries, parse results, and hashes are written to a new run directory.
- New outputs are prospective replication data and must not replace the saved
  evidence used by the paper.

## Entry points

Run all commands from `frozen_inputs/project/` with the environment described
by `pyproject.toml` and `uv.lock`.

| Tier | Preflight or plan-only entry | Guarded runner |
|---|---|---|
| E1 | `scripts/run_experiments_v2_e1.py --run-label inspection --plan-only` | same script without `--plan-only` |
| E2 | `scripts/run_experiments_v2_e2.py --run-label inspection --plan-only` | same script without `--plan-only` |
| F1 | `segan_revision_major_v2/scripts/run_e2b_v2.py --preflight --tier F1` | add `--execute --authorize-provider-calls --run-label <LABEL>` |
| F0 | `segan_revision_major_v2/scripts/run_f0_formal_v1.py --preflight` | add `--execute --authorize-provider-calls --run-label <LABEL>` |
| Qwen3.7-Plus | run with `--run-label inspection` and without `--execute`, choosing `--smoke` or `--formal` | add `--execute --authorize-provider-calls` and use a new label |

E1 and E2 are historical runners whose provider identities may now resolve to
later hosted revisions. F1, F0, and Qwen3.7-Plus additionally expose an
explicit authorization boundary. Inspect `RUNNER_INDEX.json` and each frozen
protocol before any prospective call.
