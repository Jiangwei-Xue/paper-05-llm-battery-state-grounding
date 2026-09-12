# Experiment scaffolding

This directory is the map for the experiment implementation retained under
`frozen_inputs/project/`.  It complements the one-command saved-evidence replay
with the protocols, request builders, branch schedules, parsers, live runner
sources, deterministic physics, scoring code, aggregation, and tests needed to
understand how the hosted experiments were executed.

Two execution modes are intentionally separate:

1. `RUN_ALL_OFFLINE.py` is the verified scientific reproduction path.  It uses
   saved requests and responses, performs no provider call, and rebuilds replay,
   scores, comparisons, sensitivities, and paper tables.
2. The live runner sources under `frozen_inputs/project/` can issue new hosted
   calls only when the operator supplies provider credentials in environment
   variables and passes the runner's explicit execution authorization flag.
   Such calls create new, non-canonical outputs and cannot reproduce hosted
   responses byte for byte.

Read `EXPERIMENT_MAP.md` for the scientific design and `LIVE_RUNNER_GUIDE.md`
for the guarded live entry points. `RUNNER_INDEX.json` is the machine-readable
path inventory.

