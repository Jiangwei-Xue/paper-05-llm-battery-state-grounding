#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys

root = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Replay all saved experimental evidence offline.")
parser.add_argument("--archive-context", help="Archive path recorded by the release workflow.")
parser.add_argument("--output-root", type=pathlib.Path, help="Output directory; defaults to a sibling of the extracted release.")
args = parser.parse_args()
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 required; found {sys.version}")
paper_manifest = json.loads((root / "LATEST_PAPER.json").read_text())
paper_hashes_verified = 0
for paper_file in paper_manifest["files"]:
    paper_path = root / paper_file["path"]
    if not paper_path.is_file():
        raise SystemExit(f"latest paper file missing: {paper_file['path']}")
    observed = hashlib.sha256(paper_path.read_bytes()).hexdigest()
    if observed != paper_file["sha256"]:
        raise SystemExit(f"latest paper hash mismatch: {paper_file['path']}")
    paper_hashes_verified += 1
workspace = root / "frozen_inputs/project"
scripts = root / "scripts"
outputs = (args.output_root.resolve() if args.output_root else root.parent / f"{root.name}_reproduced_outputs")
if outputs.exists():
    shutil.rmtree(outputs)
outputs.mkdir()
vendor = outputs / "runtime_vendor"
network_log = outputs / "NETWORK_ATTEMPTS.jsonl"
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["OFFLINE_NETWORK_LOG"] = str(network_log)
env["PYTHONPATH"] = os.pathsep.join([str(scripts), str(vendor)])
commands = [
    [sys.executable, str(scripts/"materialize_runtime.py"), "--base", str(root/"runtime/vendor"), "--encoded", str(root/"runtime/encoded"), "--output", str(vendor)],
    [sys.executable, str(scripts/"verify_scientific_release.py"), "--output", str(outputs/"release_contract/SCIENTIFIC_RELEASE_AUDIT.json")],
    [sys.executable, "-m", "unittest", "discover", "-s", str(root/"tests"), "-p", "test_*.py"],
    [sys.executable, str(scripts/"reparse_saved_responses_v1_3.py"), "--output", str(outputs/"parser/PARSER_REPLAY_VERIFICATION.json")],
    [sys.executable, str(scripts/"recompute_offline_v1_7.py"), "--project", str(workspace), "--output", str(outputs/"primary")],
    [sys.executable, str(scripts/"run_qwen37_offline_standardization_v1_7.py"), "--project", str(workspace), "--reference-script", str(scripts/"recompute_offline_v1_7.py"), "--output", str(outputs/"qwen37")],
    [sys.executable, str(scripts/"independent_physics_checker_v1_7.py"), "--project", str(workspace), "--offline-root", str(outputs/"primary"), "--extra-gate-sidecars", str(outputs/"qwen37/gate_action_sidecars.jsonl.gz"), "--output", str(outputs/"physics")],
    [sys.executable, str(scripts/"build_offline_completion_audit_v1_7.py"), "--project", str(workspace), "--offline-root", str(outputs/"primary"), "--qwen-root", str(outputs/"qwen37"), "--independent-report", str(outputs/"physics/INDEPENDENT_PHYSICS_CROSSCHECK.json"), "--output", str(outputs/"audit")],
    [sys.executable, str(scripts/"run_offline_optimization_sensitivities_v1_7.py"), "--project", str(workspace), "--reference-script", str(scripts/"recompute_offline_v1_7.py"), "--offline-root", str(outputs/"primary"), "--qwen-root", str(outputs/"qwen37"), "--output", str(outputs/"sensitivities")],
    [sys.executable, str(scripts/"feasibility_semantics_v1_7.py"), "--primary", str(outputs/"primary"), "--qwen", str(outputs/"qwen37"), "--output", str(outputs/"semantics")],
    [sys.executable, str(scripts/"regenerate_release_tables_v1_7.py"), "--release", str(root), "--reproduced-root", str(outputs)],
    [sys.executable, str(scripts/"release_acceptance_red_team_v1_7.py"), "--release", str(root), "--project", str(workspace), "--reproduced-root", str(outputs)],
    [sys.executable, str(scripts/"verify_vcr_h5e5.py"), "--reproduced-root", str(outputs)],
]
executed = []
for command in commands:
    completed = subprocess.run(command, cwd=root, env=env)
    display = [str(item).replace(str(root), "$RELEASE") for item in command]
    display[0] = pathlib.Path(display[0]).name
    executed.append({"command": display, "exit_code": completed.returncode})
    if completed.returncode:
        (outputs/"RUN_ALL_OFFLINE_REPORT.json").write_text(json.dumps({"status":"FAIL","commands":executed},indent=2)+"\n")
        raise SystemExit(completed.returncode)
attempts = 0
if network_log.exists():
    attempts = sum(1 for line in network_log.read_text().splitlines() if line.strip())
if attempts:
    raise SystemExit(f"network attempts blocked: {attempts}")
expected = root / "reference_outputs"
comparisons = {}
for relative in (
    "semantics/FEASIBILITY_SEMANTICS.json",
    "tables/table_feasibility_by_replay_state.tex",
    "tables/table_objective_order_sensitivity.tex",
    "vcr_h5e5/VCR_H5E5_VERIFICATION.json",
    "parser/PARSER_REPLAY_VERIFICATION.json",
):
    actual = outputs / relative
    reference = expected / relative
    if reference.exists():
        comparisons[relative] = actual.read_bytes() == reference.read_bytes()
if comparisons and not all(comparisons.values()):
    raise SystemExit(f"expected-output mismatch: {comparisons}")
report = {"status":"PASS","commands":executed,"expected_output_comparisons":comparisons,"latest_paper_files_verified":paper_hashes_verified,"provider_calls_performed":0,"network_attempts":0}
(outputs/"RUN_ALL_OFFLINE_REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(f"OFFLINE_REPRODUCTION_PASS output={outputs} provider_calls=0 network_attempts=0")
