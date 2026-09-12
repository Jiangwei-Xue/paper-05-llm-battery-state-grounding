from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseContractTests(unittest.TestCase):
    def test_expected_record_counts(self) -> None:
        expected = json.loads((ROOT / "VCR_H5E5_EXPECTED.json").read_text())
        manifest = json.loads((ROOT / "experiments_manifest.json").read_text())
        observed = {row["id"]: row["records"] for row in manifest["experiments"]}
        self.assertEqual(observed, expected["record_counts"])
        self.assertEqual(manifest["total_claim_bearing_rows"], 2940)
        self.assertEqual(manifest["saved_provider_calls_represented"], 3538)

    def test_all_paper_tables_are_mapped(self) -> None:
        tex = (ROOT / "paper_reference/main.tex").read_text()
        labels = set(re.findall(r"\\label\{(tab:[^}]+)\}", tex))
        manifest = json.loads((ROOT / "results_manifest.json").read_text())
        mapped = {row["paper_label"] for row in manifest["core_tables"]}
        self.assertEqual(labels, mapped)
        self.assertEqual(len(mapped), 9)

    def test_latest_paper_hashes(self) -> None:
        manifest = json.loads((ROOT / "LATEST_PAPER.json").read_text())
        for item in manifest["files"]:
            self.assertEqual(sha(ROOT / item["path"]), item["sha256"])

    def test_public_record_manifest(self) -> None:
        rows = [
            json.loads(line)
            for line in (ROOT / "manifests/PUBLIC_RECORD_MANIFEST.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 825)
        self.assertEqual(len({row["path"] for row in rows}), 825)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))

    def test_no_presentations_or_nested_archives(self) -> None:
        presentations = {".ppt", ".pptx", ".pps", ".ppsx", ".key", ".odp"}
        archives = {".zip", ".tar", ".tgz", ".gz", ".7z", ".rar"}
        files = [path for path in ROOT.rglob("*") if path.is_file()]
        self.assertFalse([path for path in files if path.suffix.lower() in presentations])
        self.assertFalse([path for path in files if path.suffix.lower() in archives])

    def test_no_hidden_version_control_metadata(self) -> None:
        forbidden = {".git", ".gitmodules", ".gitattributes", ".gitignore", ".hg", ".svn"}
        packaged_metadata = []
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            # A normal Git checkout necessarily has a repository-owned top-level
            # .git directory. It is runtime context, not a packaged member.
            if relative.parts and relative.parts[0] == ".git":
                continue
            if any(part in forbidden for part in relative.parts):
                packaged_metadata.append(path)
        self.assertFalse(
            packaged_metadata
        )

    def test_experiment_scaffolding_and_live_runners(self) -> None:
        index = json.loads((ROOT / "experiment_scaffolding/RUNNER_INDEX.json").read_text())
        experiments = {row["experiment"] for row in index["experiments"]}
        self.assertEqual(experiments, {"E1", "E2", "F1", "F0", "QWEN37"})
        for row in index["experiments"]:
            for key in ("protocol", "pre_call_manifest", "runner"):
                self.assertTrue((ROOT / row[key]).is_file(), f"missing {row['experiment']} {key}")

    def test_runtime_materialization_payloads(self) -> None:
        manifest = json.loads(
            (ROOT / "runtime/encoded/RUNTIME_MATERIALIZATION_MANIFEST.json").read_text()
        )
        self.assertGreater(len(manifest["files"]), 0)
        for row in manifest["files"]:
            payload = ROOT / "runtime/encoded" / row["encoded_path"]
            self.assertTrue(payload.is_file())
            self.assertEqual(sha(payload), row["encoded_sha256"])


if __name__ == "__main__":
    unittest.main()
