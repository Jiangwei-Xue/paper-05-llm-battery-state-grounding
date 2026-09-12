from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "protocol_code"))

from e2b_parser_v2 import parse_output as parse_e2b  # noqa: E402
from experiments_v2.p0 import STATE_FIELDS, parse_output as parse_p0  # noqa: E402


class ParserContractTests(unittest.TestCase):
    def test_e2b_sparse_action(self) -> None:
        parsed = parse_e2b(
            '{"actions":[{"start_offset":1,"end_offset_exclusive":3,"power_kw":10}]}',
            4,
        )
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.dense_action_kw, [0.0, 10.0, 10.0, 0.0])

    def test_e2b_rejects_overlap(self) -> None:
        parsed = parse_e2b(
            '{"actions":[{"start_offset":0,"end_offset_exclusive":2,"power_kw":1},'
            '{"start_offset":1,"end_offset_exclusive":3,"power_kw":1}]}',
            4,
        )
        self.assertFalse(parsed.ok)
        self.assertIn("segment_1_unsorted_or_overlap", parsed.diagnostics)

    def test_e2b_rejects_nonfinite_json(self) -> None:
        parsed = parse_e2b('{"actions":[{"start_offset":0,"end_offset_exclusive":1,"power_kw":NaN}]}', 1)
        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.diagnostics, ["invalid_json"])

    def test_p0_dense_contract(self) -> None:
        state = {field: [] for field in STATE_FIELDS}
        state.update(
            {
                "current_step": 0,
                "sequence": 0,
                "soc_kwh": 250.0,
                "usable_capacity_kwh": 500.0,
                "charge_limit_kw": 250.0,
                "discharge_limit_kw": 250.0,
                "export_limit_kw": 150.0,
                "reserve_kwh": 75.0,
                "forecast_version": "v1",
                "active_commitments": [],
                "revoked_commitments": [],
            }
        )
        parsed = parse_p0(json.dumps({"state": state, "actions_kw": [1, 0]}), "I0", 2)
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.dense_action_kw, [1.0, 0.0])

    def test_p0_rejects_wrong_length(self) -> None:
        parsed = parse_p0('{"state":{},"actions_kw":[1]}', "I0", 2)
        self.assertFalse(parsed.ok)
        self.assertIn("state_fields_invalid", parsed.diagnostics)
        self.assertIn("dense_length_1_expected_2", parsed.diagnostics)


if __name__ == "__main__":
    unittest.main()
