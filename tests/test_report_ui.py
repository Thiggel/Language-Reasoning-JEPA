import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_ui", ROOT / "ui/server.py")
report_ui = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(report_ui)


class ReportUiTests(unittest.TestCase):
    def test_repository_report_is_indexed_and_rendered(self):
        dashboard = report_ui.Dashboard(ROOT, None, str(ROOT), 60)
        reports = dashboard.reports()
        report = next(x for x in reports if x["id"] == "2026-07-16-autonomous-research-interface")
        source = ROOT / report["markdown_path"]
        rendered = report_ui.markdown(source.read_text(), report["bundle"])
        self.assertIn("<table>", rendered)
        self.assertIn("<figure>", rendered)
        self.assertIn("The technical details", rendered)

    def test_local_read_receipt_and_steering_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            dashboard = report_ui.Dashboard(Path(tmp), None, "/unused", 60)
            receipt = dashboard.acknowledge("report-1", "0123456789abcdef", True)
            self.assertEqual(json.loads(receipt.read_text())["report_id"], "report-1")
            note = dashboard.send_steering("hard-text", "report-1", "Please prioritize the strongest falsification test.")
            self.assertIn("strongest falsification", note.read_text())
            dashboard.acknowledge("report-1", "0123456789abcdef", False)
            self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
