import json
import ast
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

from stage_timing import CV_STAGES, StageTiming
from summarize_timings import summarize


class StageTimingTests(unittest.TestCase):
    def test_zip_entrypoint_exports_success_and_failure(self):
        # Exercise the real orchestration without loading GPU models or patient data.
        source = Path(__file__).with_name("lss_pipeline.py").read_text(encoding="utf-8")
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "run_zip")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def pipeline(**kwargs):
                timing = kwargs["timing"]
                for name in CV_STAGES:
                    timing.call(name, lambda: None)
                return {"status": "success"}
            namespace = {
                "Path": Path, "Any": object, "StageTiming": StageTiming,
                "PROJECT_ROOT": root, "OUTPUT_ROOT": root / "output",
                "_safe_run_id": lambda filename: "patient",
                "clear_previous_run_data": lambda: None,
                "extract_dicom_zip": lambda *args: root / "input",
                "run_lss_pipeline": pipeline,
                "clean_for_json": lambda result, url: result,
                "_strip_roi_detection_urls": lambda result: None,
                "_asset": lambda path, url: {"path": str(path)},
                "traceback": traceback,
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), "lss_pipeline.py", "exec"), namespace)
            result = namespace["run_zip"](root / "patient.zip", "patient.zip")
            saved = json.loads((root / "output/results.json").read_text())
            self.assertEqual(saved["timings"], result["timings"])
            self.assertEqual(result["timings"]["status"], "success")
            def failed_pipeline(**kwargs):
                with kwargs["timing"].measure("dicom_identification"):
                    raise RuntimeError("missing dicoms")
            namespace["run_lss_pipeline"] = failed_pipeline
            result = namespace["run_zip"](root / "bad.zip", "bad.zip")
            self.assertEqual(result["timings"]["status"], "failed")
            saved = json.loads((root / "output/results.json").read_text())
            self.assertIsNone(saved["timings"]["stages"]["axial_segmentation"]["seconds"])
            self.assertEqual(len((root / "benchmark/stage_timings.jsonl").read_text().splitlines()), 2)

    def test_duration_sync_and_function_return(self):
        with patch("stage_timing.perf_counter", side_effect=[0, 10, 13]), patch("stage_timing.synchronize_cuda") as sync:
            timing = StageTiming("study")
            self.assertEqual(timing.call("axial_segmentation", lambda: list(range(5))), list(range(5)))
        self.assertEqual(timing.stages["axial_segmentation"], {"status": "success", "seconds": 3})
        self.assertEqual(sync.call_count, 2)

    def test_failure_and_not_run(self):
        timing = StageTiming("failed")
        with self.assertRaises(ValueError), timing.measure("axial_segmentation"):
            raise ValueError("model failed")
        self.assertEqual(timing.stages["axial_segmentation"]["status"], "failed")
        self.assertIsNone(timing.stages["axial_measurements"]["seconds"])

    def test_persistence_partial_and_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                timing = StageTiming(str(index))
                for name in CV_STAGES:
                    timing.stages[name] = {"status": "success", "seconds": index + 1}
                timing.overhead["zip_extraction_and_dicom_routing"] = {"status": "success", "seconds": 100}
                if index == 2:
                    timing.check_result("axial_segmentation", {"results": [{"error": "missing slice"}]})
                record = timing.save(root / "output", root / "benchmark", "success")
                self.assertEqual(record["cv_total_seconds"], len(CV_STAGES) * (index + 1))
            rows = [json.loads(line) for line in (root / "benchmark/stage_timings.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(json.loads((root / "output/timings.json").read_text())["status"], "partial")
            report = summarize(rows)
            self.assertEqual(report["complete_runs_used"], 2)
            self.assertEqual(report["stages"]["axial_segmentation"]["mean_seconds"], 1.5)
            self.assertEqual(summarize(rows, 1)["stages"]["axial_segmentation"]["mean_seconds"], 2)
            self.assertIsNone(summarize([])["cv_total"]["mean_seconds"])


if __name__ == "__main__":
    unittest.main()
