"""Patient-level wall-clock timing, with CUDA completion at stage boundaries."""
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

CV_STAGES = (
    "dicom_identification", "sagittal_slice_selection", "sagittal_roi_detection",
    "sagittal_roi_classification", "sagittal_segmentation", "sagittal_measurements",
    "axial_slice_selection", "axial_roi_detection", "axial_roi_estimation",
    "axial_segmentation", "axial_measurements", "axial_roi_classification",
)


def synchronize_cuda():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


class StageTiming:
    def __init__(self, run_id, filename=None):
        self.run_id = run_id
        self.filename = filename
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = perf_counter()
        self.stages = {name: {"status": "not_run", "seconds": None} for name in CV_STAGES}
        self.overhead = {}

    @contextmanager
    def measure(self, name, overhead=False):
        target = self.overhead if overhead else self.stages
        record = {"status": "failed", "seconds": None}
        target[name] = record
        synchronize_cuda()
        started = perf_counter()
        try:
            yield
            synchronize_cuda()
            record["status"] = "success"
        finally:
            record["seconds"] = perf_counter() - started
            print(f"[STAGE] {name}: {record['seconds']:.4f} s ({record['status']})", flush=True)

    def mark_partial(self, name, reason):
        self.stages[name]["status"] = "partial"
        self.stages[name]["reason"] = reason

    def call(self, name, function, *args, **kwargs):
        with self.measure(name):
            return function(*args, **kwargs)

    def check_result(self, name, value):
        if isinstance(value, dict):
            if value.get("error") or value.get("status") in ("failed", "error"):
                self.mark_partial(name, "Tool returned an error in its results.")
            for item in value.values():
                if isinstance(item, (dict, list, tuple)):
                    self.check_result(name, item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.check_result(name, item)

    def snapshot(self, status):
        complete = all(item["status"] == "success" for item in self.stages.values())
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "input_zip_filename": self.filename,
            "started_at_utc": self.started_at,
            "status": status if status != "success" or complete else "partial",
            "units": "seconds",
            "scope": "Complete stage calls including preprocessing, model loading when performed, inference, postprocessing and image saving; excludes report generation.",
            "stages": self.stages,
            "overhead": self.overhead,
            "cv_total_seconds": sum(item["seconds"] or 0 for item in self.stages.values()),
            "processing_seconds_before_timing_export": perf_counter() - self.started,
            "processing_scope": "Server processing from run_zip entry; excludes upload, response transfer and final timing/results export.",
        }

    def save(self, output_root, benchmark_root, status):
        record = self.snapshot(status)
        output_root = Path(output_root)
        benchmark_root = Path(benchmark_root)
        output_root.mkdir(parents=True, exist_ok=True)
        benchmark_root.mkdir(parents=True, exist_ok=True)
        (output_root / "timings.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        with (benchmark_root / "stage_timings.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return record
