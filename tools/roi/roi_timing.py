"""Per-operation ROI latency, excluding model loading and output saving."""
from contextlib import contextmanager
from time import perf_counter


class ROITiming:
    def __init__(self, plane):
        self.plane = plane
        self.samples = {}

    @contextmanager
    def measure(self, operation, device="cpu", frame=None):
        import torch
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        yield
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        self.samples.setdefault(operation, []).append(elapsed)
        fps = 1 / elapsed if elapsed > 0 else 0.0
        label = f" | frame={frame}" if frame is not None else ""
        print(f"[ROI timing] {self.plane} {operation}{label}: "
              f"{elapsed * 1000:.3f} ms/frame | {fps:.2f} FPS", flush=True)

    def report(self, operations):
        for operation in operations:
            values = self.samples.get(operation, [])
            if not values:
                print(f"[ROI timing average] {self.plane} {operation}: "
                      "not executed | FPS=N/A", flush=True)
                continue
            total = sum(values)
            fps = len(values) / total if total > 0 else 0.0
            print(f"[ROI timing average] {self.plane} {operation}: "
                  f"{len(values)} frames | {total * 1000 / len(values):.3f} ms/frame | "
                  f"{fps:.2f} FPS", flush=True)
