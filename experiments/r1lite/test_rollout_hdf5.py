"""Storage and checkpoint tests that do not need ROS or a robot."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_hdf5 as storage


def jpeg() -> np.ndarray:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (20, 40, 60)).save(stream, format="JPEG")
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


class RolloutHDF5Test(unittest.TestCase):
    def _demo(self, recorder: storage.HDF5RolloutRecorder, demo_id: str) -> storage.Checkpoint:
        state = np.arange(14, dtype=np.float32)
        recorder.begin_demo(demo_id, "stack the bowl", 15.0, 100.0)
        self.assertTrue(recorder.submit_frame(
            100.0, state,
            {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in storage.CAMERAS},
            {name: 100.0 for name in storage.SAMPLE_NAMES}, state + 1, True,
        ))
        recorder.submit_step(demo_id, {
            "model_action": [0.1] * 14,
            "commanded_target": (state + 1).tolist(),
            "measured_before": state.tolist(),
            "measured_after": state.tolist(),
            "chunk_id": 1, "action_step": 1,
        })
        recorder.stop_frames()
        recorder.submit_event(demo_id, {"event": "demo_stopped"})
        return recorder.seal_demo(demo_id, 101.0)

    def test_two_demos_scores_and_viewer_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(storage, "_jpeg", side_effect=lambda _: jpeg()):
            root = Path(directory)
            recorder = storage.HDF5RolloutRecorder(root, {"prompt": "stack"}, 2)
            for demo_id, score in (("demo_1", 100), ("demo_2", 75)):
                self._demo(recorder, demo_id).wait()
                with h5py.File(root / "run.hdf5", "r") as file:
                    self.assertEqual(file[f"data/{demo_id}"].attrs["status"], "stopped_unscored")
                    self.assertEqual(len(file[f"data/{demo_id}/rollout/events_json"]), 1)
                recorder.submit_post_event(demo_id, {"event": "reset_finished"})
                recorder.finalize_demo(demo_id, score, True).wait()
            recorder.close(completed=True)
            self.assertFalse((root / "run.inprogress").exists())
            self.assertFalse((root / "run.checkpoint.tmp").exists())
            with h5py.File(root / "run.hdf5", "r") as file:
                self.assertEqual(sorted(file["data"].keys()), ["demo_1", "demo_2"])
                self.assertEqual(int(file.attrs["requested_demos"]), 2)
                self.assertEqual(int(file.attrs["scored_demos"]), 2)
                self.assertEqual(int(file.attrs["success_count"]), 1)
                self.assertEqual(float(file.attrs["success_rate"]), 0.5)
                for demo in file["data"].values():
                    self.assertEqual(demo["joint_states"].shape, (1, 14))
                    self.assertEqual(demo["absolute_actions"].shape, (1, 14))
                    self.assertEqual(demo["actions"].shape, (1, 14))
                    np.testing.assert_array_equal(demo["actions"][0], np.ones(14))
                    self.assertEqual(len(demo["obs/agentview_rgb_jpeg"]), 1)
                    self.assertEqual(len(demo["rollout/steps/model_action"]), 1)
                    self.assertEqual(len(demo["rollout/events_json"]), 2)
                self.assertTrue(bool(file["data/demo_1"].attrs["success"]))
                self.assertFalse(bool(file["data/demo_2"].attrs["success"]))

            viewer_root = os.environ.get("FASTWAM_HDF5_VIEWER_PATH")
            if viewer_root:
                sys.path.insert(0, str(Path(viewer_root).parent))
                from hdf5_visualizer import build

                manifest = build.build(root, root / "site", jobs=1)
                self.assertEqual(len(manifest["demos"]), 2)
                self.assertTrue((root / "site/manifest.json").exists())
                self.assertTrue((root / "site/media/run/demo_1/agentview.mp4").exists())

    def test_failed_copy_preserves_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(storage, "_jpeg", side_effect=lambda _: jpeg()):
            root = Path(directory)
            recorder = storage.HDF5RolloutRecorder(root, {}, 1)
            self._demo(recorder, "demo_1").wait()
            previous = (root / "run.hdf5").read_bytes()
            recorder.submit_post_event("demo_1", {"event": "reset_finished"})
            def interrupted_copy(_source: Path, destination: Path) -> None:
                destination.write_bytes(b"partial checkpoint")
                raise OSError("disk full")

            with (
                mock.patch.object(storage.shutil, "copyfile", side_effect=interrupted_copy),
                self.assertRaisesRegex(RuntimeError, "checkpoint failed"),
            ):
                recorder.finalize_demo("demo_1", 100, True).wait()
            self.assertEqual((root / "run.hdf5").read_bytes(), previous)
            with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
                recorder.close()
            self.assertTrue((root / "run.inprogress").exists())

    def test_checkpoint_does_not_block_reset_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(storage, "_jpeg", side_effect=lambda _: jpeg()):
            root = Path(directory)
            recorder = storage.HDF5RolloutRecorder(root, {}, 1)
            copy_started = threading.Event()
            allow_copy = threading.Event()
            real_copy = storage.shutil.copyfile

            def slow_copy(source: Path, destination: Path) -> None:
                copy_started.set()
                self.assertTrue(allow_copy.wait(timeout=3))
                real_copy(source, destination)

            try:
                with mock.patch.object(storage.shutil, "copyfile", side_effect=slow_copy):
                    checkpoint = self._demo(recorder, "demo_1")
                    self.assertTrue(copy_started.wait(timeout=3))
                    self.assertFalse(checkpoint.done.is_set())
                    # Reset can finish while the file copy is still in flight.
                    recorder.submit_post_event("demo_1", {"event": "reset_finished"})
                    allow_copy.set()
                    checkpoint.wait()
                recorder.finalize_demo("demo_1", 100, True).wait()
            finally:
                allow_copy.set()
                recorder.close()

    def test_writer_error_is_reported_without_replacing_good_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = storage.HDF5RolloutRecorder(root, {}, 2)
            with mock.patch.object(storage, "_jpeg", side_effect=lambda _: jpeg()):
                self._demo(recorder, "demo_1").wait()
                recorder.finalize_demo("demo_1", 100, True).wait()
            previous = (root / "run.hdf5").read_bytes()
            with mock.patch.object(storage, "_jpeg", side_effect=RuntimeError("encoder failed")):
                recorder.begin_demo("demo_2", "stack", 15.0, 200.0)
                recorder.submit_frame(
                    200.0, np.zeros(14, dtype=np.float32),
                    {name: np.zeros((8, 8, 3), dtype=np.uint8) for name in storage.CAMERAS},
                    {name: 200.0 for name in storage.SAMPLE_NAMES},
                    np.zeros(14, dtype=np.float32), False,
                )
                recorder.stop_frames()
                with self.assertRaisesRegex(RuntimeError, "writer failed|checkpoint failed"):
                    recorder.seal_demo("demo_2", 201.0).wait()
            self.assertEqual((root / "run.hdf5").read_bytes(), previous)
            with self.assertRaisesRegex(RuntimeError, "writer failed"):
                recorder.close()


if __name__ == "__main__":
    unittest.main()
