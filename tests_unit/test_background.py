"""Tests for background.JobRunner: a job's result or error reaches the
callbacks on the calling thread, followed by on_idle and any callback
queued with run_when_idle. Needs a Qt event loop but no display: a
QApplication on the offscreen platform, so widget tests in the same run
can share it."""

import math
import os
import sys
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qtpy import QtWidgets  # noqa: E402

from background import JobRunner  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until(predicate, timeout=60):
    start = time.time()
    while not predicate():
        app.processEvents()
        time.sleep(0.01)
        if time.time() - start > timeout:
            raise TimeoutError
    app.processEvents()


class JobRunnerTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.runner = JobRunner(
            None,
            on_finished=lambda result, elapsed: self.events.append(
                ("finished", result, threading.current_thread() is threading.main_thread())
            ),
            on_failed=lambda tb, summary: self.events.append(("failed", summary, "Traceback" in tb)),
            on_idle=lambda: self.events.append(("idle",)),
        )

    def tearDown(self):
        self.runner.shutdown()

    def test_result_then_idle_on_the_main_thread(self):
        self.runner.start(math.sqrt, 16.0)
        self.assertTrue(self.runner.busy)
        wait_until(lambda: not self.runner.busy)
        self.assertEqual(self.events, [("finished", 4.0, True), ("idle",)])

    def test_post_runs_on_the_result(self):
        self.runner.start(math.sqrt, 16.0, post=lambda r: r + 1)
        wait_until(lambda: not self.runner.busy)
        self.assertEqual(self.events[0][1], 5.0)

    def test_exception_is_reported_as_failure(self):
        self.runner.start(math.sqrt, -1.0)
        wait_until(lambda: not self.runner.busy)
        (kind, summary, has_traceback), idle = self.events
        self.assertEqual((kind, has_traceback, idle), ("failed", True, ("idle",)))
        self.assertTrue(summary.startswith("ValueError: "), summary)

    def test_only_the_latest_queued_callback_runs_after_idle(self):
        self.runner.start(math.sqrt, 4.0)
        self.runner.run_when_idle(lambda: self.events.append(("first",)))
        self.runner.run_when_idle(lambda: self.events.append(("second",)))
        wait_until(lambda: not self.runner.busy and len(self.events) >= 3)
        self.assertEqual(self.events, [("finished", 2.0, True), ("idle",), ("second",)])

    def test_start_while_busy_raises(self):
        self.runner.start(math.sqrt, 4.0)
        with self.assertRaises(RuntimeError):
            self.runner.start(math.sqrt, 9.0)
        wait_until(lambda: not self.runner.busy)


if __name__ == "__main__":
    unittest.main()
