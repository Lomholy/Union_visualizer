"""Running slow work in a worker process without blocking the Qt event
loop: ProcessJob runs one call, JobRunner owns a process pool and runs one
ProcessJob at a time."""

import time
import traceback
from concurrent.futures import ProcessPoolExecutor

from qtpy import QtCore


class ProcessJob(QtCore.QObject):
    """Runs fn(*args, **kwargs) in a process pool, waiting for it on the
    QThread this object is moved to, then post(result) on that QThread.

    The call runs in a worker *process*, not just a QThread:
    pythonocc-core's OpenCASCADE calls (used by the "brep" mesher) don't
    release the GIL, so a long boolean operation on a plain QThread would
    still starve the GUI thread's event loop and freeze the window.
    Blocking on future.result() is safe because waiting on inter-process
    I/O releases the GIL. post is for cheap follow-up work that must not
    cross the process boundary, such as wrapping the result in pygfx
    objects.

    Emits finished(result, elapsed_seconds) or
    failed(traceback_text, "ExceptionType: message").
    """

    finished = QtCore.Signal(object, float)
    failed = QtCore.Signal(str, str)

    def __init__(self, pool, fn, args, kwargs, post=None):
        super().__init__()
        self.pool = pool
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.post = post

    def run(self):
        start = time.time()
        try:
            result = self.pool.submit(self.fn, *self.args, **self.kwargs).result()
            if self.post is not None:
                result = self.post(result)
        except Exception as e:
            self.failed.emit(traceback.format_exc(), f"{type(e).__name__}: {e}")
            return
        self.finished.emit(result, time.time() - start)


class JobRunner(QtCore.QObject):
    """A single-worker process pool that runs at most one ProcessJob at a
    time, each on its own QThread.

    on_finished(result, elapsed) or on_failed(traceback_text, summary) is
    called on the GUI thread when a job ends, then on_idle() and any
    callback queued with run_when_idle() once its thread has stopped.
    """

    def __init__(self, parent, on_finished, on_failed, on_idle=None):
        super().__init__(parent)
        self.pool = ProcessPoolExecutor(max_workers=1)
        self.on_finished = on_finished
        self.on_failed = on_failed
        self.on_idle = on_idle
        self._thread = None
        self._job = None
        self._when_idle = None

    @property
    def busy(self):
        return self._thread is not None

    def start(self, fn, *args, post=None, **kwargs):
        """Run fn(*args, **kwargs) in the pool, then post(result) on the
        job's QThread. Must not be called while busy."""
        if self.busy:
            raise RuntimeError("JobRunner.start() called while a job is running")
        thread = QtCore.QThread(self)
        job = ProcessJob(self.pool, fn, args, kwargs, post)
        job.moveToThread(thread)
        thread.started.connect(job.run)
        job.finished.connect(self._on_job_finished)
        job.failed.connect(self._on_job_failed)
        for signal in (job.finished, job.failed):
            signal.connect(thread.quit)
            signal.connect(job.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._job = job
        thread.start()

    def run_when_idle(self, callback):
        """Call callback once the running job's thread has stopped. Only
        the latest callback queued before then is kept."""
        self._when_idle = callback

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    @QtCore.Slot(object, float)
    def _on_job_finished(self, result, elapsed):
        self.on_finished(result, elapsed)

    @QtCore.Slot(str, str)
    def _on_job_failed(self, traceback_text, summary):
        self.on_failed(traceback_text, summary)

    @QtCore.Slot()
    def _on_thread_finished(self):
        self._thread = None
        self._job = None
        if self.on_idle is not None:
            self.on_idle()
        callback, self._when_idle = self._when_idle, None
        if callback is not None:
            callback()
