"""
Background research runs for the web UI.

Runs the ResearchController in a daemon thread so the API stays
responsive, and tees the controller's console output into a per-run ring
buffer the UI polls as a live log. One run at a time: SQLite, Chroma and
arXiv rate limits are shared state.
"""

import io
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone

from src.controller.cycle import ResearchController, render_tree
from src.net import RetrievalError
from src.topic.planner import Topic

RUN_ACTIVE_STATES = ("queued", "running")


class _RunLog(io.TextIOBase):
    """Tee stdout writes into a per-run ring buffer (and the real stdout)."""

    def __init__(self, sink, buffer: deque):
        self._sink = sink
        self._buffer = buffer

    def write(self, text: str) -> int:
        self._sink.write(text)
        self._buffer.append(text)
        return len(text)

    def flush(self) -> None:
        self._sink.flush()


class ResearchRun:
    """State of one background run (polled by the UI)."""

    def __init__(self, run_id: str, topic: str, max_cycles: int):
        self.run_id = run_id
        self.topic = topic
        self.max_cycles = max_cycles
        self.status = "queued"      # queued|running|completed|failed
        self.cycles_done = 0
        self.error: str | None = None
        self.log: deque = deque(maxlen=600)
        self.tree = ""
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.finished_at: str | None = None


class RunManager:
    """Starts and tracks research runs; one active run at a time."""

    def __init__(self, on_complete=None):
        self.runs: dict[str, ResearchRun] = {}
        self._lock = threading.Lock()
        self._active_id: str | None = None
        self._on_complete = on_complete  # callback(run), e.g. refresh chat indexes

    def start(self, topic_name: str, max_cycles: int = 3) -> str:
        with self._lock:
            if self._active_id is not None:
                active = self.runs.get(self._active_id)
                if active and active.status in RUN_ACTIVE_STATES:
                    raise RuntimeError(
                        f"research run '{active.run_id}' ({active.topic}) is "
                        "already active; wait for it to finish"
                    )
                self._active_id = None

            run_id = uuid.uuid4().hex[:12]
            run = ResearchRun(run_id, topic_name, max_cycles)
            self.runs[run_id] = run
            self._active_id = run_id

        thread = threading.Thread(
            target=self._execute, args=(run,), daemon=True,
            name=f"research-{run_id}",
        )
        thread.start()
        return run_id

    def _execute(self, run: ResearchRun) -> None:
        original_stdout = sys.stdout
        run.status = "running"
        try:
            sys.stdout = _RunLog(original_stdout, run.log)
            controller = ResearchController()

            root = Topic(name=run.topic, parent=None, cycle=0)
            controller.memory.add_topic(
                root, controller.embedder.embed_texts([root.name])[0]
            )

            topic = root
            for cycle_index in range(run.max_cycles):
                next_topic = controller.run_cycle(topic, cycle_index)
                run.cycles_done = cycle_index + 1
                if next_topic is None:
                    run.log.append("\n[loop] stopping: no novel topic to explore\n")
                    break
                topic = next_topic

            run.tree = render_tree(controller.memory)
            run.status = "completed"
        except RetrievalError as error:
            run.status = "failed"
            run.error = f"retrieval failed — cycles stopped: {error}"
            run.log.append(
                f"\n[error] {run.error}\n"
                "[loop] stopping the run: no search backend reachable, "
                "continuing would analyze zero evidence\n"
            )
        except Exception as error:  # noqa: BLE001 — report, never die silently
            run.status = "failed"
            run.error = f"{type(error).__name__}: {error}"
            run.log.append(f"\n[error] {run.error}\n")
        finally:
            sys.stdout = original_stdout
            run.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if self._on_complete:
                try:
                    self._on_complete(run)
                except Exception:  # noqa: BLE001
                    pass

    def get(self, run_id: str) -> ResearchRun | None:
        return self.runs.get(run_id)

    def all(self) -> list[ResearchRun]:
        return list(self.runs.values())

    @staticmethod
    def snapshot(run: ResearchRun) -> dict:
        return {
            "run_id": run.run_id,
            "topic": run.topic,
            "max_cycles": run.max_cycles,
            "status": run.status,
            "cycles_done": run.cycles_done,
            "error": run.error,
            "log": "".join(run.log)[-8000:],
            "tree": run.tree,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        }