from langchain_core.tracers import BaseTracer, Run
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Define your custom Tracer
from langchain_core.tracers import BaseTracer, Run
from typing import Any, Optional


class InMemoryTracer(BaseTracer):
    def __init__(self):
        super().__init__()
        self.runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Called automatically whenever a top-level run completes."""
        self.runs.append(run)

    def _to_readable(self, run: Run) -> dict:
        duration = None
        if run.end_time and run.start_time:
            duration = (run.end_time - run.start_time).total_seconds()

        return {
            "name": run.name,
            "run_type": run.run_type,
            "duration_s": duration,
            "inputs": run.inputs,
            "outputs": run.outputs,
            "error": run.error,
            "tags": run.tags,
            "metadata": run.extra.get("metadata") if run.extra else None,
            "children": [self._to_readable(child) for child in run.child_runs],
        }

    def get_readable_runs(self) -> list[dict]:
        """Return all top-level runs as a readable, nested dict/object tree."""
        return [self._to_readable(run) for run in self.runs]
        

