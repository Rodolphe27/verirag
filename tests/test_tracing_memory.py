from verirag.ports import Tracer
from verirag.tracing import MemoryTracer


def test_records_are_kept_in_order() -> None:
    tracer = MemoryTracer()
    tracer.record("router", {"route": "documents"})
    tracer.record("writer", {"attempt": 1})
    assert tracer.steps() == ["router", "writer"]


def test_later_changes_to_the_data_do_not_alter_the_log() -> None:
    tracer = MemoryTracer()
    data: dict[str, object] = {"n": 1}
    tracer.record("step", data)
    data["n"] = 2
    assert tracer.records[0][1] == {"n": 1}


def test_memory_tracer_fits_the_port() -> None:
    tracer: Tracer = MemoryTracer()
    tracer.record("x", {})
