from verirag.tracing import CompositeTracer, MemoryTracer


def test_every_tracer_gets_every_record() -> None:
    first, second = MemoryTracer(), MemoryTracer()
    both = CompositeTracer(first, second)
    both.record("a", {"n": 1})
    both.record("b", {})
    assert first.steps() == second.steps() == ["a", "b"]
