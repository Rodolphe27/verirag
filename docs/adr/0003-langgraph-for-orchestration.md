# ADR 3: LangGraph for orchestration, kept thin

Status: accepted

## Context

The agents form a small state machine: search, route, write, verify, and a loop back to the
writer with feedback, with limits so it always ends. This could be an `if`/`while` in plain Python
or a graph in an orchestration library.

## Decision

Use **LangGraph** to describe the flow as a graph of nodes and conditional edges, but keep it thin:

- The agents (`Writer`, `Verifier`, `Router`) are plain classes that know nothing about LangGraph.
- Only `agents/graph.py` imports it. An import-linter contract keeps it out of the other modules.
- The routing decisions (`after_route`, `after_write`, `after_verify`) are small named functions
  that tests can read, and each path through the graph has its own test.

## Consequences

- The flow is visible: nodes and edges map one to one onto the diagram in the architecture page.
- The framework gives a state container and a built-in recursion limit as a backstop. The real
  stop is explicit: the writer runs at most `max_retries + 1` times.
- A new dependency, and an API that moves. Because only one file depends on it, replacing it with
  a plain loop would touch that file and nothing else.

## Alternatives considered

- **Plain Python loop.** Simplest, and enough today. A graph makes the flow easier to extend
  (new nodes such as a human-approval step) and easier to explain to people who know the library.
- **Multi-agent frameworks with built-in conversation between agents** (for example CrewAI or
  AutoGen). They add autonomy this project does not want: here the order of steps and the stop
  conditions are fixed on purpose, so behaviour is predictable and testable.
