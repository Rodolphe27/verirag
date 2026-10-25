# ADR 1: A modular monolith with ports and adapters

Status: accepted

## Context

VeriRAG has parts that change for different reasons: the language model (providers come and go),
the search (a file today, a vector database tomorrow), the way people use it (command line, MCP,
maybe a web page), and the agent logic itself. It also has parts that are slow, costly or random:
every model call is. The agent logic must be testable without calling a model.

## Decision

Build **one program** divided into modules with strict borders (a *modular monolith*), and shape
the modules that touch the outside world as *ports and adapters* (hexagonal architecture):

- The core (`domain`, `ports`, `agents`) describes what it needs as small interfaces
  (`typing.Protocol`): `LLM`, `Retriever`, `Tracer`.
- Adapters (`llm`, `retrieval`, `tracing`) implement them.
- One place, `cli.py`, creates the real objects and connects them (the *composition root*).
- The directions are enforced by `import-linter` contracts in CI.

## Consequences

- Tests replace the model with a scripted fake, so the whole graph is tested in milliseconds,
  with no network and no key.
- Switching from Gemini to Anthropic is one setting (`VERIRAG_LLM`); a new provider is one class
  and one line.
- There is a little more structure than a script would need: ports, a composition root, and
  contracts. It pays for itself when the first adapter is swapped or the first test runs without
  a model.
- Everything runs in one process and starts with one command. If the search ever needs to scale
  alone, `retrieval` already has clean borders and could be moved behind a network call.

## Alternatives considered

- **A single script or a flat package.** Fastest to write, but model calls would be tangled with
  the logic, and the tests would need a real model.
- **Full Clean Architecture** (four concentric rings, use-case classes, presenters). More ceremony
  than this size justifies. Ports and adapters give the part of it that matters here.
- **Microservices** (search, agents and interface as separate services). Adds network failures,
  deployment work and latency, with no benefit for one developer and one process.
