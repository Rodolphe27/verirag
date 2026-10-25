# Using VeriRAG as an MCP server

MCP (Model Context Protocol) is the standard way for an AI application to find and call tools.
`verirag serve` makes your indexed documents available to any MCP client.

## Tools

| Tool | Needs a model? | What it does |
|---|---|---|
| `list_documents` | no | Lists the indexed source files. |
| `search_documents` | no | Returns the best matching passages (source, page, score, text). `limit` is kept between 1 and 10. |
| `ask_verified` | yes (`--verified`) | Answers a question; every claim is checked against the sources. Returns `status` (`answered` or `refused`), `text` and `citations`. A refusal is a normal result, not an error. |

## Start the server

```bash
verirag index docs/sample                # once: build the index
verirag serve                            # search tools only
verirag serve --verified                 # adds ask_verified (needs GEMINI_API_KEY or ANTHROPIC_API_KEY)
```

The server talks over standard input/output (the `stdio` transport), so it prints nothing else.

## Connect a client

Most desktop MCP clients read a JSON file with a list of servers. Add an entry like this
(use the full path to your index, and to `verirag` if it is not on the PATH):

```json
{
  "mcpServers": {
    "verirag": {
      "command": "verirag",
      "args": ["--index", "/full/path/to/.verirag/index.json", "serve"]
    }
  }
}
```

To pass an API key for `--verified`, add `"env": { "GEMINI_API_KEY": "..." }` to the entry.

## Test it from Python

`tests/test_mcp_stdio.py` starts the real server as a separate process and talks to it with
the official MCP client. It is the shortest working example of a client.
