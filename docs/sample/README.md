# Sample corpus

Fourteen articles of the **EU Artificial Intelligence Act**, Regulation (EU) 2024/1689,
used for the demo and for the evaluation. They are public legal text with no personal data.

| File | Topic |
|---|---|
| `ai-act-article-003.md` | Definitions (AI system, provider, deployer, ...) |
| `ai-act-article-004.md` | AI literacy |
| `ai-act-article-005.md` | Prohibited AI practices |
| `ai-act-article-006.md` | Classification rules for high-risk AI systems |
| `ai-act-article-009.md` | Risk management system |
| `ai-act-article-010.md` | Data and data governance |
| `ai-act-article-012.md` | Record-keeping (logs) |
| `ai-act-article-013.md` | Transparency towards deployers |
| `ai-act-article-014.md` | Human oversight |
| `ai-act-article-015.md` | Accuracy, robustness and cybersecurity |
| `ai-act-article-026.md` | Obligations of deployers of high-risk systems |
| `ai-act-article-050.md` | Transparency obligations (chatbots, deepfakes) |
| `ai-act-article-053.md` | Obligations of providers of general-purpose AI models |
| `ai-act-article-099.md` | Penalties |

## Source and reuse

The text comes from the Official Journal of the European Union, served by the EU
Publications Office. The reuse of EU legislation is allowed with acknowledgement of the
source (Commission Decision 2011/833/EU). Each file ends with its source line.

Only the Official Journal version is authentic. These files are a convenience copy for
testing a retrieval system and are not legal advice.

## Rebuild

```bash
python scripts/build_sample_corpus.py docs/sample
```
