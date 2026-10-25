"""Download selected articles of the EU AI Act and save each as a Markdown file.

Source: Regulation (EU) 2024/1689, Official Journal of the European Union,
served by the EU Publications Office. The text of EU legislation may be reused
with acknowledgement of the source (Commission Decision 2011/833/EU).

Usage: python scripts/build_sample_corpus.py [output_folder]
Only the standard library is used, so the script runs anywhere.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

URL = "https://publications.europa.eu/resource/celex/32024R1689"
XHTML = "{http://www.w3.org/1999/xhtml}"
# Articles chosen for the topics people ask about: definitions, banned practices,
# high-risk rules, transparency, general-purpose models and fines.
ARTICLES = [3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 26, 50, 53, 99]


def download() -> ET.Element:
    request = urllib.request.Request(
        URL, headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"}
    )
    with urllib.request.urlopen(
        request, timeout=120
    ) as response:  # the URL is a fixed https constant
        return ET.fromstring(response.read())


def lines(element: ET.Element) -> list[str]:
    """The text of an element as separate lines: one per paragraph or table row."""
    if element.tag == f"{XHTML}p":
        return [" ".join("".join(element.itertext()).split())]
    if element.tag == f"{XHTML}tr":
        return [" ".join("".join(element.itertext()).split())]
    result: list[str] = []
    for child in element:
        result.extend(lines(child))
    return result


def main(folder: Path) -> None:
    root = download()
    blocks = {
        e.get("id"): e for e in root.iter(f"{XHTML}div") if (e.get("id") or "").startswith("art_")
    }
    folder.mkdir(parents=True, exist_ok=True)
    for number in ARTICLES:
        body = [line for line in lines(blocks[f"art_{number}"]) if line]
        title = body[0]  # the first line reads "Article 5"
        text = "\n\n".join(
            [
                f"# EU AI Act, {title}",
                *body[1:],
                "---",
                "Source: Regulation (EU) 2024/1689 (Artificial Intelligence Act), "
                f"Official Journal of the European Union, article {number}. "
                f"Retrieved from {URL}. Reuse with acknowledgement of the source.",
            ]
        )
        path = folder / f"ai-act-article-{number:03d}.md"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"{path.name}: {len(text)} characters")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/sample"))
