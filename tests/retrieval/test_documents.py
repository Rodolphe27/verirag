from pathlib import Path

import pytest

from verirag.retrieval.documents import load_chunks, load_folder


def test_markdown_chunks_have_no_page(tmp_path: Path) -> None:
    file = tmp_path / "article.md"
    file.write_text("First sentence. Second sentence.", encoding="utf-8")
    chunks = load_chunks(file)
    assert [c.source for c in chunks] == ["article.md"]
    assert chunks[0].page is None


def test_unsupported_type_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "image.png"
    file.write_bytes(b"")
    with pytest.raises(ValueError, match="unsupported"):
        load_chunks(file)


def test_folder_loads_supported_files_in_name_order(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("Text of b.", encoding="utf-8")
    (tmp_path / "a.md").write_text("Text of a.", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"")
    assert [c.source for c in load_folder(tmp_path)] == ["a.md", "b.txt"]


def minimal_pdf(*page_texts: str) -> bytes:
    """Build a tiny valid PDF by hand, one text line per page (no PDF library needed)."""
    objects: list[bytes] = []
    n = len(page_texts)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(page_texts):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents {content_id} 0 R "
            "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
            "/BaseFont /Helvetica >> >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def test_pdf_chunks_keep_their_page_number(tmp_path: Path) -> None:
    file = tmp_path / "two.pdf"
    file.write_bytes(minimal_pdf("Alpha text on page one.", "Beta text on page two."))
    chunks = load_chunks(file)
    assert [(c.page, c.text) for c in chunks] == [
        (1, "Alpha text on page one."),
        (2, "Beta text on page two."),
    ]
