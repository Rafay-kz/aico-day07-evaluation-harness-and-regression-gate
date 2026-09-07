from __future__ import annotations

import hashlib
import re
from pathlib import Path

INGESTION_VERSION = "1.0"
TOKEN_METHOD = "whitespace_separated_words"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_TRAILING_WRAP = "\"')]}>"
_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "ltd.",
        "inc.",
        "co.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "no.",
        "vol.",
        "pp.",
    }
)


def validate_chunk_params(token_size: int, overlap: int) -> None:
    if token_size <= 0:
        raise ValueError(f"token size must be a positive integer, got {token_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= token_size:
        raise ValueError(
            f"overlap must be smaller than token size "
            f"(got overlap={overlap}, tokens={token_size})"
        )


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Return (token, char_start, char_end) for every whitespace-separated token."""
    tokens: list[tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        tokens.append((text[start:i], start, i))
    return tokens


def tokenize(text: str) -> list[str]:
    """Whitespace split used for chunk size. BM25 starts from this, then folds case."""
    return [token for token, _start, _end in tokenize_with_spans(text)]


def chunk_text(
    text: str,
    *,
    source_file: str,
    token_size: int,
    overlap: int,
) -> list[dict]:
    """Chunk `text` into records that reconstruct exactly via character offsets."""
    validate_chunk_params(token_size, overlap)
    tokens = tokenize_with_spans(text)
    if not tokens:
        return []

    headings = _headings(text)
    source_name = Path(source_file).name
    chunks: list[dict] = []
    n = len(tokens)
    start = 0
    while start < n:
        hard_end = min(start + token_size, n)
        end = _end_at_boundary(tokens, start, hard_end)
        chunks.append(
            _record(
                text=text,
                tokens=tokens,
                start=start,
                end=end,
                source_file=source_name,
                headings=headings,
            )
        )
        if end >= n:
            break
        # overlap tokens, but always move forward at least one token
        start = max(start + 1, end - overlap)
    return chunks


def _end_at_boundary(
    tokens: list[tuple[str, int, int]], start: int, hard_end: int
) -> int:
    """Last sentence end in the window, else the word at hard_end. Never past hard_end."""
    for idx in range(hard_end - 1, start - 1, -1):
        if _is_sentence_end_token(tokens[idx][0]):
            return idx + 1
    return hard_end


def _is_sentence_end_token(token: str) -> bool:
    stripped = token.rstrip(_TRAILING_WRAP)
    if not stripped or stripped[-1] not in ".!?":
        return False
    if stripped.lower() in _ABBREVIATIONS:
        return False
    # "1." in markdown headings is a list marker, not a sentence
    if re.fullmatch(r"\d+\.", stripped):
        return False
    return True


def _headings(text: str) -> list[tuple[int, str]]:
    return [(match.start(), match.group(2).strip()) for match in _HEADING_RE.finditer(text)]


def _section_for(headings: list[tuple[int, str]], char_start: int) -> str:
    section = ""
    for offset, title in headings:
        if offset <= char_start:
            section = title
        else:
            break
    return section


def _record(
    *,
    text: str,
    tokens: list[tuple[str, int, int]],
    start: int,
    end: int,
    source_file: str,
    headings: list[tuple[int, str]],
) -> dict:
    char_start = tokens[start][1]
    char_end = tokens[end - 1][2]
    chunk_text_value = text[char_start:char_end]
    return {
        "chunk_id": f"{source_file}:{char_start}:{char_end}",
        "source_file": source_file,
        "char_start": char_start,
        "char_end": char_end,
        "token_count": end - start,
        "content_hash": hashlib.sha256(chunk_text_value.encode("utf-8")).hexdigest(),
        "ingestion_version": INGESTION_VERSION,
        "section": _section_for(headings, char_start),
        "text": chunk_text_value,
    }
