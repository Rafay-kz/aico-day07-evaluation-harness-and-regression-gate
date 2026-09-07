from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return a deterministic, bounded normalization of ``text``.

    1. Unicode NFKC so visually-equivalent characters collapse.
    2. Consecutive single-letter tokens are joined (``I G N O R E`` → ``IGNORE``).
    3. Remaining whitespace is collapsed to a single space.

    Isolated single-letter words such as ``a`` are left in place so benign
    questions are not rewritten.
    """
    nfkc = unicodedata.normalize("NFKC", text)
    tokens = nfkc.split()
    if not tokens:
        return ""

    rebuilt: list[str] = []
    letter_run: list[str] = []

    def _flush_run() -> None:
        if not letter_run:
            return
        if len(letter_run) >= 2:
            rebuilt.append("".join(letter_run))
        else:
            rebuilt.append(letter_run[0])
        letter_run.clear()

    for token in tokens:
        if len(token) == 1 and token.isalpha():
            letter_run.append(token)
            continue
        _flush_run()
        rebuilt.append(token)
    _flush_run()

    collapsed = " ".join(rebuilt)
    return _WHITESPACE.sub(" ", collapsed).strip()
