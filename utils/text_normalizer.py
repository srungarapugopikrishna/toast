import re


def normalize_sentence_text(
    text: str,
    filler_words,
    min_meaningful_words: int = 3,
    mode: str = "balanced",
) -> str:
    """
    Context-aware, configurable text cleanup.

    Modes:
      - conservative
      - balanced (recommended)
      - aggressive
    """

    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)

    # --------------------------------------------------
    # 1. Always-safe removals
    # --------------------------------------------------

    # Elongated vocal fillers: mmm, ahh, uhhh, ermm
    t = re.sub(r"\b(m+|a+h+|u+h+|e+r+m+)\b", "", t)

    # Multi-word discourse fillers
    t = re.sub(r"\byou know\b", "", t)
    t = re.sub(r"\bkind of\b", "", t)
    t = re.sub(r"\bsort of\b", "", t)

    # --------------------------------------------------
    # 2. Mode-based filler handling
    # --------------------------------------------------

    if mode in ("balanced", "aggressive"):
        # Sentence-start fillers
        t = re.sub(
            r"^(actually|basically|literally|so)\b[\s,]*",
            "",
            t,
            flags=re.IGNORECASE,
        )

        # Conjunction + filler
        t = re.sub(
            r"\b(and|so|but)\s+(actually|basically|literally)\b",
            r"\1",
            t,
            flags=re.IGNORECASE,
        )

        # Comma-surrounded fillers
        t = re.sub(
            r",\s*(actually|basically|literally)\s*,",
            ",",
            t,
            flags=re.IGNORECASE,
        )

    if mode == "aggressive":
        # Remove fillers everywhere (semantic risk)
        for f in filler_words:
            t = re.sub(
                r"\b" + re.escape(f) + r"\b",
                "",
                t,
                flags=re.IGNORECASE,
            )

    # --------------------------------------------------
    # 3. Remove dangling filler artifacts (IMPORTANT FIX)
    # --------------------------------------------------

    # Remove standalone 'like' when it becomes meaningless
    t = re.sub(r"\blike\b\s*,*", "", t)

    # Remove trailing conjunctions after filler removal
    t = re.sub(r"\b(and|so|but)\b\s*$", "", t)

    # Collapse multiple commas created by removals
    t = re.sub(r",\s*,+", ",", t)

    # Remove trailing commas
    t = re.sub(r",\s*$", "", t)

    # --------------------------------------------------
    # 4. Cleanup punctuation & spacing
    # --------------------------------------------------

    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r"\s+\.", ".", t)
    t = re.sub(r"\s{2,}", " ", t).strip()

    # Restore capitalization
    if t:
        t = t[0].upper() + t[1:]

    # Drop sentences with no semantic value
    words = re.findall(r"\b\w+\b", t)
    if len(words) < min_meaningful_words:
        return ""

    # Ensure proper ending
    if not t.endswith((".", "!", "?")):
        t += "."

    return t
