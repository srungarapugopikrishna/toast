# utils/repetition.py

from typing import List, Tuple, Dict


def _norm(w: str) -> str:
    return w.strip().lower()


def _avg_word_duration(words):
    if not words:
        return 0.2
    return sum((end - start) for _, start, end, _ in words) / len(words)


def score_segment(seg_words, prev_end, next_start, avg_dur):
    """
    Higher score = better flow / pronunciation
    """
    if not seg_words:
        return -1.0

    start = seg_words[0][1]
    end = seg_words[-1][2]
    dur = end - start
    expected = avg_dur * len(seg_words)

    score = 0.0

    # 1️⃣ Duration closeness (not rushed)
    if expected > 0:
        score += max(0.0, 1.0 - abs(dur - expected) / expected)

    # 2️⃣ Natural pause before
    if prev_end is not None:
        pre_gap = start - prev_end
        if 0.04 <= pre_gap <= 0.30:
            score += 0.5

    # 3️⃣ Natural pause after
    if next_start is not None:
        post_gap = next_start - end
        if 0.04 <= post_gap <= 0.30:
            score += 0.5

    return score


def detect_repetition_segments(
    words: List[Tuple[str, float, float, float]],
    max_ngram: int = 4,
):
    """
    Detects continuous repeated word sequences and returns
    silence-like cut segments: [[start, end], ...]
    """

    cuts = []
    i = 0
    n = len(words)
    avg_dur = _avg_word_duration(words)

    while i < n:
        matched = False

        for k in range(max_ngram, 0, -1):
            if i + 2 * k > n:
                continue

            seq1 = [_norm(words[j][0]) for j in range(i, i + k)]
            seq2 = [_norm(words[j][0]) for j in range(i + k, i + 2 * k)]

            if seq1 != seq2:
                continue

            seg1 = words[i : i + k]
            seg2 = words[i + k : i + 2 * k]

            prev1 = words[i - 1][2] if i > 0 else None
            next1 = seg2[0][1]

            prev2 = seg1[-1][2]
            next2 = words[i + 2 * k][1] if i + 2 * k < n else None

            s1 = score_segment(seg1, prev1, next1, avg_dur)
            s2 = score_segment(seg2, prev2, next2, avg_dur)

            if s1 >= s2:
                # remove second
                cuts.append([seg2[0][1], seg2[-1][2]])
            else:
                # remove first
                cuts.append([seg1[0][1], seg1[-1][2]])

            i += 2 * k
            matched = True
            break

        if not matched:
            i += 1

    print("🔁 Detected repetition cut segments:", cuts)
    return cuts
