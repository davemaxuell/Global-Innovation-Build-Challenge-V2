"""Deterministic, documented document-quality rules; no model scores are used.

Hard filters follow the published Gopher heuristics (Rae et al., 2021, App. A):
word counts and lengths, symbol ratios, bullet/ellipsis lines, stop-word
presence, alphabetic words, and duplicate line/paragraph/n-gram fractions.
Documents passing every filter receive a coherence score that favors complete
sentences, natural function-word density, prose-like sentence lengths and low
repetition. Model loss never enters selection, so high-loss garbage cannot
masquerade as useful difficulty.
"""
from collections import Counter
import re

STOPWORDS = frozenset("the be to of and that have with".split())
FUNCTION_WORDS = frozenset((
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "this that these those it its he she they we you i his her their our your not no so than "
    "then there which who whom whose what when where why how can could will would should may might "
    "must do does did has have had").split())
BOILERPLATE = ("lorem ipsum", "cookie", "javascript", "all rights reserved", "terms of use", "privacy policy",
               "click here", "sign up", "subscribe", "log in", "add to cart", "powered by")
TERMINAL = re.compile(r"[.!?][\"')\]]*$")
BULLET = re.compile(r"^\s*(?:[-*•●‣]|\d+[.)])\s")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _duplicate_fraction(units, text_length):
    counts = Counter(units)
    duplicated = sum(len(u) * (c - 1) for u, c in counts.items() if c > 1)
    return duplicated / max(1, text_length), sum(c - 1 for c in counts.values()) / max(1, len(units))


def _top_ngram_fraction(words, n):
    if len(words) < n:
        return 0.
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    if count < 2:
        return 0.
    return count * sum(map(len, gram)) / max(1, sum(map(len, words)))


def _duplicate_ngram_fraction(words, n):
    if len(words) < n:
        return 0.
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, gram in enumerate(grams):
        if counts[gram] > 1:
            for j in range(i, i + n):
                covered[j] = True
    return sum(len(w) for w, c in zip(words, covered) if c) / max(1, sum(map(len, words)))


def features(text):
    lines = [line for line in text.split("\n") if line.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    words = text.split()
    lowered = [w.lower().strip(".,;:!?\"'()[]") for w in words]
    n = max(1, len(words))
    sentences = [s for s in SENTENCE.split(" ".join(words)) if s.strip()]
    line_dup_chars, line_dup = _duplicate_fraction(lines, len(text))
    para_dup_chars, para_dup = _duplicate_fraction(paragraphs, len(text))
    return {
        "words": len(words),
        "mean_word_length": sum(map(len, words)) / n,
        "symbol_ratio": (text.count("#") + text.count("...") + text.count("…")) / n,
        "bullet_line_fraction": sum(bool(BULLET.match(l)) for l in lines) / max(1, len(lines)),
        "ellipsis_line_fraction": sum(l.rstrip().endswith(("...", "…")) for l in lines) / max(1, len(lines)),
        "alphabetic_word_fraction": sum(any(c.isalpha() for c in w) for w in words) / n,
        "stopword_types": len(STOPWORDS.intersection(lowered)),
        "function_word_fraction": sum(w in FUNCTION_WORDS for w in lowered) / n,
        "duplicate_line_fraction": line_dup, "duplicate_line_char_fraction": line_dup_chars,
        "duplicate_paragraph_fraction": para_dup, "duplicate_paragraph_char_fraction": para_dup_chars,
        **{f"top_{k}gram_char_fraction": _top_ngram_fraction(lowered, k) for k in (2, 3, 4)},
        **{f"duplicate_{k}gram_char_fraction": _duplicate_ngram_fraction(lowered, k) for k in (5, 10)},
        "boilerplate_line_fraction": sum(any(b in l.lower() for b in BOILERPLATE) for l in lines) / max(1, len(lines)),
        "terminal_line_fraction": sum(bool(TERMINAL.search(l.rstrip())) for l in lines) / max(1, len(lines)),
        "mean_sentence_words": len(words) / max(1, len(sentences)),
        "lines": len(lines),
    }


def hard_failures(f, rules):
    """Return the names of every violated rule; an empty list passes."""
    checks = {
        "words": rules["min_words"] <= f["words"] <= rules["max_words"],
        "mean_word_length": rules["min_mean_word_length"] <= f["mean_word_length"] <= rules["max_mean_word_length"],
        "symbol_ratio": f["symbol_ratio"] <= rules["max_symbol_ratio"],
        "bullet_lines": f["bullet_line_fraction"] <= rules["max_bullet_line_fraction"],
        "ellipsis_lines": f["ellipsis_line_fraction"] <= rules["max_ellipsis_line_fraction"],
        "alphabetic_words": f["alphabetic_word_fraction"] >= rules["min_alphabetic_word_fraction"],
        "stopwords": f["stopword_types"] >= rules["min_stopword_types"],
        "duplicate_lines": f["duplicate_line_fraction"] <= rules["max_duplicate_line_fraction"],
        "duplicate_line_chars": f["duplicate_line_char_fraction"] <= rules["max_duplicate_line_char_fraction"],
        "duplicate_paragraphs": f["duplicate_paragraph_fraction"] <= rules["max_duplicate_paragraph_fraction"],
        "duplicate_paragraph_chars": f["duplicate_paragraph_char_fraction"] <= rules["max_duplicate_paragraph_char_fraction"],
        "boilerplate": f["boilerplate_line_fraction"] <= rules["max_boilerplate_line_fraction"],
    }
    for k, limit in rules["max_top_ngram_char_fraction"].items():
        checks[f"top_{k}gram"] = f[f"top_{k}gram_char_fraction"] <= limit
    for k, limit in rules["max_duplicate_ngram_char_fraction"].items():
        checks[f"duplicate_{k}gram"] = f[f"duplicate_{k}gram_char_fraction"] <= limit
    return sorted(name for name, passed in checks.items() if not passed)


def coherence(f, weights):
    """Bounded [0, 1] score; the formula and weights are part of the frozen config."""
    sentence = f["mean_sentence_words"]
    sentence_score = 1. if 8 <= sentence <= 35 else max(0., 1 - min(abs(sentence - 8), abs(sentence - 35)) / 20)
    parts = {"terminal_lines": f["terminal_line_fraction"],
             "function_words": min(1., f["function_word_fraction"] / 0.35),
             "low_repetition": 1 - f["duplicate_5gram_char_fraction"],
             "sentence_length": sentence_score}
    if set(parts) != set(weights) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Coherence weights must name every component and sum to one")
    return sum(weights[k] * v for k, v in parts.items())
