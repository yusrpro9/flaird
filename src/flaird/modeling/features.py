"""Deterministic forensic-linguistic features used by FLAIRD."""

import math
import re
from dataclasses import dataclass

import numpy as np
from pystylometry.character import compute_character_metrics
from pystylometry.lexical import (
    compute_function_words,
    compute_hapax_ratios,
    compute_mtld,
    compute_ttr,
    compute_yule,
)
from pystylometry.ngrams import (
    compute_character_bigram_entropy,
    compute_word_bigram_entropy,
)

FEATURE_NAMES = (
    "ttr",
    "root_ttr",
    "log_ttr",
    "mtld",
    "yule_k",
    "yule_i",
    "hapax_ratio",
    "dis_hapax_ratio",
    "sichel_s",
    "honore_r",
    "determiner_ratio",
    "preposition_ratio",
    "conjunction_ratio",
    "pronoun_ratio",
    "auxiliary_ratio",
    "function_word_ratio",
    "function_word_diversity",
    "avg_word_length",
    "avg_sentence_length_chars",
    "punctuation_density",
    "punctuation_variety",
    "vowel_consonant_ratio",
    "digit_ratio",
    "uppercase_ratio",
    "whitespace_ratio",
    "character_bigram_entropy",
    "word_bigram_entropy",
    "sentence_length_cv",
    "paragraph_length_cv",
    "word_burstiness",
    "repeated_bigram_ratio",
    "question_density",
    "exclamation_density",
    "quote_density",
    "newline_density",
)


def _finite(value: object) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _cv(values: list[int]) -> float:
    return float(np.std(values) / np.mean(values)) if values and np.mean(values) else 0.0


@dataclass
class ForensicFeatureExtractor:
    """Extract a fixed, ordered vector; raw text evidence is never normalized away."""

    def extract_dict(self, text: str) -> dict[str, float]:
        text = text or ""
        words = re.findall(r"\b[\w'-]+\b", text.lower())
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        try:
            ttr, mtld, yule = compute_ttr(text), compute_mtld(text), compute_yule(text)
            hapax, function = compute_hapax_ratios(text), compute_function_words(text)
            char = compute_character_metrics(text)
            char_entropy = compute_character_bigram_entropy(text)
            word_entropy = compute_word_bigram_entropy(text)
        except (ValueError, ZeroDivisionError):
            # Pystylometry intentionally rejects some extremely short inputs.
            ttr = mtld = yule = hapax = function = char = char_entropy = word_entropy = None

        def attr(obj: object, name: str) -> float:
            return _finite(getattr(obj, name, 0.0))

        counts: dict[tuple[str, ...], int] = {}
        bigrams = list(zip(words, words[1:]))
        for item in bigrams:
            counts[item] = counts.get(item, 0) + 1
        frequencies = list({word: words.count(word) for word in set(words)}.values())
        values = {
            "ttr": attr(ttr, "ttr"),
            "root_ttr": attr(ttr, "root_ttr"),
            "log_ttr": attr(ttr, "log_ttr"),
            "mtld": attr(mtld, "mtld_average"),
            "yule_k": attr(yule, "yule_k"),
            "yule_i": attr(yule, "yule_i"),
            "hapax_ratio": attr(hapax, "hapax_ratio"),
            "dis_hapax_ratio": attr(hapax, "dis_hapax_ratio"),
            "sichel_s": attr(hapax, "sichel_s"),
            "honore_r": attr(hapax, "honore_r"),
            "determiner_ratio": attr(function, "determiner_ratio"),
            "preposition_ratio": attr(function, "preposition_ratio"),
            "conjunction_ratio": attr(function, "conjunction_ratio"),
            "pronoun_ratio": attr(function, "pronoun_ratio"),
            "auxiliary_ratio": attr(function, "auxiliary_ratio"),
            "function_word_ratio": attr(function, "total_function_word_ratio"),
            "function_word_diversity": attr(function, "function_word_diversity"),
            "avg_word_length": attr(char, "avg_word_length"),
            "avg_sentence_length_chars": attr(char, "avg_sentence_length_chars"),
            "punctuation_density": attr(char, "punctuation_density"),
            "punctuation_variety": attr(char, "punctuation_variety"),
            "vowel_consonant_ratio": attr(char, "vowel_consonant_ratio"),
            "digit_ratio": attr(char, "digit_ratio"),
            "uppercase_ratio": attr(char, "uppercase_ratio"),
            "whitespace_ratio": attr(char, "whitespace_ratio"),
            "character_bigram_entropy": attr(char_entropy, "entropy"),
            "word_bigram_entropy": attr(word_entropy, "entropy"),
            "sentence_length_cv": _cv([len(s.split()) for s in sentences]),
            "paragraph_length_cv": _cv([len(p.split()) for p in paragraphs]),
            "word_burstiness": _cv(frequencies),
            "repeated_bigram_ratio": sum(v > 1 for v in counts.values()) / max(1, len(counts)),
            "question_density": text.count("?") / max(1, len(sentences)),
            "exclamation_density": text.count("!") / max(1, len(sentences)),
            "quote_density": (text.count('"') + text.count("“") + text.count("”"))
            / max(1, len(words)),
            "newline_density": text.count("\n") / max(1, len(words)),
        }
        return {name: _finite(values[name]) for name in FEATURE_NAMES}

    def __call__(self, text: str) -> list[float]:
        values = self.extract_dict(text)
        return [values[name] for name in FEATURE_NAMES]
