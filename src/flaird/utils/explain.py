"""Hierarchical explanation module for FLAIRD.

The module explains a FLAIRD prediction layer by layer, mirroring the model hierarchy::

    text -> input reliability -> decision -> path balance -> feature groups -> features

All attributions rest on a single principle: *counterfactual removal of forensic evidence*,
evaluated exactly on the cached semantic state (the encoder runs once per text).

* Layer 3 (groups): exact Shapley values over the feature groups.
* Layer 4 (features): exact Owen values (Shapley restricted to group-contiguous orderings),
  so the feature values of a group always add up to the value of the group.
* Layer 2 (path balance): ``v(empty)`` is the decision when the forensic stream is neutral.

The module reads no dataset statistics or reference files: everything is computed from the
input text and the checkpoint itself.
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

import numpy as np
import torch

from flaird.data.data_collator import preprocess_text
from flaird.modeling.features import FEATURE_NAMES
from flaird.modeling.modeling_fusion import pool_semantic_states

# --------------------------------------------------------------------------------------
# Static constants (written from `features.py`; not derived from any dataset)
# --------------------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ttr": "Type-token ratio: share of distinct words",
    "root_ttr": "Root type-token ratio: length-corrected share of distinct words",
    "log_ttr": "Log type-token ratio: length-corrected share of distinct words",
    "mtld": "Lexical diversity that is robust to text length",
    "yule_k": "Yule's K: vocabulary repetitiveness (higher means more repetition)",
    "yule_i": "Yule's I: inverse vocabulary repetitiveness",
    "hapax_ratio": "Share of words that appear exactly once",
    "dis_hapax_ratio": "Share of words that appear exactly twice",
    "sichel_s": "Sichel's S: vocabulary richness from words occurring twice",
    "honore_r": "Honore's R: vocabulary richness from words occurring once",
    "determiner_ratio": "Share of determiners (the, a, this, ...)",
    "preposition_ratio": "Share of prepositions (of, in, on, ...)",
    "conjunction_ratio": "Share of conjunctions (and, but, or, ...)",
    "pronoun_ratio": "Share of pronouns (he, it, they, ...)",
    "auxiliary_ratio": "Share of auxiliary verbs (is, have, will, ...)",
    "function_word_ratio": "Total share of function words",
    "function_word_diversity": "Variety of distinct function words used",
    "avg_word_length": "Average word length in characters",
    "avg_sentence_length_chars": "Average sentence length in characters",
    "punctuation_density": "Punctuation marks per 100 words (library chunk summary)",
    "punctuation_variety": "Variety of distinct punctuation marks",
    "vowel_consonant_ratio": "Ratio of vowels to consonants",
    "digit_ratio": "Share of digit characters",
    "uppercase_ratio": "Uppercase letters relative to counted English letters",
    "whitespace_ratio": "Whitespace share in library text chunks",
    "character_bigram_entropy": "Unpredictability of character pairs",
    "word_bigram_entropy": "Unpredictability of word pairs",
    "sentence_length_cv": "Variation of sentence length (structural burstiness)",
    "paragraph_length_cv": "Variation of paragraph length",
    "word_burstiness": "Unevenness of the word-frequency distribution",
    "repeated_bigram_ratio": "Share of word pairs that repeat",
    "question_density": "Question marks per sentence",
    "exclamation_density": "Exclamation marks per sentence",
    "quote_density": "Quotation marks per word",
    "newline_density": "Line breaks per word",
}

# key -> (display name, meaning, color)
GROUP_DISPLAY: dict[str, tuple[str, str, str]] = {
    "lexical_diversity": (
        "Lexical diversity",
        "Richness and variety of vocabulary",
        "#1F5498",
    ),
    "function_word_style": (
        "Function-word style",
        "Grammar-glue word habits",
        "#0CB9C0",
    ),
    "surface_composition": ("Surface composition", "Character-level makeup", "#6FA8DC"),
    "structural_organization": (
        "Structural organization",
        "Sentence and paragraph shape",
        "#264772",
    ),
    "punctuation_and_rhetoric": (
        "Punctuation & rhetoric",
        "Punctuation and rhetorical marks",
        "#F0BA3C",
    ),
    "information_entropy": (
        "Information entropy",
        "Predictability of word and character sequences",
        "#8FA3BF",
    ),
    "repetition_and_burstiness": (
        "Repetition & burstiness",
        "Repetition patterns",
        "#E67E5A",
    ),
}
_NEUTRAL_GROUP_COLOR = "#8FA3BF"

_ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\ufeff"
_PLACEHOLDERS = ("[EMAIL]", "[USER]", "[PHONE]")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"\b[\w'-]+\b")
_ODD_WHITESPACE_RUN = re.compile(r"[^\S\n]{3,}")
_WHITESPACE_RUN = re.compile(r"\s+")
_STANDARD_WHITESPACE = set(" \n\t\r")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplainConfig:
    """Fixed parameters of the explanation module (there are no analysis modes)."""

    max_length: int = 512
    top_k_features: int = 10
    uncertain_band: tuple[float, float] = (0.35, 0.65)
    short_text_words: int = 30
    baseline_value: float = 0.0
    owen_method: Literal["exact", "permutation"] = "exact"
    owen_permutations: int = 512
    max_exact_evaluations: int = 600_000
    chunk_size: int = 4096
    seed: int = 13
    device: str | None = None


# --------------------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------------------


def _to_plain(value: Any, ndigits: int = 6) -> Any:
    """Recursively convert results into JSON-safe Python objects (NaN/Inf become None)."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_plain(getattr(value, f.name), ndigits) for f in fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _to_plain(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v, ndigits) for v in value]
    if isinstance(value, np.ndarray):
        return _to_plain(value.tolist(), ndigits)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return round(value, ndigits) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class _Serializable:
    """Mixin giving every result object ``to_dict`` / ``to_json``."""

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class Capabilities(_Serializable):
    fusion_type: str
    has_attention: bool
    has_gate: bool
    has_generator: bool
    groups: list[str]
    generator_labels: list[str]


@dataclass
class InputDiagnostics(_Serializable):
    num_chars: int
    num_words: int
    num_sentences: int
    num_tokens: int
    truncated: bool
    coverage: float
    zero_width_chars: int
    homoglyph_words: int
    whitespace_anomaly: float
    placeholders: dict[str, int]
    features_all_zero: bool
    reliability: Literal["high", "medium", "low"]
    warnings: list[str]


@dataclass
class DecisionSummary(_Serializable):
    logit: float
    machine_score: float
    prediction_at_0_5: bool
    verdict: str
    verdict_band: Literal["machine", "human", "inconclusive"]


@dataclass
class GeneratorAnalysis(_Serializable):
    probs: dict[str, float]
    predicted: str
    top3: list[str]
    entropy: float
    p_human: float | None
    consistent: bool | None


@dataclass
class PathBalance(_Serializable):
    semantic_logit: float
    forensic_delta: float
    final_logit: float
    semantic_share: float
    forensic_share: float
    gate: float | None
    gate_note: str | None


@dataclass
class GroupAttribution(_Serializable):
    name: str
    display_name: str
    color: str
    phi: float
    direction: Literal["machine", "human", "neutral"]
    abs_rank: int
    attention: float | None
    feature_names: list[str]


@dataclass
class FeatureAttribution(_Serializable):
    name: str
    group: str | None
    value: float
    phi: float
    direction: Literal["machine", "human", "neutral"]
    abs_rank: int
    description: str
    std_error: float | None = None


@dataclass
class TargetAttribution(_Serializable):
    """Attribution for the generator head's target class (log-odds of that class)."""

    target: str
    target_probability: float
    value_full: float
    value_empty: float
    groups: list[GroupAttribution]
    features: list[FeatureAttribution]


@dataclass
class ConsistencyChecks(_Serializable):
    value_function_error: float
    efficiency_error: float
    group_efficiency_error: float
    owen_group_consistency_error: float
    attention_vs_phi_spearman: float | None
    generator_value_function_error: float | None = None
    generator_efficiency_error: float | None = None


@dataclass
class AttributionInfo(_Serializable):
    method: str
    num_evaluations: int
    baseline_value: float
    target: str
    permutations: int | None = None


@dataclass
class ExplanationMeta(_Serializable):
    model_id: str | None
    fusion_type: str
    has_generator: bool
    config: dict[str, Any]
    runtime_seconds: dict[str, float]
    notes: list[str]


@dataclass
class FlairdExplanation(_Serializable):
    meta: ExplanationMeta
    input: InputDiagnostics
    decision: DecisionSummary
    generator: GeneratorAnalysis | None
    balance: PathBalance
    groups: list[GroupAttribution]
    features: list[FeatureAttribution]
    generator_attribution: TargetAttribution | None
    checks: ConsistencyChecks
    attribution: AttributionInfo

    def top_features(self, k: int | None = None) -> list[FeatureAttribution]:
        """Features ranked by absolute contribution."""
        ranked = sorted(self.features, key=lambda f: f.abs_rank)
        return ranked[:k] if k else ranked


# --------------------------------------------------------------------------------------
# Layer 0 - input diagnostics
# --------------------------------------------------------------------------------------


class InputDiagnoser:
    """Rule-based checks on whether a text can be analyzed reliably (no model involved)."""

    def __init__(self, config: ExplainConfig):
        self.config = config

    @staticmethod
    def _scripts(word: str) -> set[str]:
        scripts = set()
        for char in word:
            if char.isalpha():
                try:
                    scripts.add(unicodedata.name(char).split()[0])
                except ValueError:
                    continue
        return scripts

    def _homoglyph_words(self, text: str) -> int:
        return sum(
            1 for word in re.findall(r"\w+", text) if len(self._scripts(word)) > 1
        )

    @staticmethod
    def _whitespace_anomaly(text: str) -> float:
        """Share of whitespace runs containing exotic spaces or 3+ consecutive blanks."""
        runs = _WHITESPACE_RUN.findall(text)
        if not runs:
            return 0.0
        odd = sum(
            1
            for run in runs
            if any(char not in _STANDARD_WHITESPACE for char in run)
            or _ODD_WHITESPACE_RUN.search(run)
        )
        return odd / len(runs)

    def diagnose(
        self, raw_text: str, clean_text: str, num_tokens: int, features_all_zero: bool
    ) -> InputDiagnostics:
        cfg = self.config
        words = _WORD.findall(clean_text)
        sentences = [s for s in _SENTENCE_SPLIT.split(clean_text) if s.strip()]
        truncated = num_tokens > cfg.max_length
        coverage = cfg.max_length / num_tokens if truncated else 1.0
        zero_width = sum(raw_text.count(char) for char in _ZERO_WIDTH_CHARS)
        homoglyphs = self._homoglyph_words(raw_text)
        whitespace = self._whitespace_anomaly(raw_text)
        placeholders = {p: clean_text.count(p) for p in _PLACEHOLDERS}

        warnings: list[str] = []
        if features_all_zero:
            warnings.append(
                "Forensic features could not be extracted (the text is probably too short); "
                "the forensic evidence is empty."
            )
        if len(words) < cfg.short_text_words:
            warnings.append(
                f"Very short text ({len(words)} words): the result is unreliable."
            )
        if truncated:
            warnings.append(
                f"The encoder reads only the first {cfg.max_length} tokens "
                f"({coverage:.0%} of the text)."
            )
        if zero_width:
            warnings.append(
                f"{zero_width} zero-width character(s) detected (possible evasion)."
            )
        if homoglyphs:
            warnings.append(
                f"{homoglyphs} word(s) mix alphabets (possible homoglyph evasion)."
            )
        if whitespace > 0.05:
            warnings.append(
                f"Unusual whitespace in {whitespace:.0%} of whitespace runs."
            )

        if features_all_zero or len(words) < cfg.short_text_words:
            reliability = "low"
        elif coverage < 0.5 or zero_width or homoglyphs or whitespace > 0.05:
            reliability = "medium"
        else:
            reliability = "high"

        return InputDiagnostics(
            num_chars=len(clean_text),
            num_words=len(words),
            num_sentences=len(sentences),
            num_tokens=num_tokens,
            truncated=truncated,
            coverage=coverage,
            zero_width_chars=zero_width,
            homoglyph_words=homoglyphs,
            whitespace_anomaly=whitespace,
            placeholders=placeholders,
            features_all_zero=features_all_zero,
            reliability=reliability,
            warnings=warnings,
        )


# --------------------------------------------------------------------------------------
# Forward pass (the only encoder call per explanation)
# --------------------------------------------------------------------------------------


@dataclass
class ForwardState:
    """Everything the attribution needs from the single real forward pass."""

    num_tokens: int
    features: torch.Tensor  # [1, F] raw forensic features (model dtype)
    pooled_semantic: torch.Tensor  # [1, 1, H] pooled semantic state
    logit: float
    generator_logits: np.ndarray | None
    attention: np.ndarray | None  # [G]
    gate: float | None


class ForwardRunner:
    """Runs the real model once and caches the pooled semantic vector."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, config: ExplainConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(text, truncation=False, return_attention_mask=False)
        return len(encoded["input_ids"])

    @torch.inference_mode()
    def run(self, clean_text: str) -> ForwardState:
        encoded = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        # Features come from the whole preprocessed text, exactly as in training.
        features = self.model.extract_forensic_features(
            [clean_text], return_tensors=True
        )
        features = features.to(device=self.device, dtype=self.dtype)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forensic_features=features,
            output_fusion_states=True,
        )
        states = outputs.fusion_states
        pooled = pool_semantic_states(
            states.semantic_states, attention_mask, self.model.config.classifier_pooling
        )
        fusion = states.fusion_outputs
        generator = outputs.generator_logits
        return ForwardState(
            num_tokens=self.count_tokens(clean_text),
            features=features,
            pooled_semantic=pooled.unsqueeze(1),
            logit=float(outputs.logits.reshape(-1)[0]),
            generator_logits=None
            if generator is None
            else generator[0].double().cpu().numpy(),
            attention=None
            if fusion.feature_attention is None
            else fusion.feature_attention[0].double().cpu().numpy(),
            gate=None if fusion.fusion_gate is None else float(fusion.fusion_gate[0]),
        )


# --------------------------------------------------------------------------------------
# Value functions (strategy pattern): which scalar the attribution explains
# --------------------------------------------------------------------------------------


class Target(ABC):
    """A scalar read-out of the prediction head, explained by the attribution."""

    name: str

    @abstractmethod
    def reduce(self, model: torch.nn.Module, head_output: torch.Tensor) -> torch.Tensor:
        """Map head output ``[B, H]`` to one value per row ``[B]``."""


class BinaryLogitTarget(Target):
    """Machine logit of the binary head (``sigmoid`` of it is the machine score)."""

    name = "machine_logit"

    def reduce(self, model: torch.nn.Module, head_output: torch.Tensor) -> torch.Tensor:
        return model.classifier(head_output).reshape(-1)


class GeneratorClassTarget(Target):
    """Log-odds of one generator class: ``z_c - logsumexp(z_others)``."""

    def __init__(self, class_id: int, label: str):
        self.class_id = class_id
        self.name = f"generator:{label}"

    def reduce(self, model: torch.nn.Module, head_output: torch.Tensor) -> torch.Tensor:
        logits = model.generator_classifier(head_output)
        others = logits.clone()
        others[:, self.class_id] = float("-inf")
        return logits[:, self.class_id] - torch.logsumexp(others, dim=-1)


class MaskGame:
    """Cooperative game over forensic features, evaluated without re-running the encoder.

    ``v(mask)`` is the model read-out when only the features in ``mask`` keep their observed
    values and all others take the baseline value. The pooled semantic vector is passed to
    the fusion layer as a ``[B, 1, H]`` tensor with no attention mask: for both ``cls`` and
    ``mean`` pooling the fusion layers then see exactly that vector. All targets share one
    head evaluation per mask.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        state: ForwardState,
        targets: Sequence[Target],
        config: ExplainConfig,
    ):
        self.model = model
        self.state = state
        self.targets = list(targets)
        self.config = config
        self.baseline = torch.tensor(
            config.baseline_value,
            dtype=state.features.dtype,
            device=state.features.device,
        )

    @torch.inference_mode()
    def evaluate(self, bits: np.ndarray) -> np.ndarray:
        """Evaluate ``[M, F]`` boolean feature masks; returns ``[M, T]`` float64 values."""
        fusion, head = self.model.model.fusion, self.model.head
        feature_encoder = self.model.model.feature_encoder
        outputs = []
        for start in range(0, len(bits), self.config.chunk_size):
            keep = torch.from_numpy(bits[start : start + self.config.chunk_size])
            keep = keep.to(self.state.features.device)
            raw = torch.where(keep, self.state.features, self.baseline)
            semantic = self.state.pooled_semantic.expand(raw.shape[0], -1, -1)
            fused = fusion(semantic, feature_encoder(raw), None).fused_state
            head_output = head(fused)
            columns = [t.reduce(self.model, head_output).double() for t in self.targets]
            outputs.append(torch.stack(columns, dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)


# --------------------------------------------------------------------------------------
# Shapley (groups) and Owen (features) engines
# --------------------------------------------------------------------------------------


def _popcount_table(size: int) -> np.ndarray:
    table = np.zeros(size, dtype=np.int64)
    for i in range(1, size):
        table[i] = table[i >> 1] + (i & 1)
    return table


def _subset_unions(masks: Sequence[int]) -> np.ndarray:
    """Union mask of every subset of ``masks`` (subset id bit ``j`` selects ``masks[j]``)."""
    unions = np.zeros(1 << len(masks), dtype=np.uint64)
    for subset in range(1, len(unions)):
        low = (subset & -subset).bit_length() - 1
        unions[subset] = unions[subset & (subset - 1)] | np.uint64(masks[low])
    return unions


def _shapley_weights(n: int) -> np.ndarray:
    """``w_n(s) = s! (n-s-1)! / n!`` for ``s = 0..n-1``."""
    return np.array(
        [
            math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            for s in range(n)
        ]
    )


@dataclass
class AttributionResult:
    """Raw output of the attribution engine (``T`` = number of explained targets)."""

    method: str
    phi_groups: np.ndarray  # [G, T] exact Shapley values of the groups
    phi_features: np.ndarray  # [F, T] Owen values (exact or estimated)
    v_full: np.ndarray  # [T]
    v_empty: np.ndarray  # [T]
    num_evaluations: int
    stderr_features: np.ndarray | None = None  # [F, T] (permutation method only)
    permutations: int | None = None


class HierarchicalAttributor:
    """Exact group Shapley values plus exact (or sampled) feature Owen values.

    Masks are Python/NumPy ``uint64`` bitmasks over the feature positions.
    """

    def __init__(self, group_indices: Sequence[Sequence[int]], num_features: int):
        self.groups = [list(g) for g in group_indices]
        self.num_features = num_features
        flat = [i for g in self.groups for i in g]
        if len(set(flat)) != len(flat):
            raise ValueError("Feature groups must be disjoint.")
        if num_features > 63:
            raise ValueError("At most 63 features are supported (uint64 bitmasks).")
        self.group_masks = [sum(1 << i for i in g) for g in self.groups]
        self.all_mask = 0
        for mask in self.group_masks:
            self.all_mask |= mask

    # -- helpers ---------------------------------------------------------------------
    def _local_to_global(self, k: int) -> np.ndarray:
        """Global bitmask of every subset of group ``k`` (local subset id -> global mask)."""
        return _subset_unions([1 << i for i in self.groups[k]])

    def _others_unions(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Union mask and size of every subset of the groups other than ``k``."""
        others = [self.group_masks[j] for j in range(len(self.groups)) if j != k]
        return _subset_unions(others), _popcount_table(1 << len(others))

    def _bits(self, masks: np.ndarray) -> np.ndarray:
        shifts = np.arange(self.num_features, dtype=np.uint64)
        return ((masks[:, None] >> shifts) & np.uint64(1)).astype(bool)

    def expected_exact_evaluations(self) -> int:
        """Number of distinct masks the exact method evaluates."""
        n = len(self.groups)
        partial = sum((1 << len(g)) - 2 for g in self.groups)
        return (1 << n) + (1 << (n - 1)) * partial

    def _lookup(
        self, unique: np.ndarray, values: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        return values[np.searchsorted(unique, masks)]

    def _edge_values(
        self, unique: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        full = self._lookup(unique, values, np.array([self.all_mask], dtype=np.uint64))[
            0
        ]
        empty = self._lookup(unique, values, np.array([0], dtype=np.uint64))[0]
        return full, empty

    def _group_shapley(
        self, unique: np.ndarray, values: np.ndarray, num_targets: int
    ) -> np.ndarray:
        n = len(self.groups)
        weights = _shapley_weights(n)
        phi = np.zeros((n, num_targets))
        for k in range(n):
            unions, pops = self._others_unions(k)
            full = np.uint64(self.group_masks[k])
            gain = self._lookup(unique, values, unions | full) - self._lookup(
                unique, values, unions
            )
            phi[k] = weights[pops] @ gain
        return phi

    # -- exact -----------------------------------------------------------------------
    def exact(self, game: MaskGame) -> AttributionResult:
        """Exact Owen values: enumerate every (union of groups) + (subset of one group)."""
        n = len(self.groups)
        group_weights = _shapley_weights(n)
        tables = []
        needed = [_subset_unions(self.group_masks)]
        for k in range(n):
            unions, pops = self._others_unions(k)
            local_to_global = self._local_to_global(k)
            tables.append((unions, pops, local_to_global))
            needed.append((unions[:, None] | local_to_global[None, :]).ravel())
        unique = np.unique(np.concatenate(needed))
        values = game.evaluate(self._bits(unique))
        num_targets = values.shape[1]

        phi_features = np.zeros((self.num_features, num_targets))
        for k, (unions, pops, local_to_global) in enumerate(tables):
            w_groups = group_weights[pops]
            size = len(self.groups[k])
            table = self._lookup(
                unique, values, unions[:, None] | local_to_global[None, :]
            )
            w_inner = _shapley_weights(size)
            inner_pops = _popcount_table(1 << size)
            subsets = np.arange(1 << size)
            for pos, feature in enumerate(self.groups[k]):
                without = subsets[(subsets & (1 << pos)) == 0]
                gain = table[:, without | (1 << pos), :] - table[:, without, :]
                phi_features[feature] = np.einsum(
                    "s,t,stj->j", w_groups, w_inner[inner_pops[without]], gain
                )

        full, empty = self._edge_values(unique, values)
        return AttributionResult(
            method="exact",
            phi_groups=self._group_shapley(unique, values, num_targets),
            phi_features=phi_features,
            v_full=full,
            v_empty=empty,
            num_evaluations=len(unique),
        )

    # -- sampled (documented fallback) -------------------------------------------------
    def permutation(
        self, game: MaskGame, permutations: int, seed: int
    ) -> AttributionResult:
        """Antithetic group-contiguous permutation estimate of the Owen values."""
        n = len(self.groups)
        rng = np.random.default_rng(seed)
        pairs = max(1, permutations // 2)
        orders: list[list[int]] = []
        for _ in range(pairs):
            order: list[int] = []
            for k in rng.permutation(n):
                order.extend(rng.permutation(self.groups[k]).tolist())
            orders.append(order)
            orders.append(order[::-1])  # antithetic partner keeps the groups contiguous

        chains = np.zeros((len(orders), len(orders[0]) + 1), dtype=np.uint64)
        for row, order in enumerate(orders):
            for col, feature in enumerate(order, start=1):
                chains[row, col] = chains[row, col - 1] | np.uint64(1 << feature)
        unique = np.unique(
            np.concatenate([chains.ravel(), _subset_unions(self.group_masks)])
        )
        values = game.evaluate(self._bits(unique))
        num_targets = values.shape[1]

        chain_values = self._lookup(unique, values, chains.ravel()).reshape(
            chains.shape + (num_targets,)
        )
        contributions = np.zeros((len(orders), self.num_features, num_targets))
        for row, order in enumerate(orders):
            contributions[row, order] = chain_values[row, 1:] - chain_values[row, :-1]
        per_pair = 0.5 * (contributions[0::2] + contributions[1::2])
        stderr = per_pair.std(axis=0, ddof=1) / math.sqrt(pairs) if pairs > 1 else None

        full, empty = self._edge_values(unique, values)
        return AttributionResult(
            method="permutation",
            phi_groups=self._group_shapley(unique, values, num_targets),
            phi_features=per_pair.mean(axis=0),
            v_full=full,
            v_empty=empty,
            num_evaluations=len(unique),
            stderr_features=stderr,
            permutations=len(orders),
        )

    def attribute(self, game: MaskGame, config: ExplainConfig) -> AttributionResult:
        if config.owen_method == "permutation":
            return self.permutation(game, config.owen_permutations, config.seed)
        expected = self.expected_exact_evaluations()
        if expected > config.max_exact_evaluations:
            raise ValueError(
                f"Exact Owen values need {expected} evaluations (limit "
                f"{config.max_exact_evaluations}); use owen_method='permutation'."
            )
        return self.exact(game)


# --------------------------------------------------------------------------------------
# Cross-check helpers
# --------------------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        tie = values == value
        ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rank correlation (None when either input is constant)."""
    ra, rb = _average_ranks(np.asarray(a, float)), _average_ranks(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _direction(phi: float, eps: float = 1e-9) -> Literal["machine", "human", "neutral"]:
    if abs(phi) <= eps:
        return "neutral"
    return "machine" if phi > 0 else "human"


def _log_odds(logits: np.ndarray, class_id: int) -> float:
    others = np.delete(logits, class_id)
    peak = others.max()
    return float(logits[class_id] - (peak + np.log(np.exp(others - peak).sum())))


# --------------------------------------------------------------------------------------
# The explainer
# --------------------------------------------------------------------------------------


class FlairdExplainer:
    """Explains FLAIRD predictions with the hierarchical Shapley/Owen framework."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: ExplainConfig | None = None,
        model_id: str | None = None,
    ):
        self.config = config or ExplainConfig()
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = model.eval()
        if self.config.device:
            self.model.to(self.config.device)
        self.model_config = self.model.config
        self.group_names: list[str] = list(self.model_config.feature_group_names)
        self.group_indices = [list(g) for g in self.model_config.feature_group_indices]
        self.generator_labels = self._generator_labels()
        self.runner = ForwardRunner(self.model, tokenizer, self.config)
        self.diagnoser = InputDiagnoser(self.config)
        self.attributor = HierarchicalAttributor(
            self.group_indices, self.model_config.forensic_feature_dim
        )

    @staticmethod
    def load_components(
        model_id: str, device: str | None = None
    ) -> tuple[torch.nn.Module, Any]:
        """Load a released checkpoint with the Auto classes (``trust_remote_code=True``)."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True
        )
        model = model.to(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model, tokenizer

    @classmethod
    def from_pretrained(
        cls, model_id: str, config: ExplainConfig | None = None
    ) -> FlairdExplainer:
        """Build an explainer from a Hugging Face model id."""
        config = config or ExplainConfig()
        model, tokenizer = cls.load_components(model_id, config.device)
        return cls(model, tokenizer, config, model_id=model_id)

    # -- capabilities ------------------------------------------------------------------
    def _generator_labels(self) -> list[str]:
        if getattr(self.model, "generator_classifier", None) is None:
            return []
        # Keys may be strings after a JSON round trip of the config.
        mapping = {int(k): v for k, v in self.model_config.id2generator_label.items()}
        return [mapping[i] for i in sorted(mapping)]

    @property
    def capabilities(self) -> Capabilities:
        attention = self.model_config.fusion_type == "attention"
        return Capabilities(
            fusion_type=self.model_config.fusion_type,
            has_attention=attention,
            has_gate=attention,
            has_generator=bool(self.generator_labels),
            groups=self.group_names,
            generator_labels=self.generator_labels,
        )

    # -- public API --------------------------------------------------------------------
    def explain(
        self, text: str, generator_target: str | None = None
    ) -> FlairdExplanation:
        """Explain one text. ``generator_target`` selects the generator class to attribute."""
        if generator_target is not None:
            if not self.generator_labels:
                raise ValueError("This checkpoint has no generator head.")
            if generator_target not in self.generator_labels:
                raise ValueError(
                    f"Unknown generator '{generator_target}'. Valid: {self.generator_labels}"
                )
        clean = preprocess_text(text or "")
        if not clean:
            raise ValueError("The text is empty.")

        timings: dict[str, float] = {}
        with self._timer(timings, "forward"):
            state = self.runner.run(clean)
        values = state.features[0].double().cpu().numpy()

        diagnostics = self.diagnoser.diagnose(
            text, clean, state.num_tokens, bool(np.all(values == 0))
        )
        decision = self._decision(state.logit)
        generator = self._generator_analysis(state, decision.machine_score)
        target_label = generator_target or (generator.predicted if generator else None)

        targets: list[Target] = [BinaryLogitTarget()]
        if target_label is not None:
            targets.append(
                GeneratorClassTarget(
                    self.generator_labels.index(target_label), target_label
                )
            )
        game = MaskGame(self.model, state, targets, self.config)
        with self._timer(timings, "attribution"):
            result = self.attributor.attribute(game, self.config)

        return self._build(
            state,
            values,
            diagnostics,
            decision,
            generator,
            result,
            target_label,
            timings,
        )

    def explain_batch(self, texts: Sequence[str]) -> list[FlairdExplanation]:
        return [self.explain(text) for text in texts]

    def counterfactual_logit(self, text: str, removed_groups: Sequence[str]) -> float:
        """Machine logit after replacing the given feature groups by the baseline value."""
        unknown = set(removed_groups) - set(self.group_names)
        if unknown:
            raise ValueError(
                f"Unknown feature groups {sorted(unknown)}. Valid: {self.group_names}"
            )
        clean = preprocess_text(text or "")
        if not clean:
            raise ValueError("The text is empty.")
        game = MaskGame(
            self.model, self.runner.run(clean), [BinaryLogitTarget()], self.config
        )
        keep = np.ones((1, self.model_config.forensic_feature_dim), dtype=bool)
        for name in removed_groups:
            keep[0, self.group_indices[self.group_names.index(name)]] = False
        return float(game.evaluate(keep)[0, 0])

    # -- assembly ----------------------------------------------------------------------
    @staticmethod
    @contextmanager
    def _timer(store: dict[str, float], key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            store[key] = round(time.perf_counter() - start, 4)

    def _decision(self, logit: float) -> DecisionSummary:
        score = 1.0 / (1.0 + math.exp(-logit))
        low, high = self.config.uncertain_band
        if score >= high:
            band, verdict = "machine", "Likely machine-generated"
        elif score <= low:
            band, verdict = "human", "Likely human-written"
        else:
            band, verdict = "inconclusive", "Inconclusive"
        return DecisionSummary(
            logit=logit,
            machine_score=score,
            prediction_at_0_5=score >= 0.5,
            verdict=verdict,
            verdict_band=band,
        )

    def _generator_analysis(
        self, state: ForwardState, machine_score: float
    ) -> GeneratorAnalysis | None:
        if state.generator_logits is None:
            return None
        logits = state.generator_logits
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        labels = self.generator_labels
        order = np.argsort(-probs)
        entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
        p_human = float(probs[labels.index("human")]) if "human" in labels else None
        consistent = (
            None if p_human is None else abs((1.0 - p_human) - machine_score) <= 0.25
        )
        return GeneratorAnalysis(
            probs={label: float(p) for label, p in zip(labels, probs, strict=True)},
            predicted=labels[int(order[0])],
            top3=[labels[int(i)] for i in order[:3]],
            entropy=entropy,
            p_human=p_human,
            consistent=consistent,
        )

    @staticmethod
    def _group_display(name: str) -> tuple[str, str]:
        if name in GROUP_DISPLAY:
            return GROUP_DISPLAY[name][0], GROUP_DISPLAY[name][2]
        return name.replace("_", " ").title(), _NEUTRAL_GROUP_COLOR

    def _group_attributions(
        self, phi: np.ndarray, attention: np.ndarray | None
    ) -> list[GroupAttribution]:
        ranks = _average_ranks(-np.abs(phi))
        out = []
        for k, name in enumerate(self.group_names):
            display, color = self._group_display(name)
            out.append(
                GroupAttribution(
                    name=name,
                    display_name=display,
                    color=color,
                    phi=float(phi[k]),
                    direction=_direction(float(phi[k])),
                    abs_rank=int(ranks[k]) + 1,
                    attention=None if attention is None else float(attention[k]),
                    feature_names=[FEATURE_NAMES[i] for i in self.group_indices[k]],
                )
            )
        return out

    def _feature_attributions(
        self, phi: np.ndarray, values: np.ndarray, stderr: np.ndarray | None
    ) -> list[FeatureAttribution]:
        owner = {
            i: self.group_names[k] for k, g in enumerate(self.group_indices) for i in g
        }
        ranks = _average_ranks(-np.abs(phi))
        return [
            FeatureAttribution(
                name=name,
                group=owner.get(i),
                value=float(values[i]),
                phi=float(phi[i]),
                direction=_direction(float(phi[i])),
                abs_rank=int(ranks[i]) + 1,
                description=FEATURE_DESCRIPTIONS.get(name, name.replace("_", " ")),
                std_error=None if stderr is None else float(stderr[i]),
            )
            for i, name in enumerate(FEATURE_NAMES)
        ]

    def _balance(self, state: ForwardState, result: AttributionResult) -> PathBalance:
        full, empty = float(result.v_full[0]), float(result.v_empty[0])
        semantic, forensic = abs(empty), abs(full - empty)
        total = semantic + forensic
        share = 0.5 if total == 0 else semantic / total
        return PathBalance(
            semantic_logit=empty,
            forensic_delta=full - empty,
            final_logit=full,
            semantic_share=share,
            forensic_share=1.0 - share,
            gate=state.gate,
            gate_note=None
            if state.gate is not None
            else "Gate exists for attention fusion only.",
        )

    def _checks(
        self, state: ForwardState, result: AttributionResult
    ) -> ConsistencyChecks:
        full, empty = float(result.v_full[0]), float(result.v_empty[0])
        phi_f, phi_g = result.phi_features[:, 0], result.phi_groups[:, 0]
        group_sums = np.array([sum(phi_f[i] for i in g) for g in self.group_indices])
        return ConsistencyChecks(
            value_function_error=abs(full - state.logit),
            efficiency_error=abs(phi_f.sum() - (full - empty)),
            group_efficiency_error=abs(phi_g.sum() - (full - empty)),
            owen_group_consistency_error=float(np.max(np.abs(group_sums - phi_g))),
            attention_vs_phi_spearman=(
                None
                if state.attention is None
                else spearman(state.attention, np.abs(phi_g))
            ),
        )

    def _generator_attribution(
        self,
        state: ForwardState,
        values: np.ndarray,
        generator: GeneratorAnalysis,
        result: AttributionResult,
        target_label: str,
        checks: ConsistencyChecks,
    ) -> TargetAttribution:
        full, empty = float(result.v_full[1]), float(result.v_empty[1])
        class_id = self.generator_labels.index(target_label)
        checks.generator_value_function_error = abs(
            full - _log_odds(state.generator_logits, class_id)
        )
        checks.generator_efficiency_error = abs(
            result.phi_features[:, 1].sum() - (full - empty)
        )
        stderr = (
            None if result.stderr_features is None else result.stderr_features[:, 1]
        )
        return TargetAttribution(
            target=target_label,
            target_probability=generator.probs[target_label],
            value_full=full,
            value_empty=empty,
            groups=self._group_attributions(result.phi_groups[:, 1], state.attention),
            features=self._feature_attributions(
                result.phi_features[:, 1], values, stderr
            ),
        )

    def _build(
        self,
        state: ForwardState,
        values: np.ndarray,
        diagnostics: InputDiagnostics,
        decision: DecisionSummary,
        generator: GeneratorAnalysis | None,
        result: AttributionResult,
        target_label: str | None,
        timings: dict[str, float],
    ) -> FlairdExplanation:
        caps = self.capabilities
        notes: list[str] = []
        if not caps.has_attention:
            notes.append("Concatenation fusion: attention and gate are not available.")
        if not caps.has_generator:
            notes.append("Single-task checkpoint: no generator head.")

        stderr = (
            None if result.stderr_features is None else result.stderr_features[:, 0]
        )
        checks = self._checks(state, result)
        generator_attribution = None
        if target_label is not None and generator is not None:
            generator_attribution = self._generator_attribution(
                state, values, generator, result, target_label, checks
            )
        return FlairdExplanation(
            meta=ExplanationMeta(
                model_id=self.model_id,
                fusion_type=caps.fusion_type,
                has_generator=caps.has_generator,
                config=_to_plain(self.config),
                runtime_seconds=timings,
                notes=notes,
            ),
            input=diagnostics,
            decision=decision,
            generator=generator,
            balance=self._balance(state, result),
            groups=self._group_attributions(result.phi_groups[:, 0], state.attention),
            features=self._feature_attributions(
                result.phi_features[:, 0], values, stderr
            ),
            generator_attribution=generator_attribution,
            checks=checks,
            attribution=AttributionInfo(
                method=result.method,
                num_evaluations=result.num_evaluations,
                baseline_value=self.config.baseline_value,
                target="machine_logit",
                permutations=result.permutations,
            ),
        )


# --------------------------------------------------------------------------------------
# Human-readable report
# --------------------------------------------------------------------------------------


def explain_to_markdown(explanation: FlairdExplanation, top_k: int = 10) -> str:
    """Compact markdown report used by the CLI."""
    d, b = explanation.decision, explanation.balance
    lines = [
        f"## {d.verdict}",
        f"- Machine score: **{d.machine_score:.1%}** (logit {d.logit:+.3f}; "
        + "prior-adjusted score, not a calibrated probability)",
        f"- Input reliability: **{explanation.input.reliability}**",
    ]
    lines += [f"  - Warning: {w}" for w in explanation.input.warnings]
    if explanation.generator:
        g = explanation.generator
        lines.append(
            "- Generator estimate: "
            + ", ".join(f"{x} {g.probs[x]:.1%}" for x in g.top3)
        )
        if g.consistent is False:
            lines.append("  - Warning: the binary and generator heads disagree.")
    lines += [
        "",
        "### Path balance (semantic-anchored decomposition)",
        f"- Semantic logit (neutral forensic input): {b.semantic_logit:+.3f}",
        f"- Forensic contribution: {b.forensic_delta:+.3f}",
        f"- Final logit: {b.final_logit:+.3f}",
    ]
    if b.gate is not None:
        lines.append(f"- Semantic reliance (fusion gate): {b.gate:.2f}")
    lines += ["", "### Feature groups (Shapley; + pushes toward machine)"]
    for g in sorted(explanation.groups, key=lambda x: x.abs_rank):
        attention = "" if g.attention is None else f", attention {g.attention:.2f}"
        lines.append(f"- {g.display_name}: {g.phi:+.3f}{attention}")
    lines += ["", f"### Top {top_k} forensic features (Owen values)"]
    for f in explanation.top_features(top_k):
        lines.append(
            f"- {f.name} = {f.value:.4g}: {f.phi:+.3f} ({f.direction}) - {f.description}"
        )
    return "\n".join(lines)
