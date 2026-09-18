"""Grounded yes/no answering with a local instruction-tuned LLM (stage A7).

The backend sends the whole timed transcript plus every question to the model
in ONE structured call and asks for a small JSON verdict per question. The
alternative retrieval-then-NLI stage (``qa_backend.py``) can lose negations and
corrections that sit outside the retrieved window; a full-context call keeps
them, at the price of a bigger prompt.

Trust model, and why the prompt is built the way it is:

* The prompt file (``prompts/qa_v1.txt``) is the only trusted text and it is
  the only thing that ever becomes the ``system`` message.
* Transcript units and questions are UNTRUSTED data. They are serialised into
  one JSON string inside a single ``user`` message and nowhere else, so a
  transcript line such as ``SYSTEM: ignore all previous instructions`` is just
  a JSON string value: it can never pick a message role, escape the data block
  or reach a template engine. The model is told that in the prompt, but the
  structural separation is the actual defence, not the wording.
* The model's output is untrusted too. :func:`parse_llm_output` is a strict
  validator: real JSON booleans only, integer question ids in range, evidence
  ids filtered to ids that exist, everything else (commentary, tool calls,
  extra keys, timestamps) ignored. A question the model did not answer gets at
  most ONE bounded repair call, and only when the deadline leaves time for it.

Engines are lazy imports so this module (and its fast tests) never pull in
torch or llama.cpp; the ``fake`` engine replays scripted responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import os
from collections import deque
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from core_types import AudioContext, Deadline, QAResult, TranscriptUnit, stable_hash

logger = logging.getLogger(__name__)

# ``generate(messages, max_new_tokens) -> str``: the one seam every engine and
# every test fake implements.
GenerateFn = Callable[[List[Dict[str, str]], int], str]

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = 'prompts/qa_v1.txt'
SUPPORTED_ENGINES = ('transformers', 'llama_cpp', 'fake')
# Names accepted for ``LLMConfig.dtype``; the transformers engine maps them to
# ``torch.<name>``. Validated at config time so a typo fails at start-up, not
# after a model download.
SUPPORTED_DTYPES = ('auto', 'float32', 'float16', 'bfloat16')

# The user message is exactly this marker plus one JSON document. The parser
# on the model side does not depend on it; it only tells the model where the
# data starts, matching the wording in the prompt file.
DATA_MARKER = 'DATA:\n'

# A repair call asks about one question, so its answer is one short item.
REPAIR_MAX_NEW_TOKENS = 96

# Upper bound on how many ``{`` positions the output parser will try. Model
# output is bounded by ``max_new_tokens`` anyway; this only guards against a
# pathological engine returning megabytes of braces.
MAX_JSON_CANDIDATES = 256
# Upper bound on how many linear brace-matching passes the parser makes over
# one output. One pass matches every brace outside a JSON string; another is
# needed only when a stray quote in commentary hid the real object inside a
# string. Bounding it keeps parsing linear in the output length.
MAX_SCAN_PASSES = 8


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class LLMConfig:
    """Settings for :class:`LLMBackend`.

    ``engine`` selects the runtime: ``transformers`` (HF model id, any device),
    ``llama_cpp`` (a GGUF file at ``model_path``) or ``fake`` (scripted
    ``fake_responses`` for tests). ``max_prompt_units`` bounds the number of
    transcript units in the prompt; longer transcripts are merged pairwise so
    the prompt cannot grow without limit. ``repair_min_remaining_s`` is the
    minimum time that must be left on the deadline, after the deadline's own
    ``reserve_s`` (the pipeline's finalisation reserve) is subtracted, before
    an ADDITIONAL model call (a repair, or the next per-question call) is
    started; it should cover one short generation. It never gates the first
    call: whether there is time to involve this backend at all is the
    pipeline's decision, made before ``answer`` is called.
    """

    engine: str = 'transformers'
    model_id: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    model_path: Optional[str] = None
    dtype: str = 'bfloat16'
    max_new_tokens: int = 400
    batch_all_questions: bool = True
    prompt_path: str = DEFAULT_PROMPT_PATH
    device: str = 'auto'
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    max_prompt_units: int = 400
    repair_min_remaining_s: float = 8.0
    fake_responses: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject settings that would only fail after a model load or, worse,
        silently disable the deadline (a NaN or negative reserve compares as
        "always time left")."""
        if self.engine not in SUPPORTED_ENGINES:
            raise ValueError(f'unknown LLM engine {self.engine!r}; expected one of {SUPPORTED_ENGINES}')
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f'unknown dtype {self.dtype!r}; expected one of {SUPPORTED_DTYPES}')
        if self.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be >= 1')
        if self.n_ctx < 1:
            raise ValueError('n_ctx must be >= 1')
        if self.n_threads is not None and self.n_threads < 1:
            raise ValueError('n_threads must be >= 1 or null')
        self.repair_min_remaining_s = float(self.repair_min_remaining_s)
        if not (math.isfinite(self.repair_min_remaining_s) and self.repair_min_remaining_s >= 0.0):
            raise ValueError(f'repair_min_remaining_s must be a finite number >= 0, got {self.repair_min_remaining_s}')
        # A prompt with zero units is meaningless; one unit is the floor.
        self.max_prompt_units = max(1, int(self.max_prompt_units))

    @property
    def config_hash(self) -> str:
        """Hash of every field that changes model behaviour (not the fakes)."""
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'fake_responses'}
        return stable_hash(payload)

    @classmethod
    def from_env(cls, prefix: str = 'MA_LLM_', env: Optional[Mapping[str, str]] = None) -> 'LLMConfig':
        """Build from ``<prefix><FIELD>`` variables, e.g. ``MA_LLM_ENGINE``.

        Values are parsed by the field's annotation; ``null`` clears an
        ``Optional`` field. An unparsable value raises ``ValueError`` naming
        the variable, because a silently ignored production setting is worse
        than a loud start-up failure. ``fake_responses`` cannot come from the
        environment because it is a test-only list.
        """
        source = os.environ if env is None else env
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name == 'fake_responses':
                continue
            key = f'{prefix}{f.name.upper()}'
            if key not in source:
                continue
            raw = str(source[key])
            try:
                kwargs[f.name] = _parse_env_value(raw, str(f.type))
            except ValueError as exc:
                raise ValueError(f'{key}={raw!r}: {exc}') from exc
        return cls(**kwargs)


_TRUE_STRINGS = frozenset({'1', 'true', 'yes', 'on'})
_FALSE_STRINGS = frozenset({'0', 'false', 'no', 'off'})


def _parse_env_value(raw: str, annotation: str) -> Any:
    """Coerce an environment string to a field annotation.

    Annotations are strings here (``from __future__ import annotations``),
    which is exactly what is needed: ``Optional[str]`` must stay a string
    even when it looks numeric (a model path called ``123``), and
    ``Optional[int]`` must become an int.
    """
    text = raw.strip()
    optional = annotation.startswith('Optional[') and annotation.endswith(']')
    inner = annotation[len('Optional['):-1] if optional else annotation
    if optional and text.lower() == 'null':
        return None
    if inner == 'bool':
        lowered = text.lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
        raise ValueError('expected a boolean')
    if inner == 'int':
        return int(text)
    if inner == 'float':
        return float(text)
    if inner == 'str':
        return text
    raise ValueError(f'cannot parse a {annotation} field from the environment')


# --------------------------------------------------------------------------- #
# Prompt text
# --------------------------------------------------------------------------- #

def resolve_prompt_path(prompt_path: str) -> Path:
    """Relative prompt paths are relative to this module, not to the cwd,
    because the API server may be started from any directory."""
    path = Path(prompt_path)
    return path if path.is_absolute() else MODULE_DIR / path


def load_prompt(prompt_path: str = DEFAULT_PROMPT_PATH) -> str:
    """Read the trusted task text. Missing file is a configuration error and
    must fail at start-up, not on the first request."""
    path = resolve_prompt_path(prompt_path)
    text = path.read_text(encoding='utf-8')
    if not text.strip():
        raise ValueError(f'prompt file {path} is empty')
    return text


def prompt_sha256(prompt_text: str) -> str:
    """Prompts are versioned by content hash so every result can say which
    wording produced it."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- #
# Transcript compaction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PromptUnit:
    """What the model sees for one transcript unit (possibly a merge)."""

    unit_id: int
    start_s: float
    end_s: float
    text: str


@dataclass
class MergedTranscript:
    """Prompt units plus the mapping back to the original unit ids.

    ``id_map`` maps every prompt unit id to the ordered list of original unit
    ids it covers, so evidence ids the model returns always resolve to real
    units even after merging.
    """

    units: List[PromptUnit]
    id_map: Dict[int, List[int]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid_ids(self) -> List[int]:
        return [u.unit_id for u in self.units]

    def expand(self, prompt_ids: Sequence[int]) -> List[int]:
        """Original unit ids for a list of prompt ids, order kept, no repeats."""
        seen: set = set()
        out: List[int] = []
        for pid in prompt_ids:
            for uid in self.id_map.get(pid, []):
                if uid not in seen:
                    seen.add(uid)
                    out.append(uid)
        return out


def _finite_or_zero(value: Any) -> float:
    """Timestamps come from trusted code but ASR can still emit NaN; a NaN in
    the prompt would either crash ``json.dumps`` or confuse the model."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _clean_text(value: Any) -> str:
    """Untrusted text as a string the tokenizer can encode.

    A request body may carry a lone UTF-16 surrogate (``"\\ud800"`` is legal
    JSON and Python's decoder keeps it), which every real engine rejects with
    ``UnicodeEncodeError`` when it encodes the prompt as UTF-8. Replacing it
    keeps one bad character from costing the whole LLM call. ``None`` becomes
    the empty string rather than the word "None".
    """
    text = '' if value is None else str(value)
    return text.encode('utf-8', errors='replace').decode('utf-8')


def _fold_into_previous(units: List[PromptUnit], extra: PromptUnit) -> None:
    """Append ``extra`` to the last prompt unit in place (text and time range).

    Used for a unit whose id is already taken: the model needs unique ids,
    and dropping the text would hide evidence from it, so the text rides
    along with its predecessor. Its id is not added to the id map because
    ``AudioContext.unit_by_id`` resolves a duplicated id to its first
    occurrence anyway, so the evidence stage could never reach it.
    """
    last = units[-1]
    units[-1] = PromptUnit(
        unit_id=last.unit_id,
        start_s=min(last.start_s, extra.start_s),
        end_s=max(last.end_s, extra.end_s),
        text=(last.text.strip() + ' ' + extra.text.strip()).strip(),
    )


def merge_units(units: Sequence[TranscriptUnit], max_prompt_units: int) -> MergedTranscript:
    """Merge adjacent units pairwise until at most ``max_prompt_units`` remain.

    Each round merges (0,1), (2,3), ...; an odd trailing unit is kept as is.
    A merged unit keeps the FIRST id of the pair so ids stay stable across
    rounds and stay valid original ids. Rounds repeat (halving each time) until
    the bound holds, so the prompt size is bounded for any transcript length.

    A unit whose id already appeared (an upstream anomaly that
    ``check_units`` only reports) is folded into the preceding prompt unit,
    so prompt ids are unique and no transcript text is lost; the count is
    reported in ``diagnostics['duplicate_ids']``.
    """
    limit = max(1, int(max_prompt_units))
    current: List[PromptUnit] = []
    id_map: Dict[int, List[int]] = {}
    duplicates = 0
    for u in units:
        unit = PromptUnit(int(u.unit_id), _finite_or_zero(u.start_s), _finite_or_zero(u.end_s), _clean_text(u.text))
        if unit.unit_id in id_map:
            duplicates += 1
            _fold_into_previous(current, unit)
            continue
        id_map[unit.unit_id] = [unit.unit_id]
        current.append(unit)
    rounds = 0
    while len(current) > limit and len(current) > 1:
        rounds += 1
        merged: List[PromptUnit] = []
        for i in range(0, len(current), 2):
            first = current[i]
            if i + 1 >= len(current):
                merged.append(first)
                continue
            second = current[i + 1]
            merged.append(PromptUnit(
                unit_id=first.unit_id,
                start_s=min(first.start_s, second.start_s),
                end_s=max(first.end_s, second.end_s),
                text=(first.text.strip() + ' ' + second.text.strip()).strip(),
            ))
            id_map[first.unit_id] = id_map.pop(first.unit_id) + id_map.pop(second.unit_id)
        current = merged
    diagnostics = {
        'n_units_original': len(units),
        'n_units_prompt': len(current),
        'merge_rounds': rounds,
    }
    if duplicates:
        diagnostics['duplicate_ids'] = duplicates
    return MergedTranscript(units=current, id_map=id_map, diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

@dataclass
class PromptBundle:
    """Messages for one model call plus what is needed to read the answer."""

    messages: List[Dict[str, str]]
    merged: MergedTranscript

    @property
    def valid_ids(self) -> List[int]:
        return self.merged.valid_ids


def _data_block(units: Sequence[PromptUnit], questions: Sequence[str]) -> str:
    """The ONLY place untrusted text enters a message: as JSON string values.

    ``ensure_ascii=False`` keeps accents readable for the model; ``json.dumps``
    still escapes quotes, backslashes and control characters, so the block is
    one well-formed JSON document whatever the text contains.
    """
    payload = {
        'transcript': [
            {'id': u.unit_id, 'start': round(u.start_s, 2), 'end': round(u.end_s, 2), 'text': u.text}
            for u in units
        ],
        'questions': [{'q': i, 'text': _clean_text(q)} for i, q in enumerate(questions)],
    }
    return DATA_MARKER + json.dumps(payload, ensure_ascii=False, allow_nan=False)


def build_prompt(
    context: AudioContext,
    questions: Sequence[str],
    prompt_text: str,
    max_prompt_units: int = LLMConfig.max_prompt_units,
) -> PromptBundle:
    """Messages plus the id mapping (see :func:`build_messages`)."""
    merged = merge_units(context.units, max_prompt_units)
    messages = [
        {'role': 'system', 'content': prompt_text},
        {'role': 'user', 'content': _data_block(merged.units, questions)},
    ]
    return PromptBundle(messages=messages, merged=merged)


def build_messages(
    context: AudioContext,
    questions: Sequence[str],
    prompt_text: str,
    max_prompt_units: int = LLMConfig.max_prompt_units,
) -> List[Dict[str, str]]:
    """Two messages: the trusted prompt as ``system`` and one ``user`` message
    holding the transcript and questions as a JSON document. Nothing from the
    transcript or the questions ever appears outside that JSON string."""
    return build_prompt(context, questions, prompt_text, max_prompt_units).messages


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #

def _brace_matches(text: str, start: int) -> Dict[int, Optional[int]]:
    """Match every ``{`` from ``start`` onwards in ONE linear pass.

    Returns ``{open_index: index just past the closing '}'}`` with ``None``
    for an unclosed brace (truncated output). Braces inside JSON strings are
    skipped by tracking string state and escapes, so a ``}`` in a quoted value
    cannot close an object. Only braces the pass sees outside a string are
    recorded; for each of them the result equals what a fresh scan from that
    brace would find, because the string state at that point is the same
    (not in a string) and the text after it is identical. One pass per output
    replaces one full scan per brace, which was quadratic in the output
    length when many braces were unclosed.
    """
    matches: Dict[int, Optional[int]] = {}
    open_stack: List[int] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            open_stack.append(i)
        elif ch == '}' and open_stack:
            matches[open_stack.pop()] = i + 1
    for i in open_stack:
        matches[i] = None
    return matches


def _iter_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Every balanced ``{...}`` in ``text`` that parses as a JSON object, in
    order of appearance, nested ones included.

    Trying every ``{`` (not only the first) is what makes commentary such as
    ``Here is {my} answer: {"items": ...}`` and code fences harmless. Lazy, so
    the caller stops at the first usable object. A brace that a pass saw
    inside a string (a stray quote in commentary before the real object)
    starts a fresh pass from that brace, at most ``MAX_SCAN_PASSES`` times.
    ``RecursionError`` is what the C decoder raises for absurdly nested
    output such as a model stuck emitting ``[[[[``; it is as much "not an
    object" as a syntax error.
    """
    matches: Dict[int, Optional[int]] = {}
    passes = 0
    candidates = 0
    start = text.find('{')
    while start != -1 and candidates < MAX_JSON_CANDIDATES:
        if start not in matches:
            if passes >= MAX_SCAN_PASSES:
                return
            passes += 1
            matches.update(_brace_matches(text, start))
        candidates += 1
        end = matches[start]
        if end is not None:
            try:
                value = json.loads(text[start:end])
            except (ValueError, RecursionError):
                value = None
            if isinstance(value, dict):
                yield value
        start = text.find('{', start + 1)


def _is_int(value: Any) -> bool:
    """JSON ``true`` is a Python ``int``; ids must be real integers. Any
    integral type counts (a caller may hand over numpy ids), bool never."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _validate_item(item: Any, n_questions: int, valid_ids: set) -> Optional[Tuple[int, bool, List[int], List[Any]]]:
    """``(q, answer, units, dropped)`` for a well-formed item, else ``None``.

    Strictness matters here because the output is untrusted: ``"true"`` (a
    string), ``1`` and ``null`` are all rejected as answers so a model that
    drifts off the schema cannot flip a verdict by accident. Evidence ids that
    do not exist are dropped rather than fatal because a wrong id says nothing
    about the boolean, which is the part that scores.
    """
    if not isinstance(item, dict):
        return None
    q = item.get('q')
    if not _is_int(q) or not (0 <= q < n_questions):
        return None
    answer = item.get('answer')
    if not isinstance(answer, bool):
        return None
    raw_units = item.get('units', [])
    if not isinstance(raw_units, list):
        raw_units = []
    units: List[int] = []
    dropped: List[Any] = []
    for uid in raw_units:
        if not (_is_int(uid) and uid in valid_ids):
            dropped.append(uid)
        elif uid not in units:
            units.append(uid)
    if not answer:
        # The schema says false has no evidence; whatever came with it is noise.
        units = []
    return q, answer, units, dropped


def parse_llm_output(
    text: str,
    n_questions: int,
    valid_unit_ids: Sequence[int],
) -> Tuple[List[QAResult], List[int]]:
    """Strictly read ``{"items": [{"q", "answer", "units"}]}`` from model text.

    Returns the valid results sorted by question index and the indices that
    are missing or invalid. The caller decides what to do about the missing
    ones (repair or default); this function never invents an answer.

    The first balanced object with an ``items`` list wins. If there is none
    (typically output truncated by ``max_new_tokens``, or a bare array of
    items), complete item objects are salvaged individually so a partial
    answer is not thrown away.
    """
    valid_ids = {int(uid) for uid in valid_unit_ids if _is_int(uid)}
    items: Optional[List[Any]] = None
    salvage: List[Dict[str, Any]] = []
    for obj in _iter_json_objects(text if isinstance(text, str) else ''):
        if isinstance(obj.get('items'), list):
            items = obj['items']
            break
        if 'q' in obj:
            salvage.append(obj)
    if items is None:
        items = salvage

    by_q: Dict[int, QAResult] = {}
    for item in items:
        validated = _validate_item(item, n_questions, valid_ids)
        if validated is None:
            continue
        q, answer, units, dropped = validated
        if q in by_q:
            continue  # duplicates: keep the first valid item
        diagnostics: Dict[str, Any] = {}
        if dropped:
            diagnostics['evidence_dropped'] = dropped[:20]
        by_q[q] = QAResult(question_index=q, answer=answer, evidence_unit_ids=units, diagnostics=diagnostics)

    results = [by_q[q] for q in sorted(by_q)]
    missing = [q for q in range(n_questions) if q not in by_q]
    return results, missing


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #

class FakeGenerate:
    """Scripted engine for tests: returns ``responses`` in order and records
    every call. Once exhausted it returns an empty string, which the parser
    treats as "nothing answered", so a test can also exercise the defaults."""

    def __init__(self, responses: Sequence[str]) -> None:
        self.responses: Deque[str] = deque(str(r) for r in responses)
        self.calls: List[Tuple[List[Dict[str, str]], int]] = []

    def __call__(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        self.calls.append(([dict(m) for m in messages], int(max_new_tokens)))
        if not self.responses:
            logger.warning('fake LLM engine exhausted after %d calls', len(self.calls))
            return ''
        return self.responses.popleft()


def _build_transformers_generate(config: LLMConfig) -> GenerateFn:
    """Greedy chat generation with HF transformers; imports torch lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ``LLMConfig`` already validated the name against SUPPORTED_DTYPES.
    dtype = 'auto' if config.dtype == 'auto' else getattr(torch, config.dtype)
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.n_threads:
        torch.set_num_threads(int(config.n_threads))

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=dtype)
    model.to(device)
    model.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def generate(messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        # The chat template is the model's own; the data stays inside the
        # message content, so the template never sees raw transcript text.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=pad_token_id,
            )
        new_tokens = output[0, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def _build_llama_cpp_generate(config: LLMConfig) -> GenerateFn:
    """Deterministic JSON-mode chat completion with llama.cpp."""
    if not config.model_path:
        raise ValueError('llama_cpp engine needs LLMConfig.model_path (a GGUF file)')
    from llama_cpp import Llama

    llm = Llama(
        model_path=config.model_path,
        n_ctx=int(config.n_ctx),
        n_threads=config.n_threads,
        verbose=False,
    )

    def generate(messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        completion = llm.create_chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=int(max_new_tokens),
            response_format={'type': 'json_object'},
        )
        choices = completion.get('choices') or []
        if not choices:
            return ''
        return str((choices[0].get('message') or {}).get('content') or '')

    return generate


def build_generate(config: LLMConfig) -> GenerateFn:
    """The engine's ``generate`` callable for ``config.engine``."""
    if config.engine == 'fake':
        return FakeGenerate(config.fake_responses)
    if config.engine == 'transformers':
        return _build_transformers_generate(config)
    if config.engine == 'llama_cpp':
        return _build_llama_cpp_generate(config)
    raise ValueError(f'unknown LLM engine {config.engine!r}')  # pragma: no cover - guarded in LLMConfig


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #

class LLMBackend:
    """``QABackend`` that asks a local LLM for a JSON verdict per question.

    ``generate`` may be injected (tests, or an engine this module does not
    know); otherwise it is built lazily from ``config.engine`` on first use so
    that constructing the backend never loads a model.
    """

    name = 'llm'

    def __init__(
        self,
        config: LLMConfig,
        generate: Optional[GenerateFn] = None,
        prompt_text: Optional[str] = None,
    ) -> None:
        self.config = config
        self.prompt_text = prompt_text if prompt_text is not None else load_prompt(config.prompt_path)
        self.prompt_hash = prompt_sha256(self.prompt_text)
        self._generate: Optional[GenerateFn] = generate
        self.warmed_up = False
        self.last_run_info: Dict[str, Any] = {}

    # -- lifecycle -------------------------------------------------------- #

    @property
    def generate(self) -> GenerateFn:
        """The engine callable, built on first access."""
        if self._generate is None:
            self._generate = build_generate(self.config)
        return self._generate

    def warm_up(self) -> None:
        """Load the model and run one tiny generation so the first request
        pays neither the import nor the lazy-initialisation cost.

        The fake engine only loads: its scripted responses belong to the
        request that a test is about to make, not to the warm-up.
        """
        generate = self.generate
        if self.config.engine != 'fake':
            context = AudioContext(audio_sha256='warmup', duration_s=1.0, units=[
                TranscriptUnit(unit_id=0, start_s=0.0, end_s=1.0, text='Hello, how are you today?'),
            ])
            messages = build_messages(context, ['Did the clinician greet the patient?'], self.prompt_text, 1)
            generate(messages, min(32, self.config.max_new_tokens))
        self.warmed_up = True

    # -- answering -------------------------------------------------------- #

    def answer(
        self,
        context: AudioContext,
        questions: List[str],
        deadline: Optional[Deadline] = None,
    ) -> List[QAResult]:
        """Exactly one :class:`QAResult` per question, in order, always.

        Batched mode: one call for every question, then at most one repair
        call per unanswered question while the deadline allows. Per-question
        mode: one call per question; every call after the first is checked
        against the deadline. The first call is never gated on
        ``repair_min_remaining_s`` (the pipeline decides whether this backend
        gets invoked at all), but an already expired deadline makes no call and
        returns flagged defaults, because a late answer scores nothing.
        A model failure before any valid answer exists propagates so the
        pipeline can fall back to another backend; a failure after that is
        logged and only affects the questions it happened for.
        """
        n = len(questions)
        info: Dict[str, Any] = {'calls': 0, 'repair_calls': 0, 'skipped_for_time': 0, 'errors': []}
        self.last_run_info = info
        if n == 0:
            return []
        if not context.units:
            # Nothing can be established by an empty consultation; no model call.
            info['empty_transcript'] = True
            return [self._default_result(q, {'missing': True, 'empty_transcript': True}) for q in range(n)]
        if deadline is not None and deadline.expired():
            info['skipped_for_time'] = n
            return [self._default_result(q, {'missing': True, 'skipped_for_time': True}) for q in range(n)]

        bundle = build_prompt(context, questions, self.prompt_text, self.config.max_prompt_units)
        info['merge'] = dict(bundle.merged.diagnostics)
        found: Dict[int, QAResult] = {}

        if self.config.batch_all_questions:
            text = self._call(bundle.messages, self.config.max_new_tokens, info)
            results, missing = parse_llm_output(text, n, bundle.valid_ids)
            found = {r.question_index: r for r in results}
            self._repair(missing, questions, bundle, deadline, found, info)
        else:
            for q in range(n):
                if q and not self._time_allows(deadline):
                    for skipped in range(q, n):
                        found[skipped] = self._skipped_result(skipped, info)
                    break
                try:
                    result = self._ask_one(q, questions, bundle, info)
                except Exception as exc:
                    found[q] = self._call_failed(q, exc, found, info)
                    continue
                if result is not None:
                    found[q] = result

        return [self._finish(q, found.get(q), bundle.merged) for q in range(n)]

    # -- helpers ----------------------------------------------------------- #

    def _time_allows(self, deadline: Optional[Deadline]) -> bool:
        """Whether an ADDITIONAL generation may start: strictly more than
        ``repair_min_remaining_s`` must remain once the deadline's own
        ``reserve_s`` is set aside, the same budget the other QA backend
        respects through ``Deadline.fits``; no deadline means no limit."""
        if deadline is None:
            return True
        return deadline.remaining() - deadline.reserve_s > self.config.repair_min_remaining_s

    def _skipped_result(self, q: int, info: Dict[str, Any]) -> QAResult:
        """Default for a question whose call was not started for lack of time;
        flagged so the caller can tell "the model said no" from "never asked"."""
        info['skipped_for_time'] += 1
        return QAResult(question_index=q, answer=False, diagnostics={'missing': True, 'skipped_for_time': True})

    def _call(self, messages: List[Dict[str, str]], max_new_tokens: int, info: Dict[str, Any]) -> str:
        info['calls'] += 1
        text = self.generate(messages, max_new_tokens)
        return text if isinstance(text, str) else ''

    def _ask_one(
        self,
        q: int,
        questions: Sequence[str],
        bundle: PromptBundle,
        info: Dict[str, Any],
    ) -> Optional[QAResult]:
        """One call with a single-question data block for question ``q``.

        The block numbers its only question 0, so the result is re-indexed.
        Same merged transcript as the batched call, so evidence ids resolve
        the same way. ``None`` when the model did not answer it.
        """
        messages = [
            {'role': 'system', 'content': self.prompt_text},
            {'role': 'user', 'content': _data_block(bundle.merged.units, [questions[q]])},
        ]
        text = self._call(messages, min(REPAIR_MAX_NEW_TOKENS, self.config.max_new_tokens), info)
        results, _ = parse_llm_output(text, 1, bundle.valid_ids)
        if not results:
            return None
        result = results[0]
        result.question_index = q
        return result

    def _call_failed(self, q: int, exc: Exception, found: Dict[int, QAResult], info: Dict[str, Any]) -> QAResult:
        """A model failure with nothing valid to preserve propagates (the
        pipeline falls back to another backend); after a valid result exists
        it is logged and only the failed question gets the default."""
        if not found:
            raise exc
        logger.exception('LLM call for question %d failed; keeping earlier results', q)
        message = f'{type(exc).__name__}: {exc}'
        info['errors'].append(f'q{q}: {message}')
        return QAResult(question_index=q, answer=False, diagnostics={'missing': True, 'error': message})

    def _repair(
        self,
        missing: Sequence[int],
        questions: Sequence[str],
        bundle: PromptBundle,
        deadline: Optional[Deadline],
        found: Dict[int, QAResult],
        info: Dict[str, Any],
    ) -> None:
        """At most ONE extra call per missing question, deadline permitting.

        The deadline is re-checked before every call because each generation
        costs seconds; a repair that would push the request past its budget is
        worth less than a default answer delivered on time.
        """
        for q in missing:
            if not self._time_allows(deadline):
                found[q] = self._skipped_result(q, info)
                continue
            info['repair_calls'] += 1
            try:
                result = self._ask_one(q, questions, bundle, info)
            except Exception as exc:
                found[q] = self._call_failed(q, exc, found, info)
                continue
            if result is not None:
                result.diagnostics['repaired'] = True
                found[q] = result

    def _default_result(self, q: int, diagnostics: Dict[str, Any]) -> QAResult:
        return QAResult(
            question_index=q,
            answer=False,
            evidence_unit_ids=[],
            confidence=0.0,
            backend=self.name,
            diagnostics=self._stamp(diagnostics),
        )

    def _stamp(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Provenance every result carries so a run can be reproduced."""
        stamped = dict(diagnostics)
        stamped.setdefault('engine', self.config.engine)
        stamped.setdefault('model_id', self.config.model_path if self.config.engine == 'llama_cpp' else self.config.model_id)
        stamped.setdefault('prompt_sha256', self.prompt_hash)
        return stamped

    def _finish(self, q: int, result: Optional[QAResult], merged: MergedTranscript) -> QAResult:
        """Resolve prompt ids to original unit ids and stamp provenance."""
        if result is None:
            return self._default_result(q, {'missing': True})
        result.question_index = q
        result.evidence_unit_ids = merged.expand(result.evidence_unit_ids)
        result.backend = self.name
        result.diagnostics = self._stamp(result.diagnostics)
        return result
