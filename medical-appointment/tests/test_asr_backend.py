"""Tests for ``asr_backend``. Everything but the last test runs without a
model: faster-whisper objects are duck-typed stubs and time is a FakeClock."""

import io
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from asr_backend import (
    _ENV_PARSERS,
    REQUIRED_SAMPLE_RATE,
    ASRConfig,
    CachingASRBackend,
    FakeASRBackend,
    FakeClock,
    FasterWhisperBackend,
    build_asr_backend,
    resolve_compute_type,
    resolve_device,
    segments_to_units,
    units_from_dicts,
    units_to_dicts,
)
from core_types import Deadline, TranscriptUnit, check_units
from tests.conftest import make_context

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MP3 = ROOT / 'data' / 'audio' / 'conversation_sample_4.mp3'


# --------------------------------------------------------------------------- #
# Stub builders
# --------------------------------------------------------------------------- #

def stub_word(start, end, word, probability=0.9):
    return SimpleNamespace(start=start, end=end, word=word, probability=probability)


def stub_segment(start, end, text, words=None, avg_logprob=-0.2, no_speech_prob=0.01):
    return SimpleNamespace(
        id=0, start=start, end=end, text=text, words=words,
        avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
    )


def two_segments():
    return [
        stub_segment(0.0, 1.0, ' Hello there.', [
            stub_word(0.0, 0.4, ' Hello', 0.95),
            stub_word(0.4, 1.0, ' there.', 0.8),
        ]),
        stub_segment(1.2, 2.5, ' Take one pill.', [
            stub_word(1.2, 1.5, ' Take', 0.9),
            stub_word(1.5, 1.9, ' one', 0.7),
            stub_word(1.9, 2.5, ' pill.', 0.85),
        ]),
    ]


def scripted_units(n=5):
    return make_context([f'unit number {i}' for i in range(n)]).units


class StubModel:
    """Stands in for ``faster_whisper.WhisperModel``: records calls, yields stubs."""

    def __init__(self, segments, language='en'):
        self.segments = segments
        self.language = language
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({'audio': audio, **kwargs})
        info = SimpleNamespace(language=self.language, language_probability=0.99)
        return iter(self.segments), info


def waveform(seconds=1.0):
    return np.zeros(int(seconds * REQUIRED_SAMPLE_RATE), dtype=np.float32)


# --------------------------------------------------------------------------- #
# segments_to_units
# --------------------------------------------------------------------------- #

def test_segments_to_units_keeps_raw_text_and_global_word_ids():
    units = segments_to_units(two_segments())

    assert [u.unit_id for u in units] == [0, 1]
    assert units[0].text == ' Hello there.'
    assert [w.text for w in units[0].words] == [' Hello', ' there.']
    assert [w.word_id for u in units for w in u.words] == [0, 1, 2, 3, 4]
    assert units[1].words[1].probability == 0.7
    assert (units[1].start_s, units[1].end_s) == (1.2, 2.5)
    assert units[0].avg_logprob == -0.2 and units[0].no_speech_prob == 0.01
    assert check_units(units) == []


def test_segments_to_units_words_none_gives_empty_words():
    units = segments_to_units([stub_segment(3.0, 4.0, ' No words here.', words=None)])

    assert len(units) == 1
    assert units[0].words == []
    assert (units[0].start_s, units[0].end_s) == (3.0, 4.0)


def test_segments_to_units_empty_and_generator_input():
    assert segments_to_units([]) == []
    assert segments_to_units(iter([])) == []
    assert len(segments_to_units(iter(two_segments()))) == 2


def test_segments_to_units_handles_non_finite_values():
    nan = float('nan')
    segments = [
        # A NaN word is dropped from words, the rest keep consecutive ids.
        stub_segment(0.0, 1.0, ' a b c', [
            stub_word(0.0, 0.3, ' a'), stub_word(nan, 0.6, ' b'), stub_word(0.6, 1.0, ' c', nan),
        ]),
        # NaN bounds but finite words: bounds come from the words.
        stub_segment(nan, nan, ' d e', [stub_word(2.0, 2.4, ' d'), stub_word(2.4, 2.9, ' e')]),
        # NaN bounds and nothing to fall back on: dropped.
        stub_segment(nan, 5.0, ' ghost', None),
        # Missing probability attribute and None no_speech_prob are tolerated.
        SimpleNamespace(start=6.0, end=6.5, text=' f', words=[SimpleNamespace(start=6.0, end=6.5, word=' f')]),
    ]
    units = segments_to_units(segments)

    assert [u.unit_id for u in units] == [0, 1, 2]
    assert [w.text for w in units[0].words] == [' a', ' c']
    assert units[0].words[1].probability is None
    assert [w.word_id for u in units for w in u.words] == [0, 1, 2, 3, 4]
    assert (units[1].start_s, units[1].end_s) == (2.0, 2.9)
    assert units[2].words[0].probability is None
    assert units[2].avg_logprob is None and units[2].no_speech_prob is None
    assert check_units(units) == []


def test_segments_to_units_clamps_reversed_bounds_and_skips_blank():
    units = segments_to_units([
        stub_segment(2.0, 1.5, ' backwards', [stub_word(2.0, 1.8, ' backwards')]),
        stub_segment(3.0, 4.0, '   ', None),
        stub_segment(4.0, 5.0, '', []),
        stub_segment(5.0, 6.0, ' ok', None),
    ])

    assert [u.text for u in units] == [' backwards', ' ok']
    assert (units[0].start_s, units[0].end_s) == (2.0, 2.0)
    assert (units[0].words[0].start_s, units[0].words[0].end_s) == (2.0, 2.0)
    assert units[1].unit_id == 1


def test_segments_to_units_id_offsets_allow_incremental_use():
    first = segments_to_units(two_segments()[:1])
    second = segments_to_units(two_segments()[1:], first_unit_id=1, first_word_id=2)

    assert second[0].unit_id == 1
    assert [w.word_id for w in second[0].words] == [2, 3, 4]
    assert check_units(first + second) == []


def test_units_round_trip_through_dicts():
    units = segments_to_units(two_segments())
    restored = units_from_dicts(json.loads(json.dumps(units_to_dicts(units))))

    assert restored == units
    with pytest.raises(TypeError):
        units_from_dicts({'not': 'a list'})


# --------------------------------------------------------------------------- #
# ASRConfig
# --------------------------------------------------------------------------- #

def test_config_hash_is_stable_and_tracks_output_fields_only():
    base = ASRConfig()

    assert base.config_hash == ASRConfig().config_hash
    assert len(base.config_hash) == 64
    assert ASRConfig(model_size='base').config_hash != base.config_hash
    assert ASRConfig(beam_size=5).config_hash != base.config_hash
    assert ASRConfig(initial_prompt='Clinic visit.').config_hash != base.config_hash
    assert ASRConfig(compute_type='float32').config_hash != base.config_hash
    # Operational knobs must not invalidate cached transcripts.
    assert ASRConfig(cpu_threads=2, num_workers=2, stop_reserve_s=1.0,
                     download_root='/x', local_files_only=True).config_hash == base.config_hash
    assert base.model_id == 'faster-whisper:small'


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        ASRConfig(beam_size=0)
    with pytest.raises(ValueError):
        ASRConfig(stop_reserve_s=-1.0)
    with pytest.raises(ValueError):
        ASRConfig(temperature=float('nan'))
    with pytest.raises(ValueError):
        ASRConfig(model_size='')
    with pytest.raises(ValueError):
        ASRConfig(num_workers=0)


def test_from_env_parses_types_and_ignores_other_variables():
    environ = {
        'MA_ASR_MODEL_SIZE': 'base',
        'MA_ASR_BEAM_SIZE': ' 3 ',
        'MA_ASR_VAD_FILTER': 'true',
        'MA_ASR_LOCAL_FILES_ONLY': '0',
        'MA_ASR_CPU_THREADS': '2',
        'MA_ASR_STOP_RESERVE_S': '5.5',
        'MA_ASR_INITIAL_PROMPT': '',
        'MA_ASR_DOWNLOAD_ROOT': '/models/whisper',
        'MA_ASR_DEVICE': 'cpu',
        'MA_ASR_COMPUTE_TYPE': 'int8',
        'MA_QA_MODEL': 'not-for-us',
        'OTHER_MA_ASR_BEAM_SIZE': '99',
    }
    config = ASRConfig.from_env(environ=environ)

    assert config.model_size == 'base'
    assert config.beam_size == 3 and isinstance(config.beam_size, int)
    assert config.vad_filter is True
    assert config.local_files_only is False
    assert config.cpu_threads == 2
    assert config.stop_reserve_s == 5.5
    assert config.initial_prompt is None
    assert config.download_root == '/models/whisper'
    assert (config.device, config.compute_type) == ('cpu', 'int8')
    # Untouched fields keep their defaults.
    assert config.word_timestamps is True and config.language == 'en'


def test_from_env_defaults_prefix_and_base():
    assert ASRConfig.from_env(environ={}) == ASRConfig()
    custom = ASRConfig.from_env(prefix='X_', environ={'X_BEAM_SIZE': '2'}, base=ASRConfig(model_size='tiny'))
    assert (custom.model_size, custom.beam_size) == ('tiny', 2)


@pytest.mark.parametrize('key, value', [
    ('MA_ASR_BEAM_SIZE', 'three'),
    ('MA_ASR_VAD_FILTER', 'maybe'),
    ('MA_ASR_STOP_RESERVE_S', 'inf'),
    ('MA_ASR_MODEL_SIZE', '  '),
    ('MA_ASR_BEAM_SIZE', '0'),
])
def test_from_env_rejects_bad_values_naming_the_variable(key, value):
    with pytest.raises(ValueError, match=key):
        ASRConfig.from_env(environ={key: value})


def test_every_config_field_is_readable_from_env():
    from dataclasses import fields

    assert {f.name for f in fields(ASRConfig)} == set(_ENV_PARSERS)


def test_resolve_device_and_compute_type():
    assert resolve_device('cpu') == 'cpu'
    assert resolve_device('cuda', cuda_device_count=lambda: 0) == 'cuda'
    assert resolve_device('auto', cuda_device_count=lambda: 0) == 'cpu'
    assert resolve_device('auto', cuda_device_count=lambda: 2) == 'cuda'

    def broken():
        raise RuntimeError('no driver')

    assert resolve_device('auto', cuda_device_count=broken) == 'cpu'
    assert resolve_compute_type('auto', 'cpu') == 'int8'
    assert resolve_compute_type('auto', 'cuda') == 'float16'
    assert resolve_compute_type('float32', 'cuda') == 'float32'
    assert resolve_device('auto') in ('cpu', 'cuda')


# --------------------------------------------------------------------------- #
# FakeASRBackend
# --------------------------------------------------------------------------- #

def test_fake_backend_returns_copies_and_records_calls():
    script = scripted_units(3)
    backend = FakeASRBackend(script)
    backend.warm_up()

    out = backend.transcribe(waveform(2.0), REQUIRED_SAMPLE_RATE, cache_key='abc')
    out[0].text = 'mutated'

    assert backend.warmed_up
    assert backend.transcribe(waveform(2.0), REQUIRED_SAMPLE_RATE) == script
    assert backend.calls[0] == {'n_samples': 32000, 'sample_rate': 16000, 'cache_key': 'abc'}
    assert backend.last_run_info['units'] == 3
    assert backend.last_run_info['truncated'] is False
    assert (backend.model_id, backend.config_hash) == ('fake', 'fake')


def test_fake_backend_checks_sample_rate_and_raises_scripted_error():
    backend = FakeASRBackend(scripted_units(1))
    with pytest.raises(ValueError, match='16000'):
        backend.transcribe(waveform(), 44100)

    failing = FakeASRBackend(scripted_units(1), error=RuntimeError('boom'))
    with pytest.raises(RuntimeError, match='boom'):
        failing.transcribe(waveform(), REQUIRED_SAMPLE_RATE)
    assert len(failing.calls) == 1


def test_fake_backend_truncates_on_deadline_without_sleeping():
    clock = FakeClock()
    backend = FakeASRBackend(scripted_units(5), delay_s=3.0, clock=clock, stop_reserve_s=4.0)
    deadline = Deadline(budget_s=10.0, clock=clock)

    units = backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE, deadline=deadline)

    # Pulls at t=0, 3, 6 are allowed (remaining 10, 7, 4); at t=9 remaining 1 < 4.
    assert [u.unit_id for u in units] == [0, 1, 2]
    assert backend.last_run_info['truncated'] is True
    assert backend.last_run_info['seconds'] == 9.0
    assert clock() == 9.0


def test_fake_backend_without_deadline_returns_everything():
    clock = FakeClock()
    backend = FakeASRBackend(scripted_units(4), delay_s=100.0, clock=clock)
    assert len(backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE)) == 4
    assert clock() == 400.0


# --------------------------------------------------------------------------- #
# CachingASRBackend
# --------------------------------------------------------------------------- #

def test_caching_miss_then_hit_serves_stored_units(tmp_path):
    first = FakeASRBackend(scripted_units(3))
    cache = CachingASRBackend(first, tmp_path)

    out1 = cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='a' * 64)
    path = tmp_path / first.config_hash / ('a' * 64 + '.json')

    assert path.is_file()
    assert (cache.hits, cache.misses) == (0, 1)
    assert cache.last_run_info['cache'] == 'miss'
    payload = json.loads(path.read_text())
    assert payload['model_id'] == 'fake' and len(payload['units']) == 3

    # A different inner backend (same hash) must not be consulted on a hit.
    second = FakeASRBackend(scripted_units(1))
    cache2 = CachingASRBackend(second, tmp_path)
    out2 = cache2.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='a' * 64)

    assert out2 == out1
    assert second.calls == []
    assert (cache2.hits, cache2.misses) == (1, 0)
    assert cache2.last_run_info['cache'] == 'hit'
    assert cache2.last_run_info['truncated'] is False


def test_caching_passes_through_without_key(tmp_path):
    inner = FakeASRBackend(scripted_units(2))
    cache = CachingASRBackend(inner, tmp_path)

    assert len(cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE)) == 2
    assert list(tmp_path.iterdir()) == []
    assert cache.last_run_info['cache'] == 'bypass'
    assert inner.calls[0]['cache_key'] is None


@pytest.mark.parametrize('garbage', [
    'not json at all',
    '{"format": 1, "units": "nope"}',
    '{"format": 1, "units": [{"unit_id": "x"}]}',
    '{"format": 1, "units": [{"unit_id": 0, "start_s": NaN, "end_s": 1, "text": "a"}]}',
    '{"format": 99, "units": []}',
    '[]',
    # 1e400 parses to inf without going through parse_constant.
    '{"format": 1, "units": [{"unit_id": 0, "start_s": 0, "end_s": 1e400, "text": "a"}]}',
    '{"format": 1, "units": [{"unit_id": 0, "start_s": 0, "end_s": 1, "text": "a", "avg_logprob": "junk"}]}',
])
def test_caching_ignores_and_overwrites_corrupt_file(tmp_path, garbage):
    inner = FakeASRBackend(scripted_units(2))
    cache = CachingASRBackend(inner, tmp_path)
    path = cache.cache_path('key1')
    path.parent.mkdir(parents=True)
    path.write_text(garbage)

    out = cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='key1')

    assert len(out) == 2 and len(inner.calls) == 1
    assert (cache.hits, cache.misses) == (0, 1)
    assert units_from_dicts(json.loads(path.read_text())['units']) == out


def test_caching_skips_truncated_and_failed_runs(tmp_path):
    clock = FakeClock()
    inner = FakeASRBackend(scripted_units(5), delay_s=3.0, clock=clock, stop_reserve_s=4.0)
    cache = CachingASRBackend(inner, tmp_path)
    units = cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE,
                             deadline=Deadline(10.0, clock=clock), cache_key='trunc')

    assert len(units) == 3
    assert not cache.cache_path('trunc').exists()
    assert cache.last_run_info['truncated'] is True

    class ErroringInner(FakeASRBackend):
        def transcribe(self, *args, **kwargs):
            units = super().transcribe(*args, **kwargs)
            self.last_run_info['error'] = 'RuntimeError: mid-stream'
            return units

    erroring = CachingASRBackend(ErroringInner(scripted_units(2)), tmp_path)
    assert len(erroring.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='err')) == 2
    assert not erroring.cache_path('err').exists()


@pytest.mark.parametrize('bad_key', ['../escape', 'a/b', '.hidden', '', 'x' * 129, 'sp ace'])
def test_caching_rejects_unsafe_keys(tmp_path, bad_key):
    cache = CachingASRBackend(FakeASRBackend(scripted_units(1)), tmp_path)
    with pytest.raises(ValueError):
        cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key=bad_key)
    assert list(tmp_path.iterdir()) == []


def test_caching_does_not_store_non_finite_units(tmp_path):
    broken = [TranscriptUnit(unit_id=0, start_s=0.0, end_s=float('nan'), text='x')]
    cache = CachingASRBackend(FakeASRBackend(broken), tmp_path)

    out = cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='nan')

    assert len(out) == 1 and math.isnan(out[0].end_s)
    assert not cache.cache_path('nan').exists()


def test_caching_warm_up_delegates(tmp_path):
    inner = FakeASRBackend([])
    CachingASRBackend(inner, tmp_path).warm_up()
    assert inner.warmed_up


# --------------------------------------------------------------------------- #
# FasterWhisperBackend with stubs
# --------------------------------------------------------------------------- #

def make_backend(segments, clock=None, **config_kwargs):
    config = ASRConfig(device='cpu', compute_type='int8', **config_kwargs)
    model = StubModel(segments)
    factories = []

    def factory(cfg, device, compute_type):
        factories.append((cfg, device, compute_type))
        return model

    backend = FasterWhisperBackend(config, model_factory=factory, **({'clock': clock} if clock else {}))
    return backend, model, factories


def test_faster_whisper_backend_passes_config_to_model_and_converts():
    backend, model, factories = make_backend(two_segments(), beam_size=2, initial_prompt='Clinic.')
    audio = waveform(2.5)
    audio[10] = np.nan

    units = backend.transcribe(audio, REQUIRED_SAMPLE_RATE, cache_key='ignored')

    assert factories == [(backend.config, 'cpu', 'int8')]
    call = model.calls[0]
    assert call['language'] == 'en' and call['beam_size'] == 2
    assert call['word_timestamps'] is True and call['vad_filter'] is False
    assert call['condition_on_previous_text'] is False and call['temperature'] == 0.0
    assert call['initial_prompt'] == 'Clinic.'
    assert call['audio'].dtype == np.float32 and np.all(np.isfinite(call['audio']))
    assert [u.text for u in units] == [' Hello there.', ' Take one pill.']
    info = backend.last_run_info
    assert info['segments'] == 2 and info['units'] == 2
    assert info['truncated'] is False and info['error'] is None
    assert info['language'] == 'en' and info['language_probability'] == 0.99
    assert info['audio_s'] == 2.5 and info['device'] == 'cpu'
    assert backend.load_seconds is not None and backend.load_seconds >= 0.0


def test_faster_whisper_backend_rejects_bad_input_without_loading():
    backend, model, factories = make_backend(two_segments())

    with pytest.raises(ValueError, match='16000'):
        backend.transcribe(waveform(), 44100)
    with pytest.raises(ValueError, match='1-D'):
        backend.transcribe(np.zeros((2, 100), dtype=np.float32), REQUIRED_SAMPLE_RATE)
    assert factories == [] and model.calls == []


def test_faster_whisper_backend_empty_waveform_short_circuits():
    backend, model, factories = make_backend(two_segments())

    assert backend.transcribe(np.zeros(0, dtype=np.float32), REQUIRED_SAMPLE_RATE) == []
    assert backend.last_run_info['audio_s'] == 0.0
    assert backend.last_run_info['truncated'] is False
    assert factories == [] and model.calls == []


def test_faster_whisper_backend_truncates_on_deadline(monkeypatch):
    clock = FakeClock()
    backend, _, _ = make_backend([], clock=clock, stop_reserve_s=4.0)
    pulled = []

    def slow_segments(audio):
        for segment in two_segments() * 3:  # 6 segments, 3 s each
            clock.advance(3.0)
            pulled.append(segment)
            yield segment

    monkeypatch.setattr(backend, '_segment_iter', slow_segments)
    deadline = Deadline(budget_s=10.0, clock=clock)

    units = backend.transcribe(waveform(60.0), REQUIRED_SAMPLE_RATE, deadline=deadline)

    assert len(units) == 3 and len(pulled) == 3
    assert [w.word_id for u in units for w in u.words] == [0, 1, 2, 3, 4, 5, 6]
    info = backend.last_run_info
    assert info['truncated'] is True and info['segments'] == 3
    assert info['seconds'] == 9.0 and info['error'] is None


def test_faster_whisper_backend_returns_partial_units_on_mid_stream_error(monkeypatch):
    backend, _, _ = make_backend([])

    def failing_segments(audio):
        yield from two_segments()
        raise RuntimeError('decoder exploded')

    monkeypatch.setattr(backend, '_segment_iter', failing_segments)

    units = backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE)

    assert [u.unit_id for u in units] == [0, 1]
    assert backend.last_run_info['error'] == 'RuntimeError: decoder exploded'
    assert backend.last_run_info['truncated'] is False


def test_faster_whisper_backend_model_load_failure_is_contained():
    def broken_factory(cfg, device, compute_type):
        raise OSError('weights missing')

    backend = FasterWhisperBackend(ASRConfig(device='cpu'), model_factory=broken_factory)

    assert backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE) == []
    assert backend.last_run_info['error'] == 'OSError: weights missing'
    with pytest.raises(OSError):
        backend.warm_up()


def test_faster_whisper_backend_warm_up_loads_once():
    backend, model, factories = make_backend(two_segments())
    backend.warm_up()
    backend.warm_up()
    backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE)

    assert len(factories) == 1
    assert len(model.calls) == 3
    assert len(model.calls[0]['audio']) == REQUIRED_SAMPLE_RATE


def test_build_asr_backend(tmp_path):
    config = ASRConfig(device='cpu')
    plain = build_asr_backend(config)
    cached = build_asr_backend(config, cache_dir=tmp_path)

    assert isinstance(plain, FasterWhisperBackend)
    assert isinstance(cached, CachingASRBackend) and isinstance(cached.inner, FasterWhisperBackend)
    assert cached.config_hash == plain.config_hash == config.config_hash
    assert cached.model_id == config.model_id


# --------------------------------------------------------------------------- #
# Regression tests (review)
# --------------------------------------------------------------------------- #

def test_from_env_rejects_empty_boolean_instead_of_silently_disabling():
    # ``MA_ASR_WORD_TIMESTAMPS=`` used to parse as False and switch word
    # timings off for every request; it must fail at start-up instead.
    with pytest.raises(ValueError, match='MA_ASR_WORD_TIMESTAMPS'):
        ASRConfig.from_env(environ={'MA_ASR_WORD_TIMESTAMPS': ''})


@pytest.mark.parametrize('kwargs', [
    {'beam_size': True}, {'beam_size': '3'}, {'cpu_threads': 2.5}, {'vad_filter': 'yes'},
    {'model_size': 5}, {'language': ''}, {'temperature': True}, {'stop_reserve_s': '8'},
])
def test_config_rejects_wrong_types_at_construction(kwargs):
    with pytest.raises(ValueError):
        ASRConfig(**kwargs)


def test_config_accepts_integers_for_float_fields():
    # JSON configs and the pipeline's env overrides may hand over 12, not 12.0.
    config = ASRConfig(stop_reserve_s=12, temperature=0)
    assert config.stop_reserve_s == 12 and config.temperature == 0


def test_drain_stops_on_expired_deadline_even_with_zero_reserve():
    clock = FakeClock()
    backend = FakeASRBackend(scripted_units(4), delay_s=5.0, clock=clock, stop_reserve_s=0.0)
    deadline = Deadline(budget_s=10.0, clock=clock)

    units = backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE, deadline=deadline)

    # Pulls at t=0 and t=5; at t=10 the budget is exactly spent, so no third pull.
    assert [u.unit_id for u in units] == [0, 1]
    assert backend.last_run_info['truncated'] is True
    assert clock() == 10.0


def test_fake_backend_accepts_plain_clock_callable():
    backend = FakeASRBackend(scripted_units(2), delay_s=0.001, clock=time.monotonic)

    units = backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE, deadline=Deadline(5.0))

    assert len(units) == 2
    assert backend.last_run_info['error'] is None
    assert backend.last_run_info['seconds'] >= 0.002


def test_fake_backend_records_error_in_run_info_before_raising():
    backend = FakeASRBackend(scripted_units(2))
    backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE)
    backend.error = RuntimeError('boom')

    with pytest.raises(RuntimeError):
        backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE)

    assert backend.last_run_info['error'] == 'RuntimeError: boom'
    assert backend.last_run_info['units'] == 0


@pytest.mark.parametrize('raw', [
    [{'unit_id': 0, 'start_s': 0, 'end_s': float('inf'), 'text': 'a'}],
    [{'unit_id': 0, 'start_s': 0, 'end_s': 1, 'text': 'a', 'avg_logprob': 'junk'}],
    [{'unit_id': 0, 'start_s': 0, 'end_s': 1, 'text': 'a', 'no_speech_prob': float('nan')}],
    [{'unit_id': 0, 'start_s': 0, 'end_s': 1, 'text': 'a', 'speaker': 5}],
    [{'unit_id': 0, 'start_s': 0, 'end_s': 1, 'text': None}],
    [{'unit_id': 0, 'start_s': 0, 'end_s': 1, 'text': 'a', 'words': 'abc'}],
    [{'unit_id': 0, 'start_s': 0, 'end_s': 1, 'text': 'a',
      'words': [{'word_id': 0, 'start_s': 0, 'end_s': 1, 'text': 'a', 'probability': float('inf')}]}],
    [{'unit_id': 0, 'start_s': 0, 'end_s': 1, 'text': 'a',
      'words': [{'word_id': 0, 'start_s': 0, 'end_s': 1, 'text': None}]}],
    [None],
    [[1, 2]],
])
def test_units_from_dicts_rejects_malformed_values(raw):
    with pytest.raises((ValueError, TypeError)):
        units_from_dicts(raw)


def test_write_cache_leaves_no_temp_file_on_failure(tmp_path, monkeypatch):
    import asr_backend

    def failing_replace(src, dst):
        raise OSError('disk full')

    monkeypatch.setattr(asr_backend.os, 'replace', failing_replace)
    cache = CachingASRBackend(FakeASRBackend(scripted_units(1)), tmp_path)

    units = cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='k')

    assert len(units) == 1
    assert [p for p in tmp_path.rglob('*') if p.is_file()] == []


def test_caching_run_info_is_not_stale_when_inner_raises(tmp_path):
    inner = FakeASRBackend(scripted_units(2))
    cache = CachingASRBackend(inner, tmp_path)
    cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='ok')
    assert cache.last_run_info['units'] == 2

    inner.error = RuntimeError('boom')
    with pytest.raises(RuntimeError):
        cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE, cache_key='other')
    assert cache.last_run_info['cache'] == 'miss'
    assert 'units' not in cache.last_run_info

    with pytest.raises(RuntimeError):
        cache.transcribe(waveform(), REQUIRED_SAMPLE_RATE)
    assert cache.last_run_info == {'cache': 'bypass'}


def test_faster_whisper_run_info_has_full_schema_when_nothing_is_pulled():
    clock = FakeClock()
    backend, model, factories = make_backend(two_segments(), clock=clock)
    deadline = Deadline(budget_s=1.0, clock=clock)
    clock.advance(5.0)

    assert backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE, deadline=deadline) == []

    info = backend.last_run_info
    assert info['truncated'] is True and info['error'] is None
    assert info['language'] is None and info['language_probability'] is None
    # Past the deadline the model is never even loaded.
    assert factories == [] and model.calls == []


def test_faster_whisper_closes_segment_generator_after_truncation(monkeypatch):
    clock = FakeClock()
    backend, _, _ = make_backend([], clock=clock, stop_reserve_s=4.0)
    closed = []

    def slow_segments(audio):
        try:
            for segment in two_segments() * 3:
                clock.advance(3.0)
                yield segment
        finally:
            closed.append(True)

    monkeypatch.setattr(backend, '_segment_iter', slow_segments)
    units = backend.transcribe(waveform(), REQUIRED_SAMPLE_RATE, deadline=Deadline(10.0, clock=clock))

    assert len(units) == 3 and closed == [True]
    assert backend.last_run_info['truncated'] is True


# --------------------------------------------------------------------------- #
# Real model (opt-in)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_real_base_model_smoke():
    """One real faster-whisper run on a supplied clip, plus a deadline stop.

    Uses ``base`` because it is already in the local HF cache and decodes a
    two-minute clip in a few seconds; ``local_files_only`` keeps it offline.
    """
    if not SAMPLE_MP3.is_file():
        pytest.skip(f'{SAMPLE_MP3} missing')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    from faster_whisper import decode_audio

    audio = decode_audio(io.BytesIO(SAMPLE_MP3.read_bytes()), sampling_rate=REQUIRED_SAMPLE_RATE)
    duration_s = len(audio) / REQUIRED_SAMPLE_RATE
    config = ASRConfig(model_size='base', device='cpu', compute_type='int8',
                       cpu_threads=2, local_files_only=True, stop_reserve_s=1.0)
    backend = FasterWhisperBackend(config)
    backend.warm_up()

    units = backend.transcribe(audio, REQUIRED_SAMPLE_RATE, deadline=Deadline(55.0))

    info = backend.last_run_info
    assert info['error'] is None and info['truncated'] is False
    assert info['language'] == 'en'
    assert len(units) >= 5
    assert check_units(units, duration_s) == []
    assert all(u.words for u in units)
    assert units[-1].end_s <= duration_s + 0.5
    text = ' '.join(u.text.strip() for u in units).lower()
    assert len(text.split()) > 100
    assert info['seconds'] < duration_s  # real-time factor well below 1

    # The generator is lazy: a budget that only covers the first window stops
    # the run early with a strict prefix of the transcript.
    partial = backend.transcribe(audio, REQUIRED_SAMPLE_RATE, deadline=Deadline(config.stop_reserve_s + 1.0))
    assert backend.last_run_info['truncated'] is True
    assert len(partial) < len(units)
    assert [u.text for u in partial] == [u.text for u in units[:len(partial)]]
