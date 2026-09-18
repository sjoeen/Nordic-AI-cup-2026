"""Tests for ``audio_io``: synthetic WAVs built in memory, so no fixtures on
disk and no real models. The one real-MP3 test is fast (about 0.2 s) and
skips itself when the sample file is not present."""

import base64
import hashlib
import io
import logging
import re
import wave
from pathlib import Path

import av
import numpy as np
import pytest

import audio_io
from audio_io import (
    AudioDecodeError,
    DecodedAudio,
    audio_sha256,
    decode_base64_audio,
    decode_request_audio,
    load_waveform,
    probe_container_duration_s,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MP3 = ROOT / 'data' / 'audio' / 'conversation_sample_4.mp3'


def make_wav_bytes(
    duration_s: float = 3.0,
    sample_rate: int = 16000,
    channels: int = 1,
    frequency_hz: float = 440.0,
) -> bytes:
    """A 16-bit PCM WAV of a sine tone, built with the standard library."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    pcm = (0.5 * np.sin(2 * np.pi * frequency_hz * t) * 32767).astype(np.int16)
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def make_flac_silence_bytes(duration_s: float, sample_rate: int = 8000) -> bytes:
    """A FLAC of pure silence: a few tens of KB that decode to minutes of audio.

    This is the shape of payload that stays under the byte limit while
    expanding tens of times on decode (thousands of times for a real
    low-bitrate file), which is what the pre-decode duration probe and the
    bounded decoder exist to refuse.
    """
    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format='flac') as container:
        stream = container.add_stream('flac', rate=sample_rate)
        stream.layout = 'mono'
        remaining = int(round(duration_s * sample_rate))
        written = 0
        while remaining > 0:
            n = min(remaining, 32768)
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, n), dtype=np.int16), format='s16', layout='mono',
            )
            frame.sample_rate = sample_rate
            frame.pts = written
            for packet in stream.encode(frame):
                container.mux(packet)
            written += n
            remaining -= n
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def flac_with_unknown_length(flac: bytes) -> bytes:
    """Zero the STREAMINFO ``total_samples`` field, which the FLAC format
    defines as "unknown". Everything else, including every frame, is intact.

    STREAMINFO is the first metadata block (4-byte header at offset 4, 34-byte
    body at offset 8); bytes 10-17 of the body pack the sample rate (20 bits),
    channels (3), bits per sample (5) and total samples (36).
    """
    assert flac[:4] == b'fLaC'
    body = bytearray(flac[8:8 + 34])
    body[13] &= 0xF0
    body[14:18] = b'\x00\x00\x00\x00'
    return flac[:8] + bytes(body) + flac[8 + 34:]


HAVE_MP3_ENCODER = 'libmp3lame' in av.codecs_available


def make_mp3_bytes(duration_s: float = 10.0, sample_rate: int = 16000) -> bytes:
    """A real MP3 of a sine tone, encoded in memory with PyAV's LAME."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n) / sample_rate
    pcm = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format='mp3') as container:
        stream = container.add_stream('libmp3lame', rate=sample_rate)
        stream.layout = 'mono'
        stream.bit_rate = 64000
        for offset in range(0, n, 1152):
            block = pcm[offset:offset + 1152]
            frame = av.AudioFrame.from_ndarray(block[None, :], format='s16', layout='mono')
            frame.sample_rate = sample_rate
            frame.pts = offset
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


# --------------------------------------------------------------------------- #
# decode_request_audio: the happy paths
# --------------------------------------------------------------------------- #

def test_decode_request_audio_wav_16k():
    wav = make_wav_bytes(3.0, 16000)
    decoded = decode_request_audio(b64(wav))

    assert isinstance(decoded, DecodedAudio)
    assert decoded.waveform.dtype == np.float32
    assert decoded.waveform.ndim == 1
    assert decoded.waveform.flags['C_CONTIGUOUS']
    assert decoded.sample_rate == 16000
    assert abs(decoded.duration_s - 3.0) <= 0.05
    assert decoded.n_samples == decoded.waveform.shape[0]
    assert abs(decoded.n_samples / 16000 - decoded.duration_s) < 1e-9
    assert decoded.n_bytes == len(wav)
    assert decoded.audio_sha256 == hashlib.sha256(wav).hexdigest()
    # A WAV has no MP3 frame header, so the diagnostic is unavailable.
    assert decoded.header_duration_s is None
    # The tone is not silence and the s16 -> f32 conversion keeps it in range.
    assert 0.3 < float(np.abs(decoded.waveform).max()) <= 1.0
    assert np.isfinite(decoded.waveform).all()


def test_8k_wav_is_resampled_to_16k_with_same_duration():
    decoded = decode_request_audio(b64(make_wav_bytes(3.0, 8000)))
    assert decoded.sample_rate == 16000
    assert abs(decoded.duration_s - 3.0) <= 0.05
    assert abs(decoded.n_samples - 48000) <= 0.05 * 16000


def test_stereo_wav_is_downmixed_to_mono():
    decoded = decode_request_audio(b64(make_wav_bytes(1.0, 16000, channels=2)))
    assert decoded.waveform.ndim == 1
    assert abs(decoded.duration_s - 1.0) <= 0.05


def test_custom_sample_rate_is_honoured():
    decoded = decode_request_audio(b64(make_wav_bytes(2.0, 16000)), sample_rate=8000)
    assert decoded.sample_rate == 8000
    assert abs(decoded.duration_s - 2.0) <= 0.05
    assert abs(decoded.n_samples - 16000) <= 0.05 * 8000


def test_single_sample_wav_decodes():
    """The smallest non-empty input: one frame, not an error."""
    decoded = decode_request_audio(b64(make_wav_bytes(1 / 16000, 16000)))
    assert decoded.n_samples >= 1
    assert decoded.duration_s > 0.0


def test_decoding_is_deterministic():
    text = b64(make_wav_bytes(1.0))
    first, second = decode_request_audio(text), decode_request_audio(text)
    assert first.audio_sha256 == second.audio_sha256
    assert np.array_equal(first.waveform, second.waveform)


def test_repr_does_not_dump_the_waveform():
    decoded = decode_request_audio(b64(make_wav_bytes(1.0)))
    assert 'waveform' not in repr(decoded)
    assert 'duration_s' in repr(decoded)


# --------------------------------------------------------------------------- #
# decode_base64_audio: tolerance and rejection
# --------------------------------------------------------------------------- #

def test_roundtrip_plain_base64():
    payload = b'\x00\x01\x02hello\xff'
    assert decode_base64_audio(b64(payload)) == payload


def test_tolerates_surrounding_and_embedded_whitespace():
    payload = bytes(range(256)) * 4
    wrapped = base64.encodebytes(payload).decode('ascii')  # 76-col lines + '\n'
    assert '\n' in wrapped
    assert decode_base64_audio('  \n' + wrapped + '\r\n\t ') == payload
    assert decode_base64_audio(wrapped.replace('\n', '\r\n')) == payload


def test_data_uri_prefix_is_stripped_and_warned(caplog):
    payload = b'not really audio but fine for base64'
    with caplog.at_level(logging.WARNING, logger='audio_io'):
        result = decode_base64_audio('data:audio/mpeg;base64,' + b64(payload))
    assert result == payload
    assert any('data-URI' in record.getMessage() for record in caplog.records)


def test_data_uri_prefix_is_case_insensitive():
    payload = b'xyz'
    assert decode_base64_audio('DATA:Audio/Mpeg;BASE64,' + b64(payload)) == payload


def test_data_uri_prefix_with_nothing_after_it_is_empty():
    with pytest.raises(AudioDecodeError, match='empty'):
        decode_base64_audio('data:audio/mpeg;base64,')


@pytest.mark.parametrize('text', ['', '   ', '\n\r\n\t'])
def test_empty_input_rejected(text):
    with pytest.raises(AudioDecodeError, match='empty'):
        decode_base64_audio(text)


@pytest.mark.parametrize('text', [
    'not*valid*base64!!',
    'QUJD.',            # a byte outside the alphabet
    'QUJDRA',           # missing padding
    'QUJDRA=',          # wrong padding
    'QUJD-RA_',         # urlsafe alphabet is not accepted
    'QUJDРА==',         # non-ASCII characters
])
def test_bad_base64_rejected(text):
    with pytest.raises(AudioDecodeError, match='not valid base64'):
        decode_base64_audio(text)


def test_padding_only_input_rejected():
    with pytest.raises(AudioDecodeError):
        decode_base64_audio('====')


@pytest.mark.parametrize('value', [None, 123, b'QUJD', ['QUJD']])
def test_non_string_input_rejected(value):
    with pytest.raises(AudioDecodeError, match='must be a string'):
        decode_base64_audio(value)


def test_oversize_is_rejected_before_decoding(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('b64decode must not run for an oversized payload')

    monkeypatch.setattr(audio_io.base64, 'b64decode', forbidden)
    big = 'A' * 4000  # decodes to 3000 bytes
    with pytest.raises(AudioDecodeError, match='above the limit'):
        decode_base64_audio(big, max_bytes=2999)


@pytest.mark.parametrize('n_bytes', [1, 2, 3, 4, 5, 6, 100])
def test_size_limit_is_inclusive_and_exact(n_bytes):
    """Every padding case (0, 1 or 2 trailing '='): exactly at the limit is
    accepted, one byte over is refused."""
    payload = b'\x7f' * n_bytes
    assert decode_base64_audio(b64(payload), max_bytes=n_bytes) == payload
    with pytest.raises(AudioDecodeError, match='above the limit'):
        decode_base64_audio(b64(payload + b'\x7f'), max_bytes=n_bytes)


def test_size_precheck_counts_data_chars_not_padding(monkeypatch):
    """Regression: strict b64decode accepts any number of trailing '=' after a
    complete quad, so a padding-subtracting estimate could be pushed below
    the real decoded size and the pre-decode guard bypassed."""
    def forbidden(*args, **kwargs):
        raise AssertionError('b64decode must not run for an oversized payload')

    payload = 'QUJD' * 400 + '=' * 1200  # decodes to exactly 1200 bytes
    assert audio_io._decoded_size_upper_bound(payload) == 1200

    monkeypatch.setattr(audio_io.base64, 'b64decode', forbidden)
    with pytest.raises(AudioDecodeError, match='above the limit'):
        decode_base64_audio(payload, max_bytes=1199)


@pytest.mark.parametrize('text', ['QUJD=', 'QUJD==', 'QUJD===', 'QUJD' + '=' * 50])
def test_excess_trailing_padding_is_accepted_and_bounded(text):
    """Python accepts these (as does the reference utils.decode_audio); the
    size bound must still be exact on what they decode to."""
    assert decode_base64_audio(text, max_bytes=3) == b'ABC'
    with pytest.raises(AudioDecodeError, match='above the limit'):
        decode_base64_audio(text, max_bytes=2)


def test_default_limit_matches_contract():
    assert audio_io.MAX_AUDIO_BYTES == 32 * 1024 * 1024
    assert audio_io.MAX_DURATION_S == 1800


@pytest.mark.parametrize('bad', [0, -1, 1.5, True, None])
def test_invalid_max_bytes_is_a_programming_error(bad):
    with pytest.raises(ValueError, match='max_bytes'):
        decode_base64_audio('QUJD', max_bytes=bad)


# --------------------------------------------------------------------------- #
# audio_sha256 and load_waveform
# --------------------------------------------------------------------------- #

def test_audio_sha256_matches_hashlib():
    payload = b'some bytes'
    digest = audio_sha256(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
    assert audio_sha256(b'') == hashlib.sha256(b'').hexdigest()


def test_load_waveform_wav():
    waveform = load_waveform(make_wav_bytes(0.5, 16000))
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert abs(waveform.shape[0] / 16000 - 0.5) <= 0.05


@pytest.mark.parametrize('junk', [
    b'hello world, this is definitely not audio',
    b'\x00' * 1000,
    b'RIFF' + b'\x00' * 40,     # a RIFF header with nothing usable behind it
    b'\xff\xfb' + b'\x01' * 10,  # looks like an MP3 sync word, is not a frame
])
def test_non_audio_bytes_rejected(junk):
    with pytest.raises(AudioDecodeError, match='could not be decoded'):
        load_waveform(junk)
    with pytest.raises(AudioDecodeError):
        decode_request_audio(b64(junk))


def test_load_waveform_empty_bytes_rejected():
    with pytest.raises(AudioDecodeError, match='no audio bytes'):
        load_waveform(b'')


def test_wav_with_zero_frames_rejected():
    empty_wav = make_wav_bytes(0.0, 16000)
    with pytest.raises(AudioDecodeError, match='zero samples'):
        load_waveform(empty_wav)
    with pytest.raises(AudioDecodeError, match='zero samples'):
        decode_request_audio(b64(empty_wav))


@pytest.mark.parametrize('not_audio', [
    b'1\n00:00:00,000 --> 00:00:01,000\nhello\n\n',  # an SRT subtitle file
    bytes.fromhex(                                      # a 1x1 PNG
        '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489'
        '0000000d49444154789c6360000002000148afa4710000000049454e44ae426082'
    ),
])
def test_container_without_audio_stream_rejected(not_audio):
    """PyAV opens these fine but they carry no audio stream."""
    assert probe_container_duration_s(not_audio) is None
    with pytest.raises(AudioDecodeError, match='no audio stream'):
        load_waveform(not_audio)


@pytest.mark.parametrize('bad', [0, -16000, 16000.0, True])
def test_invalid_sample_rate_is_a_programming_error(bad):
    with pytest.raises(ValueError, match='sample_rate'):
        load_waveform(make_wav_bytes(0.1), sample_rate=bad)


def test_decoder_errors_are_wrapped_with_cause(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError('simulated PyAV failure')

    monkeypatch.setattr(av, 'open', boom)
    with pytest.raises(AudioDecodeError, match='OSError') as info:
        load_waveform(b'anything')
    assert isinstance(info.value.__cause__, OSError)


def test_memory_error_is_not_swallowed(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError()

    monkeypatch.setattr(av, 'open', boom)
    with pytest.raises(MemoryError):
        load_waveform(b'anything')


def test_load_waveform_matches_faster_whisper_decoder_exactly():
    """The ASR stage must see the same samples faster_whisper's own decoder
    would give it; the loop here only adds the bound and the damage handling."""
    from faster_whisper.audio import decode_audio

    fixtures = [
        make_wav_bytes(1.0, 16000),
        make_wav_bytes(1.0, 8000),
        make_wav_bytes(0.5, 44100, channels=2),
        make_wav_bytes(1 / 16000, 16000),
        make_flac_silence_bytes(2.0),
    ]
    if HAVE_MP3_ENCODER:
        fixtures.append(make_mp3_bytes(2.0))
    for data in fixtures:
        reference = decode_audio(io.BytesIO(data), sampling_rate=16000)
        assert np.array_equal(load_waveform(data), reference)


def test_load_waveform_sample_bound_is_inclusive_and_exact():
    wav = make_wav_bytes(1.0, 16000)  # exactly 16000 frames
    assert load_waveform(wav, max_samples=16000).shape[0] == 16000
    with pytest.raises(AudioDecodeError, match='audio is at least 1.0 s long, above the limit of 0.9999'):
        load_waveform(wav, max_samples=15999)


@pytest.mark.parametrize('bad', [0, -1, 1.5, True])
def test_invalid_max_samples_is_a_programming_error(bad):
    with pytest.raises(ValueError, match='max_samples'):
        load_waveform(make_wav_bytes(0.1), max_samples=bad)


@pytest.mark.skipif(not HAVE_MP3_ENCODER, reason='PyAV build has no MP3 encoder')
def test_corrupt_packets_in_the_middle_do_not_truncate_the_recording(caplog):
    """Regression: faster_whisper's decoder stops at the first packet the
    decoder rejects and silently returns what came before it, so a few KB of
    damage at the 5 s mark of a 10 s file lost the second half. Packets are
    skipped individually instead, and the loss is logged."""
    mp3 = make_mp3_bytes(10.0)
    clean = load_waveform(mp3)
    assert abs(clean.shape[0] / 16000 - 10.0) < 0.05

    middle = len(mp3) // 2
    damaged = mp3[:middle] + b'\x00' * 3000 + mp3[middle:]
    with caplog.at_level(logging.WARNING, logger='audio_io'):
        waveform = load_waveform(damaged)
    assert waveform.shape[0] / 16000 > 9.0
    messages = [r.getMessage() for r in caplog.records if 'damaged' in r.getMessage()]
    assert len(messages) == 1
    assert 'undecodable packets skipped' in messages[0]


@pytest.mark.skipif(not HAVE_MP3_ENCODER, reason='PyAV build has no MP3 encoder')
def test_clean_mp3_decodes_without_damage_warning(caplog):
    with caplog.at_level(logging.WARNING, logger='audio_io'):
        decoded = decode_request_audio(b64(make_mp3_bytes(3.0)))
    assert abs(decoded.duration_s - 3.0) < 0.1
    assert not [r for r in caplog.records if 'damaged' in r.getMessage()]


def test_demuxer_failure_keeps_the_decoded_prefix_with_a_warning(monkeypatch, caplog):
    """A demuxer that gives up mid-stream ends the recording there (PyAV's
    generator cannot resume); the prefix is kept, and the pipeline is told."""
    real_open = av.open

    class BreaksAfterTenPackets:
        def __init__(self, container):
            self._container = container

        def __enter__(self):
            self._container.__enter__()
            return self

        def __exit__(self, *exc_info):
            return self._container.__exit__(*exc_info)

        @property
        def streams(self):
            return self._container.streams

        def demux(self, stream):
            for index, packet in enumerate(self._container.demux(stream)):
                if index == 10:
                    raise av.error.InvalidDataError(1094995529, 'simulated demuxer failure')
                yield packet

    monkeypatch.setattr(av, 'open', lambda *a, **k: BreaksAfterTenPackets(real_open(*a, **k)))
    with caplog.at_level(logging.WARNING, logger='audio_io'):
        waveform = load_waveform(make_wav_bytes(3.0, 16000))
    assert 0 < waveform.shape[0] < 48000
    messages = [r.getMessage() for r in caplog.records if 'damaged' in r.getMessage()]
    assert len(messages) == 1 and 'demuxer stopped before the end' in messages[0]


def test_all_packets_undecodable_is_zero_samples_with_the_count(monkeypatch):
    real_open = av.open

    class EveryPacketRejected:
        def __init__(self, container):
            self._container = container

        def __enter__(self):
            self._container.__enter__()
            return self

        def __exit__(self, *exc_info):
            return self._container.__exit__(*exc_info)

        @property
        def streams(self):
            return self._container.streams

        def demux(self, stream):
            for packet in self._container.demux(stream):
                yield Rejecting()

    class Rejecting:
        def decode(self):
            raise av.error.InvalidDataError(1094995529, 'simulated decoder failure')

    monkeypatch.setattr(av, 'open', lambda *a, **k: EveryPacketRejected(real_open(*a, **k)))
    with pytest.raises(AudioDecodeError, match=r'zero samples \([1-9][0-9]* undecodable packets\)'):
        load_waveform(make_wav_bytes(1.0))


# --------------------------------------------------------------------------- #
# decode_request_audio: bounds and diagnostics
# --------------------------------------------------------------------------- #

def test_duration_limit_rejects_long_audio():
    text = b64(make_wav_bytes(3.0))
    with pytest.raises(AudioDecodeError, match='above the limit of 1 s'):
        decode_request_audio(text, max_duration_s=1.0)
    # The limit is inclusive: exactly-at-limit audio is fine.
    assert abs(decode_request_audio(text, max_duration_s=3.0).duration_s - 3.0) <= 0.05


@pytest.mark.parametrize('bad', [0, -5.0, float('nan'), float('inf'), True, None])
def test_invalid_max_duration_is_a_programming_error(bad):
    with pytest.raises(ValueError, match='max_duration_s'):
        decode_request_audio(b64(make_wav_bytes(0.1)), max_duration_s=bad)


def test_max_duration_shorter_than_one_sample_is_a_programming_error():
    with pytest.raises(ValueError, match='shorter than one sample'):
        decode_request_audio(b64(make_wav_bytes(0.1)), max_duration_s=1e-6)
    with pytest.raises(ValueError, match='shorter than one sample'):
        decode_request_audio(b64(make_wav_bytes(0.1)), sample_rate=8000, max_duration_s=1 / 8001)
    # One sample's worth of limit is valid and admits a one-sample file.
    assert decode_request_audio(b64(make_wav_bytes(1 / 16000)), max_duration_s=1 / 16000).n_samples == 1


def test_config_is_validated_before_the_payload_is_touched(monkeypatch):
    """A bad sample_rate is a programming error and must surface as one even
    when the payload is bad too, without doing any base64 work first."""
    def forbidden(*args, **kwargs):
        raise AssertionError('payload must not be decoded when the config is invalid')

    monkeypatch.setattr(audio_io.base64, 'b64decode', forbidden)
    for payload in ['', 'not base64!', b64(make_wav_bytes(0.1))]:
        with pytest.raises(ValueError, match='sample_rate') as info:
            decode_request_audio(payload, sample_rate=0)
        assert not isinstance(info.value, AudioDecodeError)


# --------------------------------------------------------------------------- #
# Pre-decode duration probe
# --------------------------------------------------------------------------- #

def test_probe_reads_wav_duration_from_the_header():
    assert abs(probe_container_duration_s(make_wav_bytes(3.0)) - 3.0) < 1e-3
    assert abs(probe_container_duration_s(make_wav_bytes(1.0, 8000, channels=2)) - 1.0) < 1e-3


@pytest.mark.parametrize('data', [b'', b'hello world', b'\x00' * 1000, b'RIFF' + b'\x00' * 40])
def test_probe_returns_none_for_unknown_or_unopenable_bytes(data):
    assert probe_container_duration_s(data) is None


def test_probe_never_raises_on_decoder_failure(monkeypatch):
    import av

    def boom(*args, **kwargs):
        raise RuntimeError('simulated PyAV failure')

    monkeypatch.setattr(av, 'open', boom)
    assert probe_container_duration_s(make_wav_bytes(1.0)) is None


def test_probe_ignores_nonsense_metadata_values(monkeypatch):
    assert audio_io._positive_or_none(0.0) is None
    assert audio_io._positive_or_none(-3.0) is None
    assert audio_io._positive_or_none(float('nan')) is None
    assert audio_io._positive_or_none(2.5) == 2.5


def test_long_container_is_refused_before_any_sample_is_decoded(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('load_waveform must not run for a payload that claims to be too long')

    monkeypatch.setattr(audio_io, 'load_waveform', forbidden)
    with pytest.raises(AudioDecodeError, match='container reports 3.0 s .* above the limit of 1 s'):
        decode_request_audio(b64(make_wav_bytes(3.0)), max_duration_s=1.0)


def test_compressed_silence_is_refused_before_decoding(monkeypatch):
    """Regression: a few tens of KB of FLAC silence decode to minutes; the
    byte limit cannot see that, the probe must. At 32 MiB this shape would
    decode to well over a day and take the process down."""
    flac = make_flac_silence_bytes(120.0)
    assert len(flac) * 20 < 120.0 * 8000 * 2, 'fixture no longer compresses'
    assert abs(probe_container_duration_s(flac) - 120.0) < 0.1

    monkeypatch.setattr(audio_io, 'load_waveform', lambda *a, **k: pytest.fail('decoded anyway'))
    with pytest.raises(AudioDecodeError, match='container reports 120.0 s'):
        decode_request_audio(b64(flac), max_duration_s=60.0)


def test_unknown_container_length_cannot_bypass_the_bound(caplog):
    """Regression: a FLAC may legally report its length as unknown, which
    made the probe return None and the whole file decode before the length
    check ran. The bound must hold inside the decode loop instead, stopping
    within a chunk of the limit, not at the end of the file."""
    flac = flac_with_unknown_length(make_flac_silence_bytes(120.0))
    assert probe_container_duration_s(flac) is None
    assert abs(load_waveform(flac).shape[0] / 16000 - 120.0) < 0.01, 'fixture must still decode'

    with pytest.raises(AudioDecodeError, match=r'audio is at least ([0-9.]+) s long, above the limit of 10 s') as info:
        decode_request_audio(b64(flac), max_duration_s=10.0)
    stopped_at = float(re.search(r'at least ([0-9.]+) s', str(info.value)).group(1))
    assert 10.0 < stopped_at < 30.0, 'decoding should stop within a chunk of the limit, not at 120 s'
    assert not [r for r in caplog.records if 'damaged' in r.getMessage()]


def test_probe_slack_defers_marginal_claims_to_the_exact_check(monkeypatch):
    """A claim within the slack is not trusted for a rejection: the file is
    decoded and its real length decides. Here the real length is fine."""
    limit = 1.0
    monkeypatch.setattr(audio_io, 'probe_container_duration_s',
                        lambda _b: limit * (1.0 + audio_io.DURATION_PROBE_SLACK))
    decoded = decode_request_audio(b64(make_wav_bytes(1.0)), max_duration_s=limit)
    assert abs(decoded.duration_s - 1.0) <= 0.05


def test_lying_short_claim_does_not_bypass_the_exact_check(monkeypatch):
    monkeypatch.setattr(audio_io, 'probe_container_duration_s', lambda _b: 0.5)
    with pytest.raises(AudioDecodeError, match='audio is at least 3.0 s long, above the limit of 1 s'):
        decode_request_audio(b64(make_wav_bytes(3.0)), max_duration_s=1.0)


def test_unknown_claim_falls_through_to_the_exact_check(monkeypatch):
    monkeypatch.setattr(audio_io, 'probe_container_duration_s', lambda _b: None)
    decoded = decode_request_audio(b64(make_wav_bytes(1.0)), max_duration_s=2.0)
    assert abs(decoded.duration_s - 1.0) <= 0.05
    with pytest.raises(AudioDecodeError, match='audio is at least 1.0 s long, above the limit of 0.5 s'):
        decode_request_audio(b64(make_wav_bytes(1.0)), max_duration_s=0.5)


def test_max_bytes_is_passed_through():
    wav = make_wav_bytes(1.0)
    with pytest.raises(AudioDecodeError, match='above the limit'):
        decode_request_audio(b64(wav), max_bytes=len(wav) - 1)
    assert decode_request_audio(b64(wav), max_bytes=len(wav)).n_bytes == len(wav)


def test_request_level_rejections_share_one_exception_type():
    for text in ['', 'not base64!', b64(b'junk bytes'), 'data:audio/mpeg;base64,']:
        with pytest.raises(AudioDecodeError):
            decode_request_audio(text)


def test_header_estimate_is_diagnostic_only(monkeypatch, caplog):
    """A broken header inspector must never fail the request."""
    def boom(_bytes):
        raise RuntimeError('header parser exploded')

    monkeypatch.setattr(audio_io, 'audio_duration_seconds', boom)
    with caplog.at_level(logging.WARNING, logger='audio_io'):
        decoded = decode_request_audio(b64(make_wav_bytes(1.0)))
    assert decoded.header_duration_s is None
    assert abs(decoded.duration_s - 1.0) <= 0.05
    assert any('header inspection failed' in r.getMessage() for r in caplog.records)


def test_header_mismatch_is_warned_not_rejected(monkeypatch, caplog):
    monkeypatch.setattr(audio_io, 'audio_duration_seconds', lambda _bytes: 100.0)
    with caplog.at_level(logging.WARNING, logger='audio_io'):
        decoded = decode_request_audio(b64(make_wav_bytes(1.0)))
    assert decoded.header_duration_s == 100.0
    assert abs(decoded.duration_s - 1.0) <= 0.05
    assert any('header suggests' in r.getMessage() for r in caplog.records)


def test_header_agreement_is_quiet(monkeypatch, caplog):
    monkeypatch.setattr(audio_io, 'audio_duration_seconds', lambda _bytes: 1.01)
    with caplog.at_level(logging.WARNING, logger='audio_io'):
        decoded = decode_request_audio(b64(make_wav_bytes(1.0)))
    assert decoded.header_duration_s == 1.01
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------- #
# A real supplied conversation, when it is on disk
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not SAMPLE_MP3.exists(), reason='sample MP3 not present')
def test_real_mp3_decodes_and_matches_header_duration():
    raw = SAMPLE_MP3.read_bytes()
    decoded = decode_request_audio(b64(raw))
    assert decoded.waveform.dtype == np.float32
    assert decoded.waveform.ndim == 1
    assert decoded.sample_rate == 16000
    assert decoded.n_bytes == len(raw)
    assert decoded.audio_sha256 == hashlib.sha256(raw).hexdigest()
    assert decoded.header_duration_s is not None
    assert abs(decoded.duration_s - decoded.header_duration_s) <= 1.0
    assert 30.0 < decoded.duration_s < 600.0
    assert np.isfinite(decoded.waveform).all()
    # The ASR stage gets exactly what faster_whisper's own decoder would give it.
    from faster_whisper.audio import decode_audio
    assert np.array_equal(decoded.waveform, decode_audio(io.BytesIO(raw), sampling_rate=16000))
