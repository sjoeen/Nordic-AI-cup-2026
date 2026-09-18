"""Bounded decoding of the request audio into a 16 kHz float32 waveform.

This is the first stage of the request path and the only one that touches the
raw wire payload, so it is also where the size and duration bounds live.
Everything that arrives here is untrusted: the base64 text may be malformed,
padded with a data-URI prefix, absurdly large, or decode to bytes that are not
audio at all. Every such case becomes an :class:`AudioDecodeError` (a
``ValueError``) so the pipeline can map it to one error contract instead of
catching a zoo of PyAV, binascii and numpy exceptions.

Decoding uses PyAV directly (so no ``ffmpeg`` binary is needed on the host)
in a loop that mirrors ``faster_whisper.audio.decode_audio`` and produces the
same samples bit for bit, which a test enforces. It is not that function
itself for two reasons: the loop stops as soon as the decoded length exceeds
the limit, so a payload that lies about (or omits) its duration cannot expand
into gigabytes before the length check runs; and it decodes packet by packet,
so a corrupt stretch in the middle costs those packets rather than silently
truncating the recording at the first bad one. The output is mono float32 at
the requested rate, which is exactly what the ASR stage consumes, and the
sample count divided by the rate is the duration on the ORIGINAL timeline that
every evidence span is measured against.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import av
import numpy as np

from utils import audio_duration_seconds

logger = logging.getLogger(__name__)

# Upper bound on the DECODED payload. The supplied conversations are 1-4 MB
# MP3s (about 2-4 minutes at 128 kbps); 32 MiB is roughly 35 minutes at that
# bitrate, which is far beyond any consultation while still small enough that
# decoding it cannot exhaust the 60 s request budget or the host's memory.
MAX_AUDIO_BYTES: int = 32 * 1024 * 1024

# Upper bound on the decoded length in seconds. A 30-minute waveform at 16 kHz
# is ~115 MB of float32, still comfortably within the dev box's memory.
MAX_DURATION_S: float = 1800.0

# Whisper models are trained on 16 kHz mono audio.
DEFAULT_SAMPLE_RATE: int = 16000

# If the MP3 header estimate and the decoded length disagree by more than this
# the bytes probably did not survive the trip intact; that is worth a warning
# but never a rejection, because the header estimate assumes constant bitrate.
HEADER_MISMATCH_WARN_S: float = 2.0

# The container's own duration metadata (see :func:`probe_container_duration_s`)
# is exact for WAV, FLAC, Opus and CBR or Xing-tagged MP3, but ffmpeg estimates
# it from the first frame's bitrate for a VBR MP3 without a Xing header, which
# can be off by a wide margin either way. So the probe refuses a payload only
# when it claims to exceed the limit by more than this fraction; anything
# closer is decoded and judged on its exact sample count.
DURATION_PROBE_SLACK: float = 0.10

# Decoded input samples are pooled in a FIFO and handed to the resampler in
# chunks of this many, as faster_whisper does, because one resampler call per
# codec frame (1152 samples for MP3) costs more in call overhead than in
# arithmetic. It also caps how far past ``max_samples`` a decode can run
# before the bound is noticed: one chunk, a fraction of a second of audio.
_FIFO_CHUNK_SAMPLES: int = 65536

# The README promises plain base64 with no ``data:audio/mpeg;base64,`` prefix,
# but a client that builds the request in a browser may add one anyway.
_DATA_URI_PREFIX = re.compile(r'^data:[^,]*;base64,', re.IGNORECASE)


class AudioDecodeError(ValueError):
    """The request audio could not be turned into a waveform.

    Raised for malformed base64, empty or oversized payloads, bytes that no
    decoder recognises as audio, and decoded audio longer than the configured
    limit. The message is safe to log and to return in an error response: it
    never echoes the untrusted payload itself.
    """


@dataclass
class DecodedAudio:
    """The request audio as the ASR stage wants it, plus provenance.

    ``duration_s`` is derived from the sample count and is the authoritative
    length of the original timeline. ``header_duration_s`` is the cheap MP3
    header estimate from :func:`utils.audio_duration_seconds`; it is a
    diagnostic only and is ``None`` for anything that is not an MPEG-1 Layer
    III stream (a WAV, for instance). ``audio_sha256`` is the hash of the raw
    bytes, which makes a stable cache key for transcripts.
    """

    waveform: np.ndarray = field(repr=False)
    sample_rate: int
    duration_s: float
    audio_sha256: str
    n_bytes: int
    header_duration_s: Optional[float] = None

    @property
    def n_samples(self) -> int:
        return int(self.waveform.shape[0])


def _strip_data_uri_prefix(text: str) -> str:
    """Remove a leading ``data:...;base64,`` prefix, warning when one is found.

    The prefix is not part of the protocol, so its presence means the client
    is not the reference evaluator; that is worth knowing about in the logs.
    """
    match = _DATA_URI_PREFIX.match(text)
    if match is None:
        return text
    logger.warning(
        'audio_base64 carried a data-URI prefix (%r); stripping it',
        match.group(0)[:64],
    )
    return text[match.end():]


def _check_positive_int(name: str, value: object) -> int:
    """Validate a configuration integer such as ``max_bytes`` or ``sample_rate``.

    These come from trusted code, so a bad value is a programming error and
    raises a plain ``ValueError`` rather than :class:`AudioDecodeError`, which
    is reserved for untrusted input. ``bool`` is excluded explicitly because
    it is an ``int`` subclass and ``sample_rate=True`` is never intended.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive int, got {value!r}')
    return value


def _decoded_size_upper_bound(encoded: str) -> int:
    """Decoded length of a base64 string, computed without decoding it.

    Only the non-``=`` characters carry data: every four of them are three
    bytes, a trailing two or three are one or two. Counting data characters
    rather than subtracting padding matters because strict ``b64decode``
    accepts any number of trailing ``=`` after a complete quad (``QUJD====``
    decodes to three bytes), so a padding-based formula could be pushed below
    the real size by appending ``=``. This count is exact for everything
    strict decoding accepts and an upper bound for everything it rejects, so
    an oversized payload is refused before it costs a second copy of itself
    in memory.
    """
    data_chars = len(encoded.rstrip('='))
    return (data_chars * 3) // 4


def decode_base64_audio(audio_base64: str, max_bytes: int = MAX_AUDIO_BYTES) -> bytes:
    """Turn ``request.audio_base64`` into raw audio bytes, or raise.

    Tolerates surrounding and embedded whitespace (MIME-style line wrapping)
    and a stray data-URI prefix; rejects everything else that is not strict
    standard-alphabet base64. The size bound is enforced on the encoded length
    first so that a hostile payload is refused before it is expanded.
    """
    _check_positive_int('max_bytes', max_bytes)
    if not isinstance(audio_base64, str):
        raise AudioDecodeError(
            f'audio_base64 must be a string, got {type(audio_base64).__name__}'
        )

    # ``split()`` with no argument drops every run of ASCII whitespace, which
    # covers the 76-column line wrapping ``base64.encodebytes`` produces.
    compact = ''.join(audio_base64.split())
    compact = _strip_data_uri_prefix(compact)
    if not compact:
        raise AudioDecodeError('audio_base64 is empty')

    estimated = _decoded_size_upper_bound(compact)
    if estimated > max_bytes:
        raise AudioDecodeError(
            f'audio payload is about {estimated} bytes, above the limit of {max_bytes}'
        )

    try:
        audio_bytes = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        # binascii.Error is a ValueError subclass; the tuple documents both
        # sources (bad alphabet/padding vs. non-ASCII characters).
        raise AudioDecodeError(f'audio_base64 is not valid base64: {exc}') from exc

    if not audio_bytes:
        raise AudioDecodeError('audio_base64 decoded to zero bytes')
    if len(audio_bytes) > max_bytes:
        # Unreachable while the pre-check above is exact for everything
        # strict decoding accepts; kept as the invariant the rest of the
        # pipeline relies on, so a future change to either side stays safe.
        raise AudioDecodeError(
            f'audio payload is {len(audio_bytes)} bytes, above the limit of {max_bytes}'
        )
    return audio_bytes


def audio_sha256(audio_bytes: bytes) -> str:
    """Hex SHA-256 of the raw audio bytes: the transcript cache key."""
    return hashlib.sha256(audio_bytes).hexdigest()


def probe_container_duration_s(audio_bytes: bytes) -> Optional[float]:
    """The duration the container's own metadata claims, or ``None``.

    This only parses headers, so it costs well under a millisecond and no
    memory proportional to the audio. It exists because the byte limit alone
    does not bound the decoded size: a FLAC or low-bitrate MP3 of silence
    stays under 32 MiB while decoding to tens of hours, which would exhaust
    the host's memory long before the sample-count check could run. The
    value is untrusted metadata, exact for honest files and open to lying
    ones, so callers use it only to refuse early, never to accept.

    Returns ``None`` whenever the answer is unknown: bytes PyAV cannot open
    (the decoder will produce the proper error), no audio stream, or no
    duration field. Nothing here raises except ``MemoryError``.
    """
    if not audio_bytes:
        return None
    try:
        with av.open(io.BytesIO(audio_bytes), mode='r', metadata_errors='ignore') as container:
            duration_us = container.duration
            if duration_us is None and container.streams.audio:
                # Some demuxers only fill in the per-stream duration, in the
                # stream's own time base rather than microseconds.
                stream = container.streams.audio[0]
                if stream.duration is not None and stream.time_base is not None:
                    return _positive_or_none(float(stream.duration * stream.time_base))
            if duration_us is None:
                return None
            return _positive_or_none(duration_us / 1_000_000)
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - PyAV's failure surface is broad
        logger.debug('container duration probe failed (%s)', type(exc).__name__)
        return None


def _positive_or_none(value: float) -> Optional[float]:
    """Metadata durations can be zero, negative or NaN; treat those as unknown."""
    return value if np.isfinite(value) and value > 0.0 else None


def _decode_pcm_mono(
    audio_bytes: bytes,
    sample_rate: int,
    max_samples: Optional[int],
) -> Tuple[List[np.ndarray], int, bool]:
    """Demux, decode and resample to s16 mono chunks; the bounded core of
    :func:`load_waveform`.

    Returns ``(chunks, n_skipped_packets, demux_broke)``. The sample bound is
    checked every time a chunk leaves the resampler, so the decode is
    abandoned within one chunk of crossing it whatever the container claimed.
    A packet the decoder rejects is skipped and counted, which keeps the rest
    of a recording with a corrupt stretch in it; a demuxer failure ends the
    stream where it is (PyAV's generator cannot resume after raising), and
    the flag lets the caller report that the tail is missing. This keeps at
    least what ``faster_whisper.audio.decode_audio`` keeps; that function
    stops at the first rejected packet and says nothing about it.
    """
    resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=sample_rate)
    fifo = av.audio.fifo.AudioFifo()
    chunks: List[np.ndarray] = []
    n_samples = 0
    n_skipped = 0
    demux_broke = False

    def drain(frame: Optional[av.AudioFrame]) -> None:
        # ``None`` flushes the resampler's delay line at the end of the stream.
        nonlocal n_samples
        for resampled in resampler.resample(frame):
            pcm = resampled.to_ndarray().reshape(-1)
            chunks.append(pcm)
            n_samples += int(pcm.shape[0])
        if max_samples is not None and n_samples > max_samples:
            raise AudioDecodeError(
                f'audio is at least {n_samples / sample_rate:.1f} s long, '
                f'above the limit of {max_samples / sample_rate:g} s'
            )

    with av.open(io.BytesIO(audio_bytes), mode='r', metadata_errors='ignore') as container:
        if not container.streams.audio:
            raise AudioDecodeError('container has no audio stream')
        packets = container.demux(container.streams.audio[0])
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except av.error.InvalidDataError:
                demux_broke = True
                break
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                n_skipped += 1
                continue
            for frame in frames:
                # The FIFO insists on contiguous timestamps and a skipped
                # packet (or a sloppy encoder) breaks that; the timeline is
                # rebuilt from the sample count anyway.
                frame.pts = None
                fifo.write(frame)
            if fifo.samples >= _FIFO_CHUNK_SAMPLES:
                drain(fifo.read())
        if fifo.samples > 0:
            drain(fifo.read())
        drain(None)
    return chunks, n_skipped, demux_broke


def load_waveform(
    audio_bytes: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Decode any container PyAV understands into mono float32 at ``sample_rate``.

    Stereo input is downmixed and any input rate is resampled, so the ASR
    stage never has to care what the client sent. Every decoder failure is
    re-raised as :class:`AudioDecodeError`; PyAV raises a mix of its own
    error classes, ``ValueError`` and ``OSError`` depending on how the input
    is broken, and none of that should leak into the request handler.

    ``max_samples`` bounds the OUTPUT length: decoding stops and raises as
    soon as the running sample count exceeds it, so memory is bounded by the
    limit rather than by whatever the payload expands to. ``None`` decodes
    whatever it is given; the duration bound lives in
    :func:`decode_request_audio`.

    Corrupt packets in the middle of a stream are skipped, and a stream the
    demuxer gives up on is kept up to that point, both with a warning, on the
    principle that a partial recording beats none.
    """
    _check_positive_int('sample_rate', sample_rate)
    if max_samples is not None:
        _check_positive_int('max_samples', max_samples)
    if not audio_bytes:
        raise AudioDecodeError('no audio bytes to decode')

    try:
        chunks, n_skipped, demux_broke = _decode_pcm_mono(audio_bytes, sample_rate, max_samples)
    except (AudioDecodeError, MemoryError):
        raise
    except Exception as exc:  # noqa: BLE001 - PyAV's failure surface is broad
        raise AudioDecodeError(
            f'audio bytes could not be decoded ({type(exc).__name__}: {exc})'
        ) from exc

    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    if pcm.shape[0] == 0:
        detail = f' ({n_skipped} undecodable packets)' if n_skipped else ''
        raise AudioDecodeError(f'audio decoded to zero samples{detail}')
    if n_skipped or demux_broke:
        logger.warning(
            'audio stream is damaged: %d undecodable packets skipped%s; '
            'keeping the %.2f s that decoded',
            n_skipped,
            ', demuxer stopped before the end' if demux_broke else '',
            pcm.shape[0] / sample_rate,
        )

    # s16 -> f32 the way faster_whisper does it, so the ASR stage sees the
    # same numbers it would from its own decoder. ``astype`` returns a fresh
    # C-contiguous array and the division by a power of two is exact.
    waveform = pcm.astype(np.float32)
    waveform /= 32768.0
    return waveform


def decode_request_audio(
    audio_base64: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_bytes: int = MAX_AUDIO_BYTES,
    max_duration_s: float = MAX_DURATION_S,
) -> DecodedAudio:
    """The whole ingress step: base64 text in, bounded waveform out.

    The duration limit is enforced twice. First on the container's own
    metadata, before any sample is decoded, because the byte limit does not
    bound the decoded size (compressed silence expands thousands of times)
    and a payload that honestly claims to be hours long is cheapest to refuse
    unread. Then, authoritatively, on the decoded sample count as it grows,
    because metadata can be absent (a FLAC may legally say "unknown"),
    inexact for VBR MP3 without a Xing header, or simply false; the decoder
    stops within one chunk of crossing the limit, so a lying payload costs
    at most the limit's worth of memory and time.
    """
    if not isinstance(max_duration_s, (int, float)) or isinstance(max_duration_s, bool) \
            or not np.isfinite(max_duration_s) or max_duration_s <= 0:
        raise ValueError(f'max_duration_s must be a positive finite number, got {max_duration_s!r}')
    # Configuration is checked before the payload is touched, so a bad
    # ``sample_rate`` is reported as the programming error it is even when the
    # payload is bad too, and no base64 work is wasted on it.
    _check_positive_int('sample_rate', sample_rate)
    # ``n > floor(x)`` is ``n > x`` for integer ``n``, so the sample bound is
    # exactly the duration bound, inclusive at the limit.
    max_samples = int(np.floor(max_duration_s * sample_rate))
    if max_samples < 1:
        raise ValueError(
            f'max_duration_s={max_duration_s!r} is shorter than one sample at {sample_rate} Hz'
        )

    audio_bytes = decode_base64_audio(audio_base64, max_bytes=max_bytes)
    sha256 = audio_sha256(audio_bytes)

    # The header estimate is a diagnostic; a diagnostic must never be the
    # reason a request fails, whatever the bytes look like.
    try:
        header_duration_s = audio_duration_seconds(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning('MP3 header inspection failed (%s); continuing', type(exc).__name__)
        header_duration_s = None

    claimed_duration_s = probe_container_duration_s(audio_bytes)
    if claimed_duration_s is not None and claimed_duration_s > max_duration_s * (1.0 + DURATION_PROBE_SLACK):
        raise AudioDecodeError(
            f'container reports {claimed_duration_s:.1f} s of audio, '
            f'above the limit of {max_duration_s:g} s'
        )

    waveform = load_waveform(audio_bytes, sample_rate=sample_rate, max_samples=max_samples)
    duration_s = waveform.shape[0] / sample_rate

    if header_duration_s is not None and abs(header_duration_s - duration_s) > HEADER_MISMATCH_WARN_S:
        logger.warning(
            'MP3 header suggests %.2f s but the decoded audio is %.2f s (sha256 %s)',
            header_duration_s, duration_s, sha256[:12],
        )
    logger.info(
        'decoded request audio: %d bytes, %.2f s at %d Hz (container claimed %s), sha256 %s',
        len(audio_bytes), duration_s, sample_rate,
        'unknown' if claimed_duration_s is None else f'{claimed_duration_s:.2f} s',
        sha256[:12],
    )
    return DecodedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        duration_s=duration_s,
        audio_sha256=sha256,
        n_bytes=len(audio_bytes),
        header_duration_s=header_duration_s,
    )
