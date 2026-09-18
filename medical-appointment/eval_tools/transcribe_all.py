"""Decode and transcribe the supplied conversations into cached contexts.

    .venv/bin/python -m eval_tools.transcribe_all --split dev --asr-model-size small
    .venv/bin/python -m eval_tools.transcribe_all --split all --limit 2 --out-dir work/transcripts

Every downstream offline tool (QA evaluation, the evidence oracle, error
analysis) reads ``<out-dir>/<asr_config_hash>/contexts/<transcript_id>.json``
rather than running ASR again, so a QA experiment costs seconds instead of
the ten minutes a full transcription pass takes on the dev machine. The
raw whisper output is cached next to it by ``CachingASRBackend`` under
``<out-dir>/<asr_config_hash>/<audio_sha256>.json``, so re-segmentation
parameters can change without re-running the model.

The audio goes through ``audio_io.decode_request_audio`` on the base64 form,
the same path a request takes, so cached transcripts match what the server
would produce for the same bytes. Sibling modules are imported inside the
functions that need them: this file must import cleanly without torch or
faster-whisper installed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from eval_tools.split import DEFAULT_SPLIT_PATH, SPLIT_NAMES, safe_id, select_transcripts

DEFAULT_OUT_DIR = Path('work') / 'transcripts'
TABLE_COLUMNS = ('transcript', 'audio_s', 'asr_s', 'rtf', 'units', 'words', 'anomalies', 'truncated', 'cache')


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def transcript_entry(
    transcript_id: str,
    audio_sha256: str,
    duration_s: float,
    seconds: float,
    run_info: Mapping[str, Any],
    n_units: int,
    n_words: int,
    n_anomalies: int,
) -> Dict[str, Any]:
    """One manifest row: what was transcribed, how long it took, what went wrong.

    ``rtf`` (real-time factor) is what sizes the serving budget, so it is
    stored rather than left for the reader to derive; it is ``None`` for an
    empty recording rather than a division error.
    """
    return {
        'transcript_id': safe_id(transcript_id),
        'audio_sha256': str(audio_sha256),
        'duration_s': float(duration_s),
        'seconds': float(seconds),
        'rtf': float(seconds) / float(duration_s) if duration_s and duration_s > 0 else None,
        'truncated': bool(run_info.get('truncated', False)),
        'error': run_info.get('error'),
        'cache': run_info.get('cache'),
        'n_units': int(n_units),
        'n_words': int(n_words),
        'anomalies': int(n_anomalies),
    }


def merge_manifest(
    existing: Optional[Mapping[str, Any]],
    header: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """The manifest after this run, keeping earlier transcripts and real timings.

    Runs are incremental (``--limit``, a split at a time), so entries from an
    earlier run under the same config hash survive. A cache hit reports a
    near-zero ``seconds``; when the previous entry measured a real
    transcription, its timing is kept so the manifest stays a record of what
    the model costs, and the entry is marked as served from cache.
    """
    merged: Dict[str, Any] = dict(header)
    previous: Dict[str, Any] = {}
    if existing and isinstance(existing.get('transcripts'), dict):
        previous = {str(key): dict(value) for key, value in existing['transcripts'].items()}
    for transcript_id, entry in entries.items():
        new_entry = dict(entry)
        old_entry = previous.get(transcript_id)
        if new_entry.get('cache') == 'hit' and old_entry and old_entry.get('cache') != 'hit':
            for key in ('seconds', 'rtf', 'truncated', 'error'):
                if key in old_entry:
                    new_entry[key] = old_entry[key]
        previous[transcript_id] = new_entry
    merged['transcripts'] = previous
    merged['updated'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return merged


def format_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """Fixed-width summary of manifest entries plus a totals line."""
    header = (
        f'{"transcript":<12} {"audio_s":>8} {"asr_s":>7} {"rtf":>6} {"units":>5} '
        f'{"words":>5} {"anom":>4} {"trunc":>5} {"cache":>6}'
    )
    lines = [header]
    total_audio = total_seconds = 0.0
    for entry in entries:
        rtf = entry.get('rtf')
        lines.append(
            f'{entry["transcript_id"]:<12} {entry["duration_s"]:8.1f} {entry["seconds"]:7.1f} '
            f'{(f"{rtf:.3f}" if rtf is not None else "-"):>6} {entry["n_units"]:5d} '
            f'{entry["n_words"]:5d} {entry["anomalies"]:4d} '
            f'{("yes" if entry.get("truncated") else "no"):>5} {str(entry.get("cache") or "-"):>6}'
        )
        total_audio += float(entry['duration_s'])
        total_seconds += float(entry['seconds'])
    overall = total_seconds / total_audio if total_audio > 0 else None
    lines.append(
        f'{"total":<12} {total_audio:8.1f} {total_seconds:7.1f} '
        f'{(f"{overall:.3f}" if overall is not None else "-"):>6}  ({len(entries)} transcripts)'
    )
    return '\n'.join(lines)


def write_json_atomic(payload: Mapping[str, Any], path: Path) -> Path:
    """Write JSON via a temp file and rename, so a reader never sees half a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        'w', encoding='utf-8', dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, indent=1, default=str)
            handle.write('\n')
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return target


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    """A JSON object from ``path``, or ``None`` when absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# The transcription loop (imports the model stack lazily)
# --------------------------------------------------------------------------- #

def transcribe_transcripts(
    transcript_ids: Sequence[str],
    config: Any,
    out_dir: Path,
    resegment_kwargs: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Transcribe each id into ``out_dir/<config_hash>/contexts/`` and return entries."""
    from asr_backend import build_asr_backend
    from audio_io import decode_request_audio
    from transcript import build_context, save_context
    from utils import audio_filename_for_transcript, encode_audio, load_sample_audio

    out_dir = Path(out_dir)
    contexts_dir = out_dir / config.config_hash / 'contexts'
    backend = build_asr_backend(config, cache_dir=out_dir)
    started = time.monotonic()
    backend.warm_up()
    print(f'[{config.model_id}] warm-up {time.monotonic() - started:.1f}s, hash {config.config_hash[:12]}', flush=True)

    entries: List[Dict[str, Any]] = []
    for transcript_id in transcript_ids:
        transcript_id = safe_id(transcript_id)
        audio_bytes = load_sample_audio(audio_filename_for_transcript(transcript_id))
        decoded = decode_request_audio(encode_audio(audio_bytes))
        t0 = time.monotonic()
        units = backend.transcribe(
            decoded.waveform, decoded.sample_rate, deadline=None, cache_key=decoded.audio_sha256,
        )
        seconds = time.monotonic() - t0
        run_info = dict(getattr(backend, 'last_run_info', None) or {})
        context = build_context(
            units, decoded.audio_sha256, decoded.duration_s, backend.model_id,
            backend.config_hash, resegment_kwargs=dict(resegment_kwargs),
        )
        save_context(context, contexts_dir / f'{transcript_id}.json')
        entry = transcript_entry(
            transcript_id, decoded.audio_sha256, decoded.duration_s, seconds, run_info,
            n_units=len(context.units),
            n_words=sum(len(unit.words) for unit in context.units),
            n_anomalies=len(context.diagnostics.get('unit_anomalies', [])),
        )
        entries.append(entry)
        rtf = entry['rtf']
        print(
            f'{transcript_id:<12} {decoded.duration_s:6.1f}s audio, {seconds:5.1f}s ASR'
            f'{f", RTF {rtf:.3f}" if rtf is not None else ""}, {entry["n_units"]} units'
            f'{" (cache hit)" if entry["cache"] == "hit" else ""}'
            f'{" TRUNCATED" if entry["truncated"] else ""}',
            flush=True,
        )
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', choices=SPLIT_NAMES, default='all')
    parser.add_argument('--split-file', type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument('--transcripts', nargs='*', default=None,
                        help='Explicit transcript ids (overrides --split).')
    parser.add_argument('--asr-model-size', default='small')
    parser.add_argument('--compute-type', default='auto')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--vad', action='store_true', help='Enable the VAD filter.')
    parser.add_argument('--local-files-only', action='store_true',
                        help='Never contact the model hub; fail if the weights are not cached.')
    parser.add_argument('--download-root', default=None,
                        help='Directory holding (or receiving) the model weights.')
    parser.add_argument('--max-unit-s', type=float, default=8.0)
    parser.add_argument('--pause-gap-s', type=float, default=0.7)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args(argv)

    from asr_backend import ASRConfig

    config = ASRConfig(
        model_size=args.asr_model_size, device=args.device, compute_type=args.compute_type,
        cpu_threads=args.cpu_threads, beam_size=args.beam_size, vad_filter=args.vad,
        local_files_only=args.local_files_only, download_root=args.download_root,
    )
    resegment_kwargs = {'max_unit_s': args.max_unit_s, 'pause_gap_s': args.pause_gap_s}
    transcript_ids = (
        [safe_id(item) for item in args.transcripts]
        if args.transcripts else select_transcripts(args.split, args.split_file)
    )
    if args.limit is not None:
        transcript_ids = transcript_ids[:max(0, args.limit)]
    if not transcript_ids:
        print('nothing selected')
        return 1

    entries = transcribe_transcripts(transcript_ids, config, args.out_dir, resegment_kwargs)

    run_dir = Path(args.out_dir) / config.config_hash
    header = {
        'asr_model_id': config.model_id,
        'asr_config': config.to_dict(),
        'config_hash': config.config_hash,
        'resolved_runtime': list(config.resolved_runtime()),
        'resegment': resegment_kwargs,
    }
    manifest = merge_manifest(
        read_json_if_exists(run_dir / 'manifest.json'), header,
        {entry['transcript_id']: entry for entry in entries},
    )
    write_json_atomic(manifest, run_dir / 'manifest.json')
    print()
    print(format_table(entries))
    print(f'contexts in {run_dir / "contexts"}, manifest {run_dir / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
