"""Convert raw faster-whisper JSON dumps (work/raw_whisper/<model>/<tid>.json,
written by the integrator's bootstrap script) into cached AudioContexts in the
layout ``transcribe_all.py`` produces, so offline evaluation can start
without re-running ASR.

    .venv/bin/python -m eval_tools.import_raw_whisper --model base --out-dir work/transcripts
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def raw_to_segments(raw):
    segments = []
    for s in raw['segments']:
        words = [SimpleNamespace(start=w['start'], end=w['end'], word=w['word'], probability=w.get('probability'))
                 for w in s.get('words', [])]
        segments.append(SimpleNamespace(start=s['start'], end=s['end'], text=s['text'], words=words or None,
                                        avg_logprob=s.get('avg_logprob'), no_speech_prob=s.get('no_speech_prob')))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--raw-dir', default='work/raw_whisper')
    parser.add_argument('--out-dir', default='work/transcripts')
    parser.add_argument('--max-unit-s', type=float, default=8.0)
    parser.add_argument('--pause-gap-s', type=float, default=0.7)
    args = parser.parse_args()

    from asr_backend import ASRConfig, segments_to_units
    from transcript import build_context, save_context

    config = ASRConfig(model_size=args.model, device='cpu', compute_type='int8', beam_size=1)
    out = Path(args.out_dir) / config.config_hash
    (out / 'contexts').mkdir(parents=True, exist_ok=True)
    manifest = {'asr_model_id': config.model_id, 'asr_config': config.__dict__, 'config_hash': config.config_hash,
                'resegment': {'max_unit_s': args.max_unit_s, 'pause_gap_s': args.pause_gap_s}, 'transcripts': {}}
    raw_files = sorted(Path(args.raw_dir, args.model).glob('sample_*.json'))
    for path in raw_files:
        raw = json.loads(path.read_text())
        units = segments_to_units(raw_to_segments(raw))
        context = build_context(units, audio_sha256=raw['audio_sha256'], duration_s=raw['duration_s'],
                                asr_model_id=config.model_id, asr_config_hash=config.config_hash,
                                resegment_kwargs={'max_unit_s': args.max_unit_s, 'pause_gap_s': args.pause_gap_s})
        save_context(context, out / 'contexts' / f"{raw['transcript_id']}.json")
        manifest['transcripts'][raw['transcript_id']] = {
            'seconds': raw['seconds'], 'rtf': raw['rtf'], 'n_units': len(context.units),
            'n_words': sum(len(u.words) for u in context.units), 'duration_s': raw['duration_s'],
            'anomalies': len(context.diagnostics.get('unit_anomalies', [])),
        }
        print(f"{raw['transcript_id']}: {len(context.units)} units from {len(raw['segments'])} segments")
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    print(f'wrote {len(raw_files)} contexts to {out}')


if __name__ == '__main__':
    main()
