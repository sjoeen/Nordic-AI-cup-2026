"""Measure faster-whisper speed and memory on this machine for a few clips.

    .venv/bin/python -m eval_tools.bench_asr --models base small --clips sample_4 sample_20

Prints real-time factor per model so the serving config can be sized against
the 60-second budget. Development tool only; nothing here runs at serve time.
"""

import argparse
import io
import resource
import time

from faster_whisper import WhisperModel, decode_audio

from utils import audio_filename_for_transcript, load_sample_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['base', 'small'])
    parser.add_argument('--clips', nargs='+', default=['sample_4', 'sample_20'])
    parser.add_argument('--compute-type', default='int8')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--vad', action='store_true')
    args = parser.parse_args()

    for model_name in args.models:
        t0 = time.monotonic()
        model = WhisperModel(
            model_name, device=args.device, compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
        )
        load_s = time.monotonic() - t0
        print(f'[{model_name}] load {load_s:.1f}s', flush=True)
        for clip in args.clips:
            audio_bytes = load_sample_audio(audio_filename_for_transcript(clip))
            waveform = decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)
            duration = len(waveform) / 16000
            t0 = time.monotonic()
            segments, info = model.transcribe(
                waveform, language='en', beam_size=args.beam_size,
                word_timestamps=True, vad_filter=args.vad,
                condition_on_previous_text=False,
            )
            segments = list(segments)
            took = time.monotonic() - t0
            n_words = sum(len(s.words or []) for s in segments)
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(
                f'[{model_name}] {clip}: audio {duration:.1f}s, transcribe '
                f'{took:.1f}s, RTF {took / duration:.3f}, {len(segments)} '
                f'segments, {n_words} words, peak RSS {rss_mb:.0f} MB',
                flush=True,
            )
            if segments:
                print(f'    first: [{segments[0].start:.2f}-{segments[0].end:.2f}] {segments[0].text.strip()[:100]}')
        del model


if __name__ == '__main__':
    main()
