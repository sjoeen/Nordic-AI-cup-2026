"""Download every model the configured pipeline needs into the local cache,
so the container (or a fresh host) can serve with HF_HUB_OFFLINE=1.

    python prefetch_models.py [configs/default.json]
"""

import json
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else 'configs/default.json'
    cfg = json.load(open(path, encoding='utf-8'))
    from faster_whisper import WhisperModel

    WhisperModel(cfg['asr']['model_size'], device='cpu', compute_type='int8')
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer(cfg['qa']['embed_model'], device='cpu')
    CrossEncoder(cfg['qa']['nli_model'], device='cpu')
    print('prefetched', cfg['asr']['model_size'], cfg['qa']['embed_model'], cfg['qa']['nli_model'])


if __name__ == '__main__':
    main()
