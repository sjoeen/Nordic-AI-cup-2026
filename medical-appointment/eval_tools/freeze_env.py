"""Record the environment and source snapshot behind a run.

    .venv/bin/python -m eval_tools.freeze_env --run-id baseline_v1

Writes ``manifests/env_<run_id>.json`` with the interpreter version,
``pip freeze`` of the running interpreter, versions of the libraries that
change numerical behaviour (torch, ctranslate2, faster-whisper, ...), CUDA
availability, the git commit, and the SHA-256 of every top-level ``.py``
file and of ``data/question_train.csv``. A result without this is a number
nobody can reproduce; with it, a later regression can be bisected to a
source change, a dependency bump or a hardware change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from eval_tools.split import safe_id

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_DIR = Path('manifests')
DEFAULT_EXTRA_FILES: Sequence[str] = ('data/question_train.csv',)
DEFAULT_LIBRARIES: Sequence[str] = (
    'torch', 'ctranslate2', 'faster-whisper', 'transformers', 'sentence-transformers',
    'numpy', 'pydantic', 'fastapi', 'llama-cpp-python',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(root: Path, extra_files: Iterable[str] = DEFAULT_EXTRA_FILES) -> Dict[str, str]:
    """``{relative_path: sha256}`` for top-level ``*.py`` files plus ``extra_files``.

    Only the top level is hashed on purpose: that is where the serving code
    lives, and it keeps the manifest independent of caches and virtualenvs
    that sit in subdirectories. Missing extra files are recorded as ``None``
    rather than failing, so the manifest still says what was absent.
    """
    root = Path(root)
    hashes: Dict[str, Any] = {
        path.name: sha256_file(path) for path in sorted(root.glob('*.py')) if path.is_file()
    }
    for relative in extra_files:
        path = root / relative
        hashes[str(relative)] = sha256_file(path) if path.is_file() else None
    return hashes


def pip_freeze(python: str = sys.executable) -> List[str]:
    """``pip freeze`` of ``python`` as a sorted list; the error text on failure."""
    try:
        completed = subprocess.run(
            [python, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f'<pip freeze failed: {type(exc).__name__}: {exc}>']
    if completed.returncode != 0:
        return [f'<pip freeze exited {completed.returncode}: {completed.stderr.strip()[:500]}>']
    return sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())


def library_versions(names: Iterable[str] = DEFAULT_LIBRARIES) -> Dict[str, Optional[str]]:
    """Installed versions by distribution name, ``None`` when not installed.

    Read from package metadata rather than by importing, because importing
    torch costs seconds and hundreds of megabytes on a machine shared with
    the models under test.
    """
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cuda_availability(import_torch: bool = True) -> Dict[str, Any]:
    """Whether torch sees a GPU; ``available`` is ``None`` when not checked."""
    if not import_torch:
        return {'available': None, 'checked': False}
    try:
        import torch

        available = bool(torch.cuda.is_available())
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else []
        return {'available': available, 'checked': True, 'devices': devices, 'torch_cuda': torch.version.cuda}
    except Exception as exc:  # noqa: BLE001 - a broken torch install is a finding, not a crash
        return {'available': None, 'checked': True, 'error': f'{type(exc).__name__}: {exc}'}


def git_commit(root: Path) -> Optional[str]:
    """The HEAD commit, or ``None`` outside a repository or without git."""
    try:
        completed = subprocess.run(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def build_manifest(
    run_id: str,
    root: Path = ROOT,
    freeze: Callable[[], List[str]] = pip_freeze,
    versions: Callable[[], Dict[str, Optional[str]]] = library_versions,
    cuda: Callable[[], Dict[str, Any]] = cuda_availability,
    commit: Callable[[Path], Optional[str]] = git_commit,
) -> Dict[str, Any]:
    """Everything the manifest holds; collectors are injectable for tests."""
    return {
        'run_id': safe_id(run_id, 'run_id'),
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'python': {'version': platform.python_version(), 'executable': sys.executable},
        'platform': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
        'git_commit': commit(Path(root)),
        'libraries': versions(),
        'cuda': cuda(),
        'source_sha256': source_hashes(Path(root)),
        'pip_freeze': freeze(),
    }


def write_manifest(manifest: Mapping[str, Any], manifest_dir: Path = DEFAULT_MANIFEST_DIR) -> Path:
    target = Path(manifest_dir) / f'env_{safe_id(manifest["run_id"], "run_id")}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manifest-dir', type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument('--no-torch', action='store_true', help='Skip importing torch for the CUDA check.')
    parser.add_argument('--python', default=sys.executable,
                        help='Interpreter whose packages are frozen (default: the running one).')
    args = parser.parse_args(argv)

    manifest = build_manifest(
        args.run_id,
        freeze=lambda: pip_freeze(args.python),
        cuda=lambda: cuda_availability(import_torch=not args.no_torch),
    )
    path = write_manifest(manifest, args.manifest_dir)
    libraries = ', '.join(f'{name} {version}' for name, version in manifest['libraries'].items() if version)
    print(f'wrote {path}')
    print(f'python {manifest["python"]["version"]}, commit {manifest["git_commit"]}, cuda {manifest["cuda"].get("available")}')
    print(f'libraries: {libraries}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
