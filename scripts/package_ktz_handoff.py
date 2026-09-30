"""Read-only scientific snapshot, ZIP64 archive and SHA-256 verification.

Exclude repository history/environments and non-selected galaxy analyses.
Freeze small/live files, stream large immutable inputs, reject changing large
files. No source file or background calculation is modified/stopped.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


GALAXIES = ('NGC628', 'NGC1566', 'NGC4254', 'NGC4303')
ROOT_SUFFIXES = {'.tex', '.bib', '.md', '.ipynb', '.py', '.pdf', '.txt', '.cls', '.sty', '.bst'}
TEXT_SUFFIXES = {'.tex', '.bib', '.md', '.ipynb', '.py', '.json', '.log', '.csv', '.tsv', '.txt'}
HIDDEN = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints', '.mplconfig'}
COMPILE_SUFFIXES = {'.aux', '.log', '.out', '.toc', '.synctex', '.gz', '.pyc'}


def now():
    return datetime.now(timezone.utc).isoformat()


def selected_galaxy(name):
    found = re.match(r'NGC0?(\d+)', name, flags=re.I)
    return not found or 'NGC'+found.group(1) in GALAXIES


def admissible(path, root):
    rel = path.relative_to(root)
    if any(part in HIDDEN or part.startswith('.') for part in rel.parts):
        return False
    if path.suffix in {'.pyc', '.tmp', '.part', '.lock'}:
        return False
    if any(not selected_galaxy(part) for part in rel.parts):
        return False
    if 'williams' in path.name.lower():
        return False
    if path.is_symlink() and not path.resolve().is_relative_to(root):
        return False
    return path.is_file()


def collect(root):
    files = set()
    for path in root.iterdir():
        if path.suffix in ROOT_SUFFIXES and admissible(path, root):
            files.add(path)
    for name in ('scripts', 'validation', 'figs', 'Li2021', 'outputs/implications'):
        for path in (root/name).rglob('*'):
            if admissible(path, root):
                if name == 'Li2021' and path.suffix in COMPILE_SUFFIXES:
                    continue
                files.add(path)
    for path in (root/'data').rglob('*'):
        if not admissible(path, root):
            continue
        rel = path.relative_to(root/'data')
        if rel.parts[0] != 'phangs':
            files.add(path)
        elif len(rel.parts) == 2:
            if path.name.startswith(('NGC0628_', 'NGC628_')) or path.suffix == '.md':
                files.add(path)
        elif rel.parts[1] in ('NGC1566', 'NGC4254', 'NGC4303'):
            files.add(path)
        elif rel.parts[1] == 'environment_masks' and path.suffix != '.gz':
            files.add(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def stat_key(stat):
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def stable_bytes(path):
    for attempt in range(4):
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            if stat_key(before) != stat_key(after) or len(payload) != after.st_size:
                raise RuntimeError('File changed while being frozen.')
            if path.suffix in ('.json', '.ipynb'):
                json.loads(payload)
            if path.suffix == '.npz':
                with zipfile.ZipFile(io.BytesIO(payload)) as z:
                    if z.testzip() is not None:
                        raise RuntimeError('Incomplete live NPZ checkpoint.')
            return payload, after
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
            if attempt == 3:
                raise
            time.sleep(.15)


def command(args, root):
    result = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def build(root, destination):
    root, destination = root.resolve(), destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite an existing archive or partial archive.')
    top = destination.stem
    progress_path = destination.with_suffix('.progress.json')
    start_time = now()
    files = collect(root)
    frozen, skipped = {}, []
    # Small/live files are frozen before streaming multi-GB completed inputs.
    for path in files:
        if path.suffix in TEXT_SUFFIXES or path.stat().st_size < 2**20:
            try:
                frozen[path] = stable_bytes(path)
            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                if 'cnp_refinement_v1' in path.parts and path.suffix != '.py':
                    skipped.append({'path': path.relative_to(root).as_posix(), 'reason': str(exc)})
                else:
                    raise
    files = [p for p in files if p.relative_to(root).as_posix() not in {s['path'] for s in skipped}]
    versions = {}
    for name in ('numpy', 'scipy', 'astropy', 'matplotlib', 'ipython', 'nbformat', 'nbclient', 'ipykernel'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    environment = {'python': sys.version, 'python_executable_on_original_host': sys.executable,
        'platform': platform.platform(), 'versions': versions,
        'note': 'Recorded runtime, not a clean-machine lockfile. No virtual environment or TeX installation included.'}
    live_status = {}
    base = 'outputs/implications/NGC628_source_integral_refit/'
    names = ['ngc628_source_precision_summary.json', 'ngc628_fft_real_map_benchmark.json',
        'ngc628_age_library_validation.json', 'ngc628_cnp_pilot_validation.json',
        'ngc628_cnp_pipeline_progress.json', 'cnp_refinement_v1/progress.json',
        'cnp_refinement_v1/integral_progress.json', 'cnp_refinement_v1/stage_1_validation.json',
        'cnp_refinement_v1/stage_2_validation.json']
    for name in names:
        path = root/(base+name)
        live_status[name] = json.loads(frozen[path][0]) if path in frozen else {'not_present_at_snapshot': True}
    status = {'snapshot_freeze_started_UTC': start_time,
        'snapshot_frozen_text_completed_UTC': now(), 'selected_galaxies': GALAXIES,
        'source_workspace': str(root), 'git_HEAD': command(['git', 'rev-parse', 'HEAD'], root),
        'tracked_worktree_status': command(['git', 'status', '--porcelain=v1', '--untracked-files=no'], root),
        'archive_is_global_atomic_snapshot': False, 'skipped_incomplete_live_files': skipped,
        'no_source_files_changed_or_jobs_stopped': True,
        'global_source_integral_new_seven_parameter_posterior_completed': False,
        'stored_PDFs_recompiled_for_packaging': False,
        'excluded': ['.git', '.venv', '__pycache__', '.ipynb_checkpoints', 'tmp',
            'non-selected galaxy analyses', 'Williams data/explicitly named caches', 'compilation caches'],
        'latest_source_integral_status': live_status}
    notes_path = root/'outputs/handoff/说明与思考_2026-09-19.md'
    notes, _ = stable_bytes(notes_path)
    generated = {'00_说明与思考.md': notes,
        'environment_snapshot.json': (json.dumps(environment, ensure_ascii=False, indent=2)+'\n').encode(),
        'snapshot_status.json': (json.dumps(status, ensure_ascii=False, indent=2)+'\n').encode()}
    records = []
    total_size = sum(path.stat().st_size for path in files)
    print(json.dumps({'phase': 'building snapshot ZIP', 'files': len(files),
        'source_bytes': total_size, 'destination': str(destination)}, ensure_ascii=False), flush=True)
    def progress(phase, completed):
        value = {'phase': phase, 'completed_source_files': completed,
            'source_files': len(files), 'archive_partial_bytes': partial.stat().st_size if partial.exists() else None,
            'updated_UTC': now(), 'destination': str(destination)}
        progress_path.write_text(json.dumps(value, indent=2)+'\n')
    with zipfile.ZipFile(partial, mode='x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, payload in generated.items():
            archive.writestr(f'{top}/{name}', payload, compresslevel=6)
            records.append((name, len(payload), hashlib.sha256(payload).hexdigest(), 'generated'))
        for i, path in enumerate(files):
            rel = path.relative_to(root).as_posix()
            stored = path.suffix.lower() in {'.npz', '.png', '.pdf', '.jpg', '.jpeg', '.zip'}
            compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            digest = hashlib.sha256()
            if path in frozen:
                payload, st = frozen.pop(path)
                archive.writestr(f'{top}/{rel}', payload, compress_type=compression, compresslevel=6)
                digest.update(payload)
                length = len(payload)
            else:
                before = path.stat()
                info = zipfile.ZipInfo.from_file(path, f'{top}/{rel}')
                info.compress_type = compression
                with path.open('rb') as source, archive.open(info, 'w') as target:
                    length = 0
                    for block in iter(lambda: source.read(2**20), b''):
                        target.write(block)
                        digest.update(block)
                        length += len(block)
                st = path.stat()
                if stat_key(st) != stat_key(before) or length != before.st_size:
                    raise RuntimeError('Large source changed during snapshot: '+rel)
            records.append((rel, length, digest.hexdigest(), datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()))
            if (i+1) % 25 == 0 or i+1 == len(files):
                progress('compressing stable source files', i+1)
                print(f'Archived {i+1}/{len(files)} source files', flush=True)
        manifest = 'path\tbytes\tsha256\tsource_mtime_UTC\n'+''.join(
            f'{name}\t{length}\t{digest}\t{mtime}\n' for name, length, digest, mtime in records)
        archive.writestr(f'{top}/manifest.tsv', manifest.encode(), compresslevel=6)
    progress('verifying every member SHA-256 and ZIP CRC', len(files))
    with zipfile.ZipFile(partial) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != len(records)+1:
            raise RuntimeError('Duplicate or missing archive members.')
        for name, length, expected, _ in records:
            digest, count = hashlib.sha256(), 0
            with archive.open(f'{top}/{name}') as source:
                for block in iter(lambda: source.read(2**20), b''):
                    digest.update(block)
                    count += len(block)
            if count != length or digest.hexdigest() != expected:
                raise RuntimeError('Archive checksum failed: '+name)
        archive.read(f'{top}/manifest.tsv')  # Also validates the manifest CRC.
    if destination.exists():
        raise FileExistsError('Destination appeared during packaging; refusing to overwrite it.')
    partial.rename(destination)
    digest = hashlib.sha256()
    with destination.open('rb') as source:
        for block in iter(lambda: source.read(2**20), b''):
            digest.update(block)
    summary = {'phase': 'complete and verified', 'archive': str(destination),
        'archive_bytes': destination.stat().st_size, 'source_files': len(files),
        'total_members': len(records)+1, 'archive_sha256': digest.hexdigest(),
        'skipped_incomplete_live_files': skipped, 'completed_UTC': now(),
        'verification': 'every listed member SHA-256/length and ZIP CRC, plus unique-member count'}
    progress_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
