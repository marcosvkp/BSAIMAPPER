"""
preprocess_data.py — Pré-processamento V5

Gera e salva por dificuldade em data/processed/:

  {base}_mel.npy    : (num_frames, 64)  espectrograma mel
  {base}_ctx.npy    : (num_frames, 8)   features de contexto
  {base}_timing.npy : (num_frames,)     targets de timing
  {base}_notes.npy  : (N, 8)           sequência de notas
  {base}_stars.npy  : (1,)             estrelas

Uso:
  python preprocess_data.py                   # processa tudo
  python preprocess_data.py --force           # re-processa mesmo os já existentes
  python preprocess_data.py --workers 12      # mais threads
"""

import os
import argparse
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_loader import create_dataset_entry
from parser.loader import get_all_valid_difficulties
from audio_processor import TOTAL_FEATURES, MEL_BINS, CTX_FEATS

STAR_BINS   = [(0, 4), (4, 5.5), (5.5, 7), (7, 9), (9, 99)]
STAR_LABELS = ["0–4★", "4–5.5★", "5.5–7★", "7–9★", "9★+"]


def process_one(map_folder: str, diff_info, processed_dir: str, force: bool):
    """Processa uma dificuldade e salva os arquivos .npy."""
    map_id = os.path.basename(map_folder)
    diff_name, diff_filename, stars = diff_info
    base = f"{map_id}_{diff_name}"

    paths = {
        'mel':    os.path.join(processed_dir, f"{base}_mel.npy"),
        'ctx':    os.path.join(processed_dir, f"{base}_ctx.npy"),
        'timing': os.path.join(processed_dir, f"{base}_timing.npy"),
        'notes':  os.path.join(processed_dir, f"{base}_notes.npy"),
        'stars':  os.path.join(processed_dir, f"{base}_stars.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()) and not force:
        return None, base, "skipped"

    try:
        result = create_dataset_entry(map_folder, diff_filename, diff_name, stars)
        if result is None:
            return None, base, "read_error"

        mel_spec, ctx_feats, timing, notes, stars_val = result

        # Validações de shape
        if mel_spec.shape[1] != MEL_BINS:
            return None, base, f"mel_bins_mismatch: {mel_spec.shape[1]}"
        if ctx_feats.shape[1] != CTX_FEATS:
            return None, base, f"ctx_feats_mismatch: {ctx_feats.shape[1]}"

        np.save(paths['mel'],    mel_spec.astype(np.float32))
        np.save(paths['ctx'],    ctx_feats.astype(np.float32))
        np.save(paths['timing'], timing.astype(np.float32))
        np.save(paths['notes'],  notes.astype(np.float32))
        np.save(paths['stars'],  np.array([stars_val], dtype=np.float32))

        return len(notes), base, "ok"

    except Exception as e:
        import traceback
        return None, base, f"error: {e}\n{traceback.format_exc()}"


def _star_report(counts: dict):
    total = sum(counts.values()) or 1
    print("\n  Distribuição de estrelas processadas:")
    for (lo, hi), label in zip(STAR_BINS, STAR_LABELS):
        n   = sum(v for k, v in counts.items() if lo <= k < hi)
        bar = "█" * int(n / total * 40)
        print(f"    {label:>8}  {bar:<40}  {n}")
    print()


def preprocess_all(raw_dir="data/raw_maps", processed_dir="data/processed",
                   max_workers=8, force=False):
    os.makedirs(processed_dir, exist_ok=True)

    folders = [
        os.path.join(raw_dir, d)
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    print(f"  {len(folders)} pastas de mapas encontradas.")
    print("  Coletando dificuldades válidas...")

    tasks = []
    for folder in folders:
        for diff_info in get_all_valid_difficulties(folder):
            tasks.append((folder, diff_info))

    print(f"  {len(tasks)} dificuldades válidas.\n")
    if force:
        print("  ⚠  Modo --force: tudo será re-processado.\n")

    ok_count = skip_count = err_count = 0
    total_notes = 0
    star_counts: dict = defaultdict(int)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(process_one, folder, diff_info, processed_dir, force): diff_info
            for folder, diff_info in tasks
        }
        for fut in as_completed(futures):
            note_count, base, status = fut.result()
            _, _, stars = futures[fut]

            if status == "ok":
                ok_count    += 1
                total_notes += note_count or 0
                bucket = round(float(stars) * 2) / 2
                star_counts[bucket] += 1
                if ok_count % 100 == 0:
                    print(f"  ... {ok_count} processados")
            elif status == "skipped":
                skip_count += 1
            else:
                print(f"  [ERRO] {base}: {status}")
                err_count += 1

    _star_report(star_counts)
    print(f"  Processados : {ok_count}")
    print(f"  Pulados     : {skip_count}")
    print(f"  Erros       : {err_count}")
    print(f"  Total notas : {total_notes:,}")

    if not force and skip_count > 0:
        print("\n  ℹ  Use --force para re-processar após alterar audio_processor.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pré-processa mapas ranqueados para treino.")
    p.add_argument("--force",         action="store_true",
                   help="Re-processa mesmo os já existentes")
    p.add_argument("--workers",       type=int, default=os.cpu_count() or 4,
                   help="Threads paralelas")
    p.add_argument("--raw-dir",       default="data/raw_maps",
                   help="Pasta com mapas brutos")
    p.add_argument("--processed-dir", default="data/processed",
                   help="Pasta de saída para .npy")
    args = p.parse_args()

    preprocess_all(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_workers=args.workers,
        force=args.force,
    )
