import os
import numpy as np
from data_loader import create_dataset_entry
from parser.loader import get_all_valid_difficulties
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from audio_processor import TOTAL_FEATURES

# ─────────────────────────────────────────────────────────────────
# Estrutura de arquivos gerados por dificuldade:
#
#   {base}_x.npy      — audio features       (num_frames, 8)
#   {base}_timing.npy — timing targets        (num_frames,)   float32 0/0.3/1.0
#   {base}_notes.npy  — sequência de notas    (N, 6)          float32
#                       colunas: has_note, hand, col, layer, cut, beat_norm
#   {base}_stars.npy  — estrelas              (1,)            float32
#
# Os arquivos _y.npy e _meta.npy do sistema anterior não existem mais.
# ─────────────────────────────────────────────────────────────────

STAR_BINS = [(0, 4), (4, 5.5), (5.5, 7), (7, 9), (9, 99)]
STAR_LABELS = ["0–4★", "4–5.5★", "5.5–7★", "7–9★", "9★+"]


def process_map_difficulty(map_folder, diff_info, processed_dir, force=False):
    map_id = os.path.basename(map_folder)
    diff_name, diff_filename, stars = diff_info
    base_name = f"{map_id}_{diff_name}"

    paths = {
        'x':      os.path.join(processed_dir, f"{base_name}_x.npy"),
        'timing': os.path.join(processed_dir, f"{base_name}_timing.npy"),
        'notes':  os.path.join(processed_dir, f"{base_name}_notes.npy"),
        'stars':  os.path.join(processed_dir, f"{base_name}_stars.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()) and not force:
        return None, base_name, "already_processed"

    try:
        result = create_dataset_entry(map_folder, diff_filename, diff_name, stars)
        if result is None:
            return None, base_name, "read_error"

        features, timing_targets, note_sequence, stars_val = result

        if features.shape[1] != TOTAL_FEATURES:
            return None, base_name, (
                f"feature_mismatch: got {features.shape[1]}, expected {TOTAL_FEATURES}"
            )

        np.save(paths['x'],      features)
        np.save(paths['timing'], timing_targets)
        np.save(paths['notes'],  note_sequence)
        np.save(paths['stars'],  np.array([stars_val]))

        return len(note_sequence), base_name, "success"

    except Exception as e:
        import traceback
        return None, base_name, f"fatal_error: {e}\n{traceback.format_exc()}"


def _print_star_report(star_counts):
    total = sum(star_counts.values()) or 1
    print("\n  Distribuição de estrelas processadas:")
    for (lo, hi), label in zip(STAR_BINS, STAR_LABELS):
        count = sum(v for k, v in star_counts.items() if lo <= k < hi)
        bar = "█" * int(count / total * 40)
        print(f"    {label:>8}  {bar:<40}  {count}")
    print()


def preprocess_all(raw_dir="data/raw_maps", processed_dir="data/processed",
                   max_workers=8, force=False):
    os.makedirs(processed_dir, exist_ok=True)

    map_folders = [
        os.path.join(raw_dir, d)
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    print(f"Encontradas {len(map_folders)} pastas de mapas.")
    print("Buscando dificuldades válidas com estrelas ScoreSaber...")

    all_tasks = []
    for folder in map_folders:
        for diff_info in get_all_valid_difficulties(folder):
            all_tasks.append((folder, diff_info))

    print(f"Total: {len(all_tasks)} dificuldades válidas.\n")
    if force:
        print("⚠️  Modo FORCE: todos os arquivos serão re-processados.\n")

    processed_count = skipped_count = error_count = 0
    total_notes  = 0
    star_counts  = defaultdict(int)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_map_difficulty, folder, diff_info, processed_dir, force): diff_info
            for folder, diff_info in all_tasks
        }
        for future in as_completed(futures):
            note_count, base_name, status = future.result()
            _, _, stars = futures[future]

            if status == "success":
                processed_count += 1
                total_notes += note_count or 0
                bucket = round(float(stars) * 2) / 2
                star_counts[bucket] += 1
                if processed_count % 50 == 0:
                    print(f"  ... {processed_count} processados")
            elif status == "already_processed":
                skipped_count += 1
            else:
                print(f"  [{'AVISO' if 'mismatch' in status else 'ERRO'}] {base_name}: {status}")
                error_count += 1

    _print_star_report(star_counts)
    print(f"Processados : {processed_count}")
    print(f"Pulados     : {skipped_count}")
    print(f"Erros       : {error_count}")
    print(f"Total notas : {total_notes:,}")

    if not force and skipped_count > 0:
        print(f"\n  ℹ️  Use --force para re-processar após mudar audio_processor.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force",         action="store_true")
    p.add_argument("--workers",       type=int, default=os.cpu_count() or 4)
    p.add_argument("--raw-dir",       default="data/raw_maps")
    p.add_argument("--processed-dir", default="data/processed")
    args = p.parse_args()

    preprocess_all(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_workers=args.workers,
        force=args.force,
    )