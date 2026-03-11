import os
import numpy as np
from data_loader import create_dataset_entry
from parser.loader import get_all_valid_difficulties
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────
# Parâmetros da janela adaptativa
# ─────────────────────────────────────────────────────────────────
# A janela de contexto para calcular complexidade/verticalidade é
# proporcional ao número de frames da música, não fixa em 200.
# Isso evita janelas grandes demais em músicas curtas (que borrariam
# toda a informação) ou pequenas demais em músicas longas (que não
# capturam a estrutura de seção).

WINDOW_SECONDS   = 3.0   # Tamanho alvo da janela em segundos
STEP_SECONDS     = 0.25  # Resolução do cálculo (a cada ~0.25s)
HOP_LENGTH       = 512   # Deve ser igual ao usado em audio_processor.py
SR               = 22050 # Deve ser igual ao usado em audio_processor.py

# Frames por segundo com os parâmetros acima
FRAMES_PER_SEC = SR / HOP_LENGTH  # ~43.07 frames/s


def calculate_metadata(targets, num_frames_total=None):
    """
    Gera metadados (Complexidade e Verticalidade) baseados no contexto local do mapa.

    Melhorias vs versão anterior:
    - Janela adaptativa baseada em WINDOW_SECONDS (não fixa em 200 frames)
    - Step adaptativo baseado em STEP_SECONDS
    - Limites de densidade calibrados por densidade global do mapa
      (um mapa de 9★ tem densidade global muito mais alta que um de 3★ —
       usar thresholds fixos fazia tudo virar complexity=2 em mapas densos)
    - Suavização temporal dos metadados para evitar transições bruscas
      que confundiriam o modelo

    Args:
        targets        : ndarray (num_frames, 12) — target de posicionamento
        num_frames_total: int opcional (usado para janela mínima em músicas curtas)

    Returns:
        metadata : ndarray (num_frames, 2) — [complexity_class, vertical_class]
                   complexity : 0=simples, 1=médio, 2=denso
                   vertical   : 0=baixo,   1=médio,  2=alto
    """
    num_frames = targets.shape[0]
    metadata = np.zeros((num_frames, 2), dtype=np.float32)

    # ── Janela adaptativa ────────────────────────────────────────
    window = int(WINDOW_SECONDS * FRAMES_PER_SEC)
    step   = max(1, int(STEP_SECONDS * FRAMES_PER_SEC))

    # Garante janela mínima de 20 frames e máxima de num_frames//2
    window = max(20, min(window, num_frames // 2 if num_frames > 40 else num_frames))
    half_window = window // 2

    has_note = np.any(targets > 0.1, axis=1)  # (num_frames,) bool

    # ── Densidade global para calibrar thresholds ─────────────────
    # Em mapas muito densos (streams de 8★+), quase todos os frames têm nota.
    # Os thresholds fixos (0.05 / 0.15) causavam complexity=2 em tudo.
    # Calculamos thresholds relativos à mediana local para ser justo.
    global_density = float(np.mean(has_note))

    # Thresholds adaptativos:
    # - Se densidade global for alta (mapa difícil), elevamos o bar para complexity=2
    # - Se for baixa (mapa fácil), mantemos sensível
    thresh_medium = max(0.04, global_density * 0.4)   # 40% da densidade global
    thresh_high   = max(0.12, global_density * 0.85)  # 85% da densidade global

    # ── Cálculo frame a frame (com step) ─────────────────────────
    raw_comp = np.zeros(num_frames, dtype=np.float32)
    raw_vert = np.zeros(num_frames, dtype=np.float32)

    for i in range(0, num_frames, step):
        start = max(0, i - half_window)
        end   = min(num_frames, i + half_window)

        # Complexidade — baseada em densidade local de notas
        chunk_notes = has_note[start:end]
        density = float(np.mean(chunk_notes)) if len(chunk_notes) > 0 else 0.0

        if density >= thresh_high:
            comp_val = 2.0
        elif density >= thresh_medium:
            comp_val = 1.0
        else:
            comp_val = 0.0

        # Verticalidade — altura média ponderada das notas na janela
        chunk_targets = targets[start:end]
        l0 = float(np.sum(chunk_targets[:, 0:4]))   # layer 0 (baixo)
        l1 = float(np.sum(chunk_targets[:, 4:8]))   # layer 1 (meio)
        l2 = float(np.sum(chunk_targets[:, 8:12]))  # layer 2 (alto)
        total = l0 + l1 + l2 + 1e-6

        avg_height = (0.0 * l0 + 1.0 * l1 + 2.0 * l2) / total
        vert_val = float(round(avg_height))

        # Preenche o step inteiro com o valor calculado
        fill_end = min(num_frames, i + step)
        raw_comp[i:fill_end] = comp_val
        raw_vert[i:fill_end] = vert_val

    # ── Suavização temporal ───────────────────────────────────────
    # Evita transições bruscas frame-a-frame que introduziriam ruído
    # no treino. Usamos uma média móvel leve (~0.5s).
    smooth_w = max(1, int(0.5 * FRAMES_PER_SEC))
    kernel   = np.ones(smooth_w) / smooth_w

    smooth_comp = np.convolve(raw_comp, kernel, mode='same')
    smooth_vert = np.convolve(raw_vert, kernel, mode='same')

    # Re-arredonda para classe inteira após suavização
    metadata[:, 0] = np.round(smooth_comp).clip(0, 2)
    metadata[:, 1] = np.round(smooth_vert).clip(0, 2)

    return metadata


def process_map_difficulty(map_folder, diff_info, processed_dir, force=False):
    """
    Processa uma única dificuldade de um mapa.
    diff_info: (diff_name, diff_filename, stars)

    Args:
        force: se True, reprocessa mesmo que os arquivos já existam.
               Use force=True ao rodar após atualizar audio_processor.py.
    """
    map_id = os.path.basename(map_folder)
    diff_name, diff_filename, stars = diff_info

    base_name = f"{map_id}_{diff_name}"

    save_path_x     = os.path.join(processed_dir, f"{base_name}_x.npy")
    save_path_y     = os.path.join(processed_dir, f"{base_name}_y.npy")
    save_path_meta  = os.path.join(processed_dir, f"{base_name}_meta.npy")
    save_path_stars = os.path.join(processed_dir, f"{base_name}_stars.npy")

    all_exist = all(os.path.exists(p) for p in [
        save_path_x, save_path_y, save_path_meta, save_path_stars
    ])

    if all_exist and not force:
        return None, base_name, "already_processed"

    try:
        data = create_dataset_entry(map_folder, diff_filename, diff_name, stars)

        if data is None:
            return None, base_name, "read_error"

        features, targets, vertical_dist, stars_val = data

        # Valida dimensão de features — deve ser 93 após a atualização do audio_processor
        expected_features = 93
        if features.shape[1] != expected_features:
            return None, base_name, (
                f"feature_mismatch: got {features.shape[1]}, expected {expected_features}. "
                f"Verifique se audio_processor.py foi atualizado."
            )

        metadata = calculate_metadata(targets)

        np.save(save_path_x,     features)
        np.save(save_path_y,     targets)
        np.save(save_path_meta,  metadata)
        np.save(save_path_stars, np.array([stars_val]))

        return vertical_dist, base_name, "success"

    except Exception as e:
        import traceback
        return None, base_name, f"fatal_error: {e}\n{traceback.format_exc()}"


def _print_star_report(star_counts):
    """Imprime um relatório visual da distribuição de estrelas processadas."""
    bins = [(0, 4), (4, 5.5), (5.5, 7), (7, 9), (9, 99)]
    labels = ["0–4★", "4–5.5★", "5.5–7★", "7–9★", "9★+"]
    total = sum(star_counts.values()) or 1

    print("\n  Distribuição de estrelas processadas:")
    for (lo, hi), label in zip(bins, labels):
        count = sum(v for k, v in star_counts.items() if lo <= k < hi)
        bar = "█" * int(count / total * 40)
        print(f"    {label:>8}  {bar:<40}  {count}")
    print()


def preprocess_all(raw_dir="data/raw_maps", processed_dir="data/processed",
                   max_workers=8, force=False):
    """
    Processa todos os mapas do diretório raw.

    Args:
        force: passa True para re-processar todos os arquivos
               (necessário após atualizar audio_processor.py para 93 features).
    """
    os.makedirs(processed_dir, exist_ok=True)

    map_folders = [
        os.path.join(raw_dir, d)
        for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ]

    print(f"Encontradas {len(map_folders)} pastas de mapas.")
    print("Buscando todas as dificuldades válidas (com estrelas ScoreSaber)...")

    all_tasks = []
    for folder in map_folders:
        valid_diffs = get_all_valid_difficulties(folder)
        for diff_info in valid_diffs:
            all_tasks.append((folder, diff_info))

    print(f"Total de {len(all_tasks)} dificuldades válidas encontradas.")

    if force:
        print("⚠️  Modo FORCE ativo: todos os arquivos serão re-processados.\n")
    else:
        print("   (Use force=True para re-processar arquivos existentes)\n")

    print("Processando features, targets, metadados adaptativos e estrelas...\n")

    processed_count = 0
    skipped_count   = 0
    error_count     = 0
    star_counts     = defaultdict(int)  # Para o relatório de distribuição

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_map_difficulty, folder, diff_info, processed_dir, force): diff_info
            for folder, diff_info in all_tasks
        }

        for future in as_completed(futures):
            dist, base_name, status = future.result()
            diff_info = futures[future]
            _, _, stars = diff_info

            if status == "success":
                processed_count += 1
                # Agrupa por faixa de 0.5★ para o relatório
                star_bucket = round(float(stars) * 2) / 2
                star_counts[star_bucket] += 1
                # Log resumido para não poluir o terminal
                if processed_count % 50 == 0:
                    print(f"  ... {processed_count} processados até agora")

            elif status == "already_processed":
                skipped_count += 1

            elif status == "read_error":
                print(f"  [ERRO] Falha ao ler: {base_name}")
                error_count += 1

            elif status.startswith("feature_mismatch"):
                print(f"  [AVISO] {base_name}: {status}")
                error_count += 1

            elif status.startswith("fatal_error"):
                print(f"  [FATAL] {base_name}:\n{status}")
                error_count += 1

    _print_star_report(star_counts)

    print(f"Concluído!")
    print(f"  Processados : {processed_count}")
    print(f"  Pulados     : {skipped_count}")
    print(f"  Erros       : {error_count}")

    if error_count > 0:
        print(f"\n  ⚠️  {error_count} erros encontrados. Verifique os logs acima.")

    if skipped_count > 0 and not force:
        print(f"\n  ℹ️  {skipped_count} arquivos pulados (já processados).")
        print("      Se você atualizou audio_processor.py, rode com force=True para re-gerar.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pré-processa o dataset de mapas Beat Saber.")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-processa todos os arquivos, mesmo os já existentes. "
             "Necessário após atualizar audio_processor.py."
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--raw-dir",       default="data/raw_maps")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    preprocess_all(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_workers=args.workers,
        force=args.force,
    )