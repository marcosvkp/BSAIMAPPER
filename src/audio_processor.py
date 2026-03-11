import librosa
import numpy as np
import os
import subprocess
import imageio_ffmpeg

# ─────────────────────────────────────────────────────────────────
# Número total de features extraídas por `extract_features`:
#   80  Mel Spectrogram (dB)
#    2  Grid rhythmic (beat / half-beat pulses)
#    1  Posição relativa na música  (0.0 → 1.0)
#    1  RMS energia local normalizada
#    1  Onset strength normalizada
#    3  Variância de energia em 3 janelas (0.5s / 2s / 5s)
#    1  Spectral centroid normalizado  (brilho tonal)
#    1  Spectral flatness normalizado  (ruído vs tom)
#    1  Zero-crossing rate normalizada  (percussividade)
#    2  Detecção de seção: drop (alta energia) / breakdown (baixa)
#   ─────
#   93  TOTAL  →  lembre de atualizar input_size=93 no models_optimized.py
# ─────────────────────────────────────────────────────────────────

TOTAL_FEATURES = 93


def detect_bpm(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        return float(tempo[0])
    return float(tempo)


def _safe_normalize(arr):
    """Normaliza para [0, 1] de forma segura."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _rolling_variance(signal, window_frames):
    """
    Variância local usando convolução (eficiente).
    Retorna array com mesmo comprimento de `signal`.
    """
    n = len(signal)
    half = window_frames // 2
    padded = np.pad(signal, half, mode='edge')
    var = np.array([
        np.var(padded[i:i + window_frames])
        for i in range(n)
    ], dtype=np.float32)
    return var


def extract_features(file_path, bpm, sr=22050, n_mels=80, hop_length=512):
    """
    Extrai features ricas de áudio para o modelo.

    Retorna:
        features : ndarray (num_frames, TOTAL_FEATURES)
        sr       : int
        hop_length : int
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        num_samples = len(y)
        frame_duration = hop_length / sr  # segundos por frame

        # ── 1. Mel Spectrogram ────────────────────────────────────────
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mels_db = librosa.power_to_db(mels, ref=np.max).T  # (T, 80)
        num_frames = mels_db.shape[0]

        # ── 2. Grid Features ──────────────────────────────────────────
        seconds_per_beat = 60.0 / bpm
        frames_per_beat = seconds_per_beat / frame_duration
        grid_features = np.zeros((num_frames, 2), dtype=np.float32)

        for i in range(num_frames):
            beat_pos = i / frames_per_beat
            dist_beat = abs(beat_pos - round(beat_pos))
            if dist_beat < 0.1:
                grid_features[i, 0] = 1.0 - (dist_beat * 10)
            dist_half = abs((beat_pos * 2) - round(beat_pos * 2))
            if dist_half < 0.1:
                grid_features[i, 1] = 1.0 - (dist_half * 10)

        # ── 3. Posição relativa na música (estrutura global) ──────────
        # O modelo aprende que versos, refrões e drops ocorrem em posições
        # características (ex: drops geralmente após ~40% da música).
        song_position = np.linspace(0.0, 1.0, num_frames, dtype=np.float32).reshape(-1, 1)

        # ── 4. RMS energia local normalizada ─────────────────────────
        rms_raw = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_raw = rms_raw[:num_frames]
        if len(rms_raw) < num_frames:
            rms_raw = np.pad(rms_raw, (0, num_frames - len(rms_raw)))
        rms_norm = _safe_normalize(rms_raw).reshape(-1, 1)

        # ── 5. Onset Strength normalizada ─────────────────────────────
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_env = onset_env[:num_frames]
        if len(onset_env) < num_frames:
            onset_env = np.pad(onset_env, (0, num_frames - len(onset_env)))
        onset_norm = _safe_normalize(onset_env).reshape(-1, 1)

        # ── 6. Variância de energia em 3 janelas temporais ────────────
        # Diferencia seções estáveis (versos) de seções dinâmicas (drops).
        # Janelas: ~0.5s, ~2s, ~5s
        w_short  = max(1, int(0.5 / frame_duration))
        w_medium = max(1, int(2.0 / frame_duration))
        w_long   = max(1, int(5.0 / frame_duration))

        var_short  = _safe_normalize(_rolling_variance(rms_raw, w_short)).reshape(-1, 1)
        var_medium = _safe_normalize(_rolling_variance(rms_raw, w_medium)).reshape(-1, 1)
        var_long   = _safe_normalize(_rolling_variance(rms_raw, w_long)).reshape(-1, 1)

        # ── 7. Spectral Centroid (brilho tonal) ───────────────────────
        # Drops costumam ter centroid alto (muita energia em freq. agudas).
        # Breakdowns têm centroid baixo.
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        centroid = centroid[:num_frames]
        if len(centroid) < num_frames:
            centroid = np.pad(centroid, (0, num_frames - len(centroid)))
        centroid_norm = _safe_normalize(centroid).reshape(-1, 1)

        # ── 8. Spectral Flatness (ruído vs tom puro) ──────────────────
        # Alta flatness = seção percussiva/noise (streams, bursts).
        # Baixa flatness = seção melódica (single notes, padrões simples).
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        flatness = flatness[:num_frames]
        if len(flatness) < num_frames:
            flatness = np.pad(flatness, (0, num_frames - len(flatness)))
        flatness_norm = _safe_normalize(flatness).reshape(-1, 1)

        # ── 9. Zero-Crossing Rate (percussividade) ────────────────────
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        zcr = zcr[:num_frames]
        if len(zcr) < num_frames:
            zcr = np.pad(zcr, (0, num_frames - len(zcr)))
        zcr_norm = _safe_normalize(zcr).reshape(-1, 1)

        # ── 10. Detecção binária de seção: Drop / Breakdown ───────────
        # Dois canais que "acendem" quando a música está em estado extremo.
        # Drop    = energia alta E variância alta (seção explosiva)
        # Breakdown = energia baixa E variância baixa (seção calma)
        #
        # Usamos suavização para evitar flicker frame-a-frame.
        smooth_w = max(1, int(1.0 / frame_duration))  # ~1 segundo
        rms_smooth = np.convolve(rms_raw, np.ones(smooth_w) / smooth_w, mode='same')
        rms_smooth_norm = _safe_normalize(rms_smooth)
        var_smooth_norm = _safe_normalize(
            np.convolve(_rolling_variance(rms_raw, w_medium).flatten(),
                        np.ones(smooth_w) / smooth_w, mode='same')
        )

        is_drop = ((rms_smooth_norm > 0.65) & (var_smooth_norm > 0.50)).astype(np.float32).reshape(-1, 1)
        is_breakdown = ((rms_smooth_norm < 0.30) & (var_smooth_norm < 0.35)).astype(np.float32).reshape(-1, 1)

        # ── Concatenação final ────────────────────────────────────────
        final_features = np.concatenate([
            mels_db,        # 80
            grid_features,  #  2
            song_position,  #  1
            rms_norm,       #  1
            onset_norm,     #  1
            var_short,      #  1
            var_medium,     #  1
            var_long,       #  1
            centroid_norm,  #  1
            flatness_norm,  #  1
            zcr_norm,       #  1
            is_drop,        #  1
            is_breakdown,   #  1
        ], axis=1)  # Total: 93

        assert final_features.shape[1] == TOTAL_FEATURES, (
            f"Feature count mismatch: got {final_features.shape[1]}, expected {TOTAL_FEATURES}"
        )

        return final_features, sr, hop_length

    except Exception as e:
        print(f"Erro ao extrair features de {file_path}: {e}")
        return None, None, None


def analyze_energy(file_path, hop_length=512, sr=22050):
    """
    Analisa a energia da música ao longo do tempo.
    Retorna um array normalizado (0-1) com a mesma resolução temporal das features.
    """
    y, _ = librosa.load(file_path, sr=sr)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    min_len = min(len(rms), len(onset_env))
    combined = rms[:min_len] * 0.6 + onset_env[:min_len] * 0.4

    norm_energy = _safe_normalize(combined)

    window_size = max(1, int(sr / hop_length))  # ~1 segundo
    smoothed_energy = np.convolve(norm_energy, np.ones(window_size) / window_size, mode='same')

    return smoothed_energy


def add_silence(file_path, output_path, silence_duration_ms=3000):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    silence_sec = silence_duration_ms / 1000.0
    filter_str = f"adelay={silence_duration_ms}|{silence_duration_ms},apad=pad_dur={silence_sec}"

    cmd = [ffmpeg_exe, '-y', '-i', file_path, '-af', filter_str]
    if output_path.endswith('.egg'):
        cmd.extend(['-f', 'ogg'])
    cmd.append(output_path)

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        try:
            cmd[4] = f"adelay={silence_duration_ms},apad=pad_dur={silence_sec}"
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            raise e


if __name__ == "__main__":
    pass