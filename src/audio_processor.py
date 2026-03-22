"""
audio_processor.py — Extração de Features de Áudio V5

Dois grupos de features são extraídas:

  mel_spec  (MEL_BINS=64 valores)
    Espectrograma Mel normalizado por log.
    Captura timbre, instrumentação e textura harmônica.
    O PatternModel usa isso para aprender "qual tipo de energia musical aqui".

  ctx_feats (8 valores)
    Features de contexto temporal já usadas na V4:
      1. onset_strength  — força dos ataques (principal sinal de "tem nota")
      2. onset_peaks     — picos binários suavizados
      3. rms             — energia local (volume)
      4. beat_phase      — fase cíclica no beat (0→1)
      5. halfbeat_phase  — fase no meio-beat
      6. song_position   — posição global na música (0→1)
      7. is_drop         — seção de alta energia (binário suave)
      8. is_breakdown    — seção calma (binário suave)

Total: 72 features por frame.

Por que 64 bins mel em vez de 128?
  - 128 bins adicionam ~50% de tempo de extração e ~30% de parâmetros no conv
  - Para o problema de POSICIONAMENTO (col/layer) 64 bins capturam frequências
    suficientes (kick, snare, melodia principal) sem custo computacional alto
  - O onset_strength já captura a informação transiente de alta resolução
"""

import librosa
import numpy as np
import subprocess
import imageio_ffmpeg

# ─────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────

MEL_BINS       = 64
CTX_FEATS      = 8
TOTAL_FEATURES = MEL_BINS + CTX_FEATS  # 72

SR         = 22050
HOP_LENGTH = 512
FRAMES_PER_SEC = SR / HOP_LENGTH  # ~43.07 fps


def detect_bpm(file_path: str) -> float:
    """Detecta o BPM de um arquivo de áudio."""
    y, sr = librosa.load(file_path, sr=SR)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        return float(tempo[0])
    return float(tempo)


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Normaliza para [0,1]. Retorna zeros se o range for muito pequeno."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def extract_features(file_path: str, bpm: float,
                     sr: int = SR, hop_length: int = HOP_LENGTH):
    """
    Extrai as 72 features de áudio para um arquivo.

    Retorna:
        mel_spec   : ndarray (num_frames, MEL_BINS)    — espectrograma mel log
        ctx_feats  : ndarray (num_frames, CTX_FEATS)   — features de contexto
        sr         : int
        hop_length : int

    Retorna (None, None, None, None) em caso de erro.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        frame_dur = hop_length / sr

        # ── Mel Spectrogram ───────────────────────────────────────
        # log(1 + mel) — range estável, invariante à escala de amplitude
        mel_raw = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=hop_length,
            n_mels=MEL_BINS, fmax=8000,
        )
        mel_log = librosa.power_to_db(mel_raw, ref=np.max)
        # Normaliza para [0, 1] por bin para invariância de volume
        mel_norm = np.zeros_like(mel_log)
        for i in range(MEL_BINS):
            mel_norm[i] = _safe_normalize(mel_log[i])
        mel_spec = mel_norm.T  # (num_frames, MEL_BINS)
        num_frames = mel_spec.shape[0]

        # ── 1. Onset Strength ─────────────────────────────────────
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_env = onset_env[:num_frames]
        if len(onset_env) < num_frames:
            onset_env = np.pad(onset_env, (0, num_frames - len(onset_env)))
        onset_norm = _safe_normalize(onset_env).reshape(-1, 1)

        # ── 2. Onset Peaks (binário suavizado) ────────────────────
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length, units='frames'
        )
        onset_peaks = np.zeros(num_frames, dtype=np.float32)
        valid = onset_frames[onset_frames < num_frames]
        onset_peaks[valid] = 1.0
        smooth_w = max(1, int(0.05 * FRAMES_PER_SEC))
        onset_peaks_smooth = np.convolve(
            onset_peaks, np.ones(smooth_w) / smooth_w, mode='same'
        ).reshape(-1, 1)

        # ── 3. RMS energia local ───────────────────────────────────
        rms_raw = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_raw = rms_raw[:num_frames]
        if len(rms_raw) < num_frames:
            rms_raw = np.pad(rms_raw, (0, num_frames - len(rms_raw)))
        rms_norm = _safe_normalize(rms_raw).reshape(-1, 1)

        # ── 4. Beat Phase ──────────────────────────────────────────
        seconds_per_beat = 60.0 / bpm
        frames_per_beat  = seconds_per_beat / frame_dur
        beat_phase = np.array(
            [(i / frames_per_beat) % 1.0 for i in range(num_frames)],
            dtype=np.float32,
        ).reshape(-1, 1)

        # ── 5. Half-Beat Phase ────────────────────────────────────
        halfbeat_phase = np.array(
            [((i / frames_per_beat) * 2) % 1.0 for i in range(num_frames)],
            dtype=np.float32,
        ).reshape(-1, 1)

        # ── 6. Posição na música ───────────────────────────────────
        song_position = np.linspace(0.0, 1.0, num_frames, dtype=np.float32).reshape(-1, 1)

        # ── 7 & 8. Drop / Breakdown ───────────────────────────────
        smooth_s  = max(1, int(1.0 / frame_dur))
        rms_smooth = np.convolve(rms_raw, np.ones(smooth_s) / smooth_s, mode='same')
        rms_s_norm = _safe_normalize(rms_smooth)

        w_var  = max(1, int(2.0 / frame_dur))
        half_v = w_var // 2
        rms_pad   = np.pad(rms_raw, half_v, mode='edge')
        var_local = np.array(
            [np.var(rms_pad[i:i + w_var]) for i in range(num_frames)],
            dtype=np.float32,
        )
        var_norm = _safe_normalize(var_local)

        is_drop      = ((rms_s_norm > 0.65) & (var_norm > 0.50)).astype(np.float32).reshape(-1, 1)
        is_breakdown = ((rms_s_norm < 0.30) & (var_norm < 0.35)).astype(np.float32).reshape(-1, 1)

        # ── Concatenação das features de contexto ─────────────────
        ctx_feats = np.concatenate([
            onset_norm,
            onset_peaks_smooth,
            rms_norm,
            beat_phase,
            halfbeat_phase,
            song_position,
            is_drop,
            is_breakdown,
        ], axis=1).astype(np.float32)   # (num_frames, 8)

        assert mel_spec.shape   == (num_frames, MEL_BINS), \
            f"mel_spec shape wrong: {mel_spec.shape}"
        assert ctx_feats.shape  == (num_frames, CTX_FEATS), \
            f"ctx_feats shape wrong: {ctx_feats.shape}"

        return mel_spec, ctx_feats, sr, hop_length

    except Exception as e:
        print(f"  [audio_processor] Erro ao extrair features de {file_path}: {e}")
        return None, None, None, None


def analyze_energy(file_path: str,
                   hop_length: int = HOP_LENGTH,
                   sr: int = SR) -> np.ndarray:
    """
    Retorna perfil de energia normalizado (0-1).
    Combina RMS e onset strength para ponderar candidatos a nota na geração.
    """
    y, _ = librosa.load(file_path, sr=sr)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    min_len  = min(len(rms), len(onset_env))
    combined = rms[:min_len] * 0.6 + onset_env[:min_len] * 0.4
    norm     = _safe_normalize(combined)
    window   = max(1, int(sr / hop_length))
    return np.convolve(norm, np.ones(window) / window, mode='same')


def add_silence(file_path: str, output_path: str, silence_ms: int = 3000):
    """
    Prepend e append silêncio ao áudio (padrão Beat Saber).
    Salva em formato OGG se output_path terminar com .egg.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    filter_str = (
        f"adelay={silence_ms}|{silence_ms},"
        f"apad=pad_dur={silence_ms / 1000.0}"
    )
    cmd = [ffmpeg_exe, '-y', '-i', file_path, '-af', filter_str]
    if output_path.endswith('.egg'):
        cmd.extend(['-f', 'ogg'])
    cmd.append(output_path)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        # Fallback: mono delay syntax
        cmd[4] = f"adelay={silence_ms},apad=pad_dur={silence_ms / 1000.0}"
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
