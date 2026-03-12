import librosa
import numpy as np
import subprocess
import imageio_ffmpeg

# ─────────────────────────────────────────────────────────────────
# Features extraídas por `extract_features`:
#
#   1  onset_strength  — força dos onsets (batidas)     → timing principal
#   2  onset_peaks     — picos binários suavizados      → onde estão os beats
#   3  rms             — energia local (volume)         → intensidade da seção
#   4  beat_phase      — fase no beat (0→1 cíclico)    → quantização rítmica
#   5  halfbeat_phase  — fase no meio-beat              → subdivisões
#   6  song_position   — posição na música (0→1)        → estrutura global
#   7  is_drop         — seção de drop (binário suave)  → alta energia
#   8  is_breakdown    — seção calma (binário suave)    → baixa energia
#  ──────
#   8  TOTAL
#
# Filosofia: menos é mais.
# O Mel Spectrogram (80 features) é excelente para identificar timbres
# e instrumentos, mas para o problema de TIMING ("tem nota aqui?")
# o onset_strength + rms + fase rítmica capturam >90% da informação
# relevante em 8 features em vez de 93.
# Isso libera capacidade da GRU para aprender padrões, não compressão.
# ─────────────────────────────────────────────────────────────────

TOTAL_FEATURES = 8
SR         = 22050
HOP_LENGTH = 512
FRAMES_PER_SEC = SR / HOP_LENGTH  # ~43.07


def detect_bpm(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        return float(tempo[0])
    return float(tempo)


def _safe_normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def extract_features(file_path, bpm, sr=SR, hop_length=HOP_LENGTH):
    """
    Extrai 8 features temporais focadas em timing para o TimingNet.

    Retorna:
        features   : ndarray (num_frames, 8)
        sr         : int
        hop_length : int
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        frame_duration = hop_length / sr

        # ── 1. Onset Strength ─────────────────────────────────────
        # Principal sinal de "tem algo acontecendo aqui".
        # Picos de onset_strength correspondem a ataques de instrumentos,
        # batidas e transientes — exatamente onde mappers colocam notas.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        num_frames = len(onset_env)
        onset_norm = _safe_normalize(onset_env).reshape(-1, 1)

        # ── 2. Onset Peaks (binário suavizado) ────────────────────
        # Detecta picos locais do onset_strength e cria um sinal binário
        # suavizado — "aqui tem um onset forte".
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length, units='frames'
        )
        onset_peaks = np.zeros(num_frames, dtype=np.float32)
        onset_peaks[onset_frames[onset_frames < num_frames]] = 1.0
        # Suaviza com janela de ~50ms para dar contexto ao redor do onset
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

        # ── 4. Beat Phase (fase cíclica no beat) ──────────────────
        # Em vez de pulsos binários (grid_features anterior), usamos a
        # fase contínua: 0.0 = no beat, 0.5 = no meio beat, 1.0 = próximo beat.
        # Isso dá ao modelo informação contínua sobre onde está no ciclo rítmico.
        seconds_per_beat = 60.0 / bpm
        frames_per_beat  = seconds_per_beat / frame_duration
        beat_phase = np.array([
            (i / frames_per_beat) % 1.0
            for i in range(num_frames)
        ], dtype=np.float32).reshape(-1, 1)

        # ── 5. Half-Beat Phase ────────────────────────────────────
        halfbeat_phase = np.array([
            ((i / frames_per_beat) * 2) % 1.0
            for i in range(num_frames)
        ], dtype=np.float32).reshape(-1, 1)

        # ── 6. Posição na música ───────────────────────────────────
        song_position = np.linspace(0.0, 1.0, num_frames, dtype=np.float32).reshape(-1, 1)

        # ── 7 & 8. Drop / Breakdown detection ─────────────────────
        smooth_s = max(1, int(1.0 / frame_duration))  # ~1 segundo
        rms_smooth = np.convolve(rms_raw, np.ones(smooth_s) / smooth_s, mode='same')
        rms_s_norm = _safe_normalize(rms_smooth)

        # Variância local de energia (~2s) para detectar dinâmica
        w_var = max(1, int(2.0 / frame_duration))
        half_v = w_var // 2
        rms_pad = np.pad(rms_raw, half_v, mode='edge')
        var_local = np.array([
            np.var(rms_pad[i:i + w_var]) for i in range(num_frames)
        ], dtype=np.float32)
        var_norm = _safe_normalize(var_local)

        is_drop      = ((rms_s_norm > 0.65) & (var_norm > 0.50)).astype(np.float32).reshape(-1, 1)
        is_breakdown = ((rms_s_norm < 0.30) & (var_norm < 0.35)).astype(np.float32).reshape(-1, 1)

        # ── Concatenação ──────────────────────────────────────────
        features = np.concatenate([
            onset_norm,          # 1
            onset_peaks_smooth,  # 2
            rms_norm,            # 3
            beat_phase,          # 4
            halfbeat_phase,      # 5
            song_position,       # 6
            is_drop,             # 7
            is_breakdown,        # 8
        ], axis=1)

        assert features.shape[1] == TOTAL_FEATURES, (
            f"Feature mismatch: {features.shape[1]} != {TOTAL_FEATURES}"
        )

        return features, sr, hop_length

    except Exception as e:
        print(f"Erro ao extrair features de {file_path}: {e}")
        return None, None, None


def analyze_energy(file_path, hop_length=HOP_LENGTH, sr=SR):
    """
    Retorna perfil de energia normalizado (0-1) para uso na geração.
    Usado pelo generate_from_url para ponderar candidatos a nota.
    """
    y, _ = librosa.load(file_path, sr=sr)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    min_len = min(len(rms), len(onset_env))
    combined = rms[:min_len] * 0.6 + onset_env[:min_len] * 0.4
    norm = _safe_normalize(combined)
    window = max(1, int(sr / hop_length))
    return np.convolve(norm, np.ones(window) / window, mode='same')


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