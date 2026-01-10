import librosa
import numpy as np
import os
import subprocess
import imageio_ffmpeg

def detect_bpm(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        return float(tempo[0])
    return float(tempo)

def analyze_energy(file_path, hop_length=512, sr=22050):
    """
    Retorna um perfil de energia normalizado (0-1) ao longo do tempo.
    Útil para detectar seções calmas vs intensas.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Suavização (Moving Average)
        window_size = int(sr * 2 / hop_length) # Janela de ~2 segundos
        if window_size > 0:
            rms = np.convolve(rms, np.ones(window_size)/window_size, mode='same')

        # Normalização Robusta (Percentil 95 para ignorar picos extremos)
        p95 = np.percentile(rms, 95)
        if p95 > 0:
            rms = rms / p95

        return np.clip(rms, 0, 1)
    except Exception as e:
        print(f"Erro ao analisar energia: {e}")
        return np.zeros(100)

def extract_features(file_path, bpm, sr=22050, n_mels=80, hop_length=512):
    """
    Extrai Mel Spectrogram, features de ritmo (Grid), e features de energia (RMS e Onset).
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        
        # 1. Mel Spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mels_db = librosa.power_to_db(mels, ref=np.max).T # (Time, n_mels)
        
        num_frames = mels_db.shape[0]
        
        # 2. Grid Features (Onde caem os beats?)
        seconds_per_beat = 60.0 / bpm
        frame_duration = hop_length / sr
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

        # 3. Energy Features (Volume e Mudanças)
        # RMS (Volume/Energia bruta)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Spectral Flux (Onset Strength) - Mudanças súbitas (batidas)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Normalizar e ajustar tamanho
        rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
        onset_env = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-6)

        # Garantir que tenham o mesmo tamanho que o melspectrogram
        if len(rms) < num_frames:
            rms = np.pad(rms, (0, num_frames - len(rms)))
        else:
            rms = rms[:num_frames]

        if len(onset_env) < num_frames:
            onset_env = np.pad(onset_env, (0, num_frames - len(onset_env)))
        else:
            onset_env = onset_env[:num_frames]

        # Reshape para concatenar
        rms = rms.reshape(-1, 1)
        onset_env = onset_env.reshape(-1, 1)

        # Concatenar tudo: 80 (audio) + 2 (ritmo) + 1 (volume) + 1 (onset) = 84 features
        final_features = np.concatenate([mels_db, grid_features, rms, onset_env], axis=1)
        
        return final_features, sr, hop_length
    except Exception as e:
        print(f"Erro ao extrair features de {file_path}: {e}")
        return None, None, None

def add_silence(file_path, output_path, silence_duration_ms=3000):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    silence_sec = silence_duration_ms / 1000.0
    filter_str = f"adelay={silence_duration_ms}|{silence_duration_ms},apad=pad_dur={silence_sec}"
    
    cmd = [ffmpeg_exe, '-y', '-i', file_path, '-af', filter_str]
    if output_path.endswith('.egg'): cmd.extend(['-f', 'ogg'])
    cmd.append(output_path)
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Fallback mono
        try:
            cmd[4] = f"adelay={silence_duration_ms},apad=pad_dur={silence_sec}"
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            raise e

if __name__ == "__main__":
    pass
