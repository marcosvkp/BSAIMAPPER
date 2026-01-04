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

def extract_features(file_path, bpm, sr=22050, n_mels=80, hop_length=512):
    """
    Extrai Mel Spectrogram E features de ritmo (Grid).
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        
        # 1. Mel Spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mels_db = librosa.power_to_db(mels, ref=np.max).T # (Time, n_mels)
        
        num_frames = mels_db.shape[0]
        
        # 2. Grid Features (Onde caem os beats?)
        # Criar canais extras que "pulsam" nos beats
        seconds_per_beat = 60.0 / bpm
        frame_duration = hop_length / sr
        frames_per_beat = seconds_per_beat / frame_duration
        
        grid_features = np.zeros((num_frames, 2), dtype=np.float32)
        
        # Canal 0: Pulsos nos beats inteiros (1/1)
        # Canal 1: Pulsos nos meios beats (1/2)
        
        for i in range(num_frames):
            beat_pos = i / frames_per_beat
            # Distância para o beat inteiro mais próximo
            dist_beat = abs(beat_pos - round(beat_pos))
            if dist_beat < 0.1: # Tolerância
                grid_features[i, 0] = 1.0 - (dist_beat * 10) # Decaimento suave
                
            # Distância para o meio beat
            dist_half = abs((beat_pos * 2) - round(beat_pos * 2))
            if dist_half < 0.1:
                grid_features[i, 1] = 1.0 - (dist_half * 10)

        # Concatenar: Agora temos 80 (audio) + 2 (ritmo) = 82 features
        final_features = np.concatenate([mels_db, grid_features], axis=1)
        
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
