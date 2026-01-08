import librosa
import numpy as np
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
        
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mels_db = librosa.power_to_db(mels, ref=np.max).T # (Time, n_mels)
        
        num_frames = mels_db.shape[0]
        
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

        final_features = np.concatenate([mels_db, grid_features], axis=1)
        
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
    
    combined = rms * 0.6 + onset_env * 0.4
    
    norm_energy = (combined - np.min(combined)) / (np.max(combined) - np.min(combined) + 1e-6)
    
    window_size = int(sr / hop_length)
    smoothed_energy = np.convolve(norm_energy, np.ones(window_size)/window_size, mode='same')
    
    return smoothed_energy

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
        try:
            cmd[4] = f"adelay={silence_duration_ms},apad=pad_dur={silence_sec}"
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            raise e

if __name__ == "__main__":
    pass
