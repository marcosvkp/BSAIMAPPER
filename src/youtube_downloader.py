"""
youtube_downloader.py — Download de áudio do YouTube

Baixa o áudio de um vídeo do YouTube, converte para MP3 (análise)
e OGG/egg (Beat Saber), e salva a thumbnail como cover.png.

Usa imageio-ffmpeg para localizar o ffmpeg automaticamente
(funciona em Windows, Linux e macOS sem configuração manual).
"""

import os
import requests
import imageio_ffmpeg
import subprocess
from pytubefix import YouTube
from PIL import Image
from io import BytesIO


def download_from_youtube(url: str, output_folder: str = "data/input_music"):
    """
    Baixa áudio + thumbnail de um vídeo do YouTube.

    Retorna:
        (mp3_path, cover_path) — caminhos dos arquivos salvos,
        ou (None, None) em caso de erro.
    """
    os.makedirs(output_folder, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        print(f"  Acessando: {url}")
        yt = YouTube(url)
        print(f"  Título: {yt.title}")

        # ── Thumbnail ─────────────────────────────────────────────
        cover_path = os.path.join(output_folder, "cover.png")
        try:
            resp = requests.get(yt.thumbnail_url, timeout=10)
            img  = Image.open(BytesIO(resp.content)).resize((256, 256))
            img.save(cover_path)
        except Exception as e:
            print(f"  ⚠  Thumbnail indisponível: {e}")
            cover_path = None

        # ── Áudio ─────────────────────────────────────────────────
        stream = (
            yt.streams.filter(only_audio=True, file_extension='mp4').first()
            or yt.streams.filter(only_audio=True).first()
        )
        if not stream:
            print("  ❌ Nenhum stream de áudio encontrado.")
            return None, None

        print("  Baixando stream de áudio...")
        raw_file = stream.download(output_path=output_folder)
        raw_file = os.path.abspath(raw_file)

        base      = os.path.splitext(raw_file)[0]
        mp3_path  = base + ".mp3"
        egg_path  = base + ".egg"

        # Converte para MP3 (análise) e OGG/egg (Beat Saber) via ffmpeg
        def _ffmpeg(in_file, out_file, extra_args=None):
            cmd = [ffmpeg, '-y', '-i', in_file]
            if extra_args:
                cmd.extend(extra_args)
            cmd.append(out_file)
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("  Convertendo para MP3...")
        _ffmpeg(raw_file, mp3_path,
                ['-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k'])

        print("  Convertendo para OGG (.egg)...")
        _ffmpeg(raw_file, egg_path, ['-vn', '-f', 'ogg'])

        os.remove(raw_file)
        print(f"  ✅ MP3: {mp3_path}")
        return mp3_path, cover_path

    except Exception as e:
        print(f"  ❌ Erro no download: {e}")
        return None, None


if __name__ == "__main__":
    url = input("URL do YouTube: ").strip()
    if url:
        mp3, cover = download_from_youtube(url)
        if mp3:
            print(f"Concluído! MP3: {mp3} | Cover: {cover}")
