import os
import requests
from pytubefix import YouTube
import imageio_ffmpeg
from pydub import AudioSegment
from PIL import Image
from io import BytesIO

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

def download_from_youtube(url, output_folder="data/input_music"):
    """
    Baixa o áudio de um vídeo do YouTube usando pytubefix,
    converte para MP3 e também cria uma versão OGG (.egg) usando pydub.
    Também baixa a thumbnail e salva como cover.png.
    
    Retorna uma tupla: (caminho_mp3, caminho_cover)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Configurar pydub para usar o ffmpeg do imageio-ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_path
    
    try:
        print(f"Acessando vídeo: {url}")
        yt = YouTube(url)
        print(f"Título: {yt.title}")
        
        # --- Download Thumbnail ---
        cover_path = os.path.join(output_folder, "cover.png")
        try:
            print(f"Baixando thumbnail: {yt.thumbnail_url}")
            response = requests.get(yt.thumbnail_url)
            img = Image.open(BytesIO(response.content))
            # Resize to 256x256 as per Beat Saber standard (optional but good)
            img = img.resize((256, 256))
            img.save(cover_path)
            print(f"Cover salvo: {cover_path}")
        except Exception as e:
            print(f"Erro ao baixar thumbnail: {e}")
            cover_path = None

        # --- Download Audio ---
        # Baixar apenas o stream de áudio (geralmente mp4/m4a ou webm)
        stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        
        if not stream:
             stream = yt.streams.filter(only_audio=True).first()

        if not stream:
            print("Nenhum stream de áudio encontrado.")
            return None, None

        print("Baixando stream de áudio...")
        downloaded_file = stream.download(output_path=output_folder)
        downloaded_file = os.path.abspath(downloaded_file)
        print(f"Arquivo original baixado: {downloaded_file}")

        # Conversão
        base_name = os.path.splitext(downloaded_file)[0]
        mp3_filename = base_name + ".mp3"
        ogg_filename = base_name + ".egg"

        print("Convertendo áudio...")
        
        try:
            audio = AudioSegment.from_file(downloaded_file)
            
            print(f"Salvando MP3: {mp3_filename}")
            audio.export(mp3_filename, format="mp3")
            
            print(f"Salvando OGG (.egg): {ogg_filename}")
            audio.export(ogg_filename, format="ogg")
            
            os.remove(downloaded_file)
            print("Arquivo original removido.")
            
            return mp3_filename, cover_path
            
        except Exception as e_pydub:
            print(f"Erro no pydub (provavelmente falta ffprobe): {e_pydub}")
            print("Tentando conversão direta via ffmpeg...")
            
            import subprocess
            
            # Converter para MP3 via subprocesso
            cmd_mp3 = [
                ffmpeg_path, '-y', '-i', downloaded_file, 
                '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', mp3_filename
            ]
            subprocess.run(cmd_mp3, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"MP3 salvo via ffmpeg: {mp3_filename}")

            # Converter para OGG via subprocesso
            cmd_ogg = [
                ffmpeg_path, '-y', '-i', downloaded_file,
                '-vn', '-acodec', 'libvorbis', ogg_filename
            ]
            subprocess.run(cmd_ogg, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"OGG salvo via ffmpeg: {ogg_filename}")
            
            os.remove(downloaded_file)
            return mp3_filename, cover_path

    except Exception as e:
        print(f"Erro geral: {e}")
        return None, None

if __name__ == "__main__":
    # Exemplo de uso
    url = input("Insira a URL do YouTube: ")
    if url:
        mp3, cover = download_from_youtube(url)
        if mp3:
            print(f"Processo concluído! MP3: {mp3}, Cover: {cover}")
