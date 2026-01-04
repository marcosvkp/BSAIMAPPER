import os
from pytubefix import YouTube
import imageio_ffmpeg
from pydub import AudioSegment

from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

def download_from_youtube(url, output_folder="data/input_music"):
    """
    Baixa o áudio de um vídeo do YouTube usando pytubefix,
    converte para MP3 e também cria uma versão OGG (.egg) usando pydub.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Configurar pydub para usar o ffmpeg do imageio-ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_path
    
    # O pydub precisa do ffprobe também para ler alguns formatos.
    # Como o imageio-ffmpeg não traz o ffprobe, vamos tentar forçar a leitura
    # ou assumir que o ffmpeg consegue converter se passarmos os argumentos certos.
    # Mas o erro anterior foi justamente falta de ffprobe.
    
    # Alternativa: Usar subprocesso direto com o ffmpeg do imageio se o pydub falhar,
    # mas vamos tentar primeiro baixar um formato mais amigável (m4a/mp4) que o pydub
    # talvez consiga ler sem ffprobe, ou tratar o erro.

    try:
        print(f"Acessando vídeo: {url}")
        yt = YouTube(url)
        print(f"Título: {yt.title}")
        
        # Baixar apenas o stream de áudio (geralmente mp4/m4a ou webm)
        # Vamos preferir m4a/mp4 que é mais comum
        stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        
        if not stream:
             stream = yt.streams.filter(only_audio=True).first()

        if not stream:
            print("Nenhum stream de áudio encontrado.")
            return None

        print("Baixando stream de áudio...")
        downloaded_file = stream.download(output_path=output_folder)
        downloaded_file = os.path.abspath(downloaded_file)
        print(f"Arquivo original baixado: {downloaded_file}")

        # Conversão
        base_name = os.path.splitext(downloaded_file)[0]
        mp3_filename = base_name + ".mp3"
        ogg_filename = base_name + ".egg"

        print("Convertendo áudio...")
        
        # Tentar carregar com pydub. Se falhar por falta de ffprobe,
        # teremos que usar o ffmpeg via linha de comando (subprocess)
        try:
            audio = AudioSegment.from_file(downloaded_file)
            
            print(f"Salvando MP3: {mp3_filename}")
            audio.export(mp3_filename, format="mp3")
            
            print(f"Salvando OGG (.egg): {ogg_filename}")
            audio.export(ogg_filename, format="ogg")
            
            # Remover original
            os.remove(downloaded_file)
            print("Arquivo original removido.")
            
            return mp3_filename
            
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
            return mp3_filename

    except Exception as e:
        print(f"Erro geral: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    url = input("Insira a URL do YouTube: ")
    if url:
        path = download_from_youtube(url)
        if path:
            print(f"Processo concluído! Arquivo final: {path}")
