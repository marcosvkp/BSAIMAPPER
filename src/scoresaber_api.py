import requests
import time
import json

def get_scoresaber_info(map_hash: str, retries: int = 3, delay: float = 5.0) -> dict | None:
    """
    Busca as informações de dificuldade (estrelas) da API do ScoreSaber para um determinado hash de mapa.
    Retorna um dicionário com as estrelas por dificuldade (ex: {"ExpertPlus": {"stars": 8.87}}).
    """
    # Endpoint para obter todas as dificuldades de um hash
    difficulties_url = f"https://scoresaber.com/api/leaderboard/get-difficulties/{map_hash}"
    
    for attempt in range(retries):
        try:
            response = requests.get(difficulties_url, timeout=15)
            response.raise_for_status() # Lança exceção para status de erro (4xx ou 5xx)
            difficulties_data = response.json()
            
            # Se não houver dados, o mapa não é ranqueado no ScoreSaber
            if not difficulties_data:
                return None

            map_difficulties = {}
            # Para cada dificuldade encontrada, busca as informações detalhadas (que contêm as estrelas)
            for diff_info in difficulties_data:
                leaderboard_id = diff_info.get('leaderboardId')
                if not leaderboard_id:
                    continue

                leaderboard_url = f"https://scoresaber.com/api/leaderboard/by-id/{leaderboard_id}/info"
                lb_response = requests.get(leaderboard_url, timeout=15)
                lb_response.raise_for_status()
                leaderboard_details = lb_response.json()

                # Extrai as informações relevantes
                # difficultyRaw example: "_ExpertPlus_SoloStandard"
                raw_diff_name = diff_info.get('difficultyRaw')
                if raw_diff_name:
                    parts = raw_diff_name.split('_')
                    if len(parts) > 1:
                        difficulty_name = parts[1] # "ExpertPlus"
                    else:
                        difficulty_name = raw_diff_name # Fallback se não houver underscore
                else:
                    difficulty_name = None # Não foi possível determinar o nome da dificuldade

                stars = leaderboard_details.get('stars')
                
                if difficulty_name and stars is not None:
                    map_difficulties[difficulty_name] = {'stars': stars}

            return map_difficulties if map_difficulties else None

        except requests.exceptions.RequestException as e:
            print(f"  [ScoreSaber] Tentativa {attempt + 1} falhou para o hash {map_hash}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None # Retorna None após todas as tentativas falharem

if __name__ == "__main__":
    # Exemplo de uso: substitua por um hash real de um mapa ranqueado
    test_hash = "0000000000000000000000000000000000000000" # Exemplo de hash inválido
    # Um hash real seria algo como: "6140306140306140306140306140306140306140"
    print(f"Testando get_scoresaber_info para hash: {test_hash}")
    stars_data = get_scoresaber_info(test_hash)
    if stars_data:
        print("Dados de estrelas obtidos:")
        print(json.dumps(stars_data, indent=4))
    else:
        print("Não foi possível obter dados de estrelas para o hash fornecido.")
