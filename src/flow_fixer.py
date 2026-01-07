import math

class FlowFixer:
    """
    Simulador de Paridade e Corretor de Flow (V2 - Refinado).
    Foco: Ergonomia central e uso inteligente de bombas.
    """

    # Constantes de Direção
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    DOT = 8

    # Constantes de Tipo
    LEFT_HAND = 0
    RIGHT_HAND = 1
    BOMB = 3

    @staticmethod
    def fix(notes, bpm):
        if not notes: return []

        notes.sort(key=lambda x: x['_time'])
        
        left_notes = [n for n in notes if n['_type'] == FlowFixer.LEFT_HAND]
        right_notes = [n for n in notes if n['_type'] == FlowFixer.RIGHT_HAND]
        
        fixed_left = FlowFixer._process_hand(left_notes, bpm, FlowFixer.LEFT_HAND)
        fixed_right = FlowFixer._process_hand(right_notes, bpm, FlowFixer.RIGHT_HAND)
        
        all_objects = fixed_left + fixed_right
        all_objects.sort(key=lambda x: x['_time'])
        
        return all_objects

    @staticmethod
    def _process_hand(hand_notes, bpm, hand_type):
        if not hand_notes: return []

        processed_objects = []
        seconds_per_beat = 60 / bpm
        
        # --- CONFIGURAÇÃO DE TIMING ---
        # Se o tempo for menor que isso, o movimento TEM que ser contínuo (inverte nota).
        # Se for maior, permitimos reset com bomba.
        # Aumentado para 0.8s (quase 1 segundo) para evitar bombas em sequências rápidas.
        BOMB_THRESHOLD = 0.8 

        prev_ended_up = False 
        prev_note = hand_notes[0]
        processed_objects.append(prev_note)

        # Estado inicial
        if prev_note['_cutDirection'] in [FlowFixer.UP, FlowFixer.UP_LEFT, FlowFixer.UP_RIGHT]:
            prev_ended_up = True
        
        for i in range(1, len(hand_notes)):
            curr_note = hand_notes[i]
            time_diff = (curr_note['_time'] - prev_note['_time']) * seconds_per_beat
            
            # --- 1. CORREÇÃO DE ERGONOMIA (O problema do meio) ---
            # Verifica se a transição é fisicamente estranha ANTES de checar paridade
            if FlowFixer._is_awkward_transition(prev_note, curr_note, prev_ended_up, hand_type):
                # Se for estranho (ex: Cima -> Lado no meio), força uma diagonal ou Dot
                curr_note['_cutDirection'] = FlowFixer._fix_awkward_angle(curr_note, prev_ended_up, hand_type)

            # --- 2. ANÁLISE DE PARIDADE ---
            curr_cut = curr_note['_cutDirection']
            is_dot = curr_cut == FlowFixer.DOT
            
            # Determina intenção atual
            curr_goes_up = False
            if curr_cut in [FlowFixer.UP, FlowFixer.UP_LEFT, FlowFixer.UP_RIGHT]:
                curr_goes_up = True
            elif curr_cut in [FlowFixer.DOWN, FlowFixer.DOWN_LEFT, FlowFixer.DOWN_RIGHT]:
                curr_goes_up = False
            elif is_dot:
                curr_goes_up = curr_note['_lineLayer'] >= 1
            else:
                # Laterais: Se cortou lateral, o braço geralmente fica na altura que estava
                curr_goes_up = prev_ended_up 

            # Detecta Reset (Tentativa de ir para onde já está)
            is_reset = (prev_ended_up and curr_goes_up) or (not prev_ended_up and not curr_goes_up)

            if is_reset:
                if time_diff < BOMB_THRESHOLD:
                    # CASO 1: Reset Rápido -> PROIBIDO BOMBA -> INVERTER NOTA
                    new_cut = FlowFixer._get_inverse_cut(curr_cut)
                    
                    # Verifica se a inversão cria um ângulo ruim
                    if FlowFixer._is_bad_angle(new_cut, curr_note['_lineIndex'], hand_type):
                        curr_note['_cutDirection'] = FlowFixer.DOT
                    else:
                        curr_note['_cutDirection'] = new_cut
                    
                    # Atualiza estado lógico (agora flui)
                    curr_goes_up = not prev_ended_up 

                else:
                    # CASO 2: Reset Longo -> BOMBA DE RESET
                    # Adiciona bomba APENAS na Layer 0 (Baixo) como solicitado
                    bomb_time = (prev_note['_time'] + curr_note['_time']) / 2
                    
                    # Posição da bomba: Tenta colocar na mesma coluna da próxima nota para bloquear
                    bomb = {
                        "_time": bomb_time,
                        "_lineIndex": curr_note['_lineIndex'],
                        "_lineLayer": 0, # SEMPRE EM BAIXO
                        "_type": FlowFixer.BOMB,
                        "_cutDirection": 0
                    }
                    processed_objects.append(bomb)
                    # Reset aceito, vida que segue

            processed_objects.append(curr_note)
            prev_note = curr_note
            prev_ended_up = curr_goes_up

        return processed_objects

    @staticmethod
    def _is_awkward_transition(prev, curr, prev_up, hand):
        """
        Detecta movimentos difíceis, especialmente nas colunas do meio.
        Ex: Braço em cima (UP) -> Corte Lateral (LEFT) na coluna do meio.
        Isso exige descer o braço e girar o pulso estranhamente.
        """
        curr_cut = curr['_cutDirection']
        line = curr['_lineIndex']
        
        # Se é Dot, geralmente é seguro
        if curr_cut == FlowFixer.DOT: return False

        # Cenário: Braço está em CIMA
        if prev_up:
            # E tenta cortar puramente para o LADO (Left/Right)
            if curr_cut in [FlowFixer.LEFT, FlowFixer.RIGHT]:
                # Nas colunas do meio (1 e 2), isso é muito desconfortável vindo de cima
                if line in [1, 2]:
                    return True
        
        # Cenário: Braço está em BAIXO
        if not prev_up:
            # E tenta cortar puramente para o LADO
            if curr_cut in [FlowFixer.LEFT, FlowFixer.RIGHT]:
                # Também pode ser estranho dependendo da rotação anterior, mas menos crítico.
                # Vamos focar no problema "Cima -> Lado" que é o pior.
                pass

        return False

    @staticmethod
    def _fix_awkward_angle(note, prev_up, hand):
        """
        Substitui um ângulo ruim por uma diagonal fluida ou Dot.
        """
        cut = note['_cutDirection']
        
        # Se estava em cima e tentou cortar lado, melhor cortar DIAGONAL BAIXO
        if prev_up:
            if hand == FlowFixer.LEFT_HAND:
                # Mão Esq em cima -> Corte Diagonal Baixo-Esq é natural
                return FlowFixer.DOWN_LEFT
            else:
                # Mão Dir em cima -> Corte Diagonal Baixo-Dir é natural
                return FlowFixer.DOWN_RIGHT
        
        # Se estava em baixo, melhor cortar DIAGONAL CIMA
        else:
            if hand == FlowFixer.LEFT_HAND:
                return FlowFixer.UP_LEFT
            else:
                return FlowFixer.UP_RIGHT
                
        return FlowFixer.DOT # Fallback

    @staticmethod
    def _get_inverse_cut(cut):
        pairs = {
            FlowFixer.UP: FlowFixer.DOWN,
            FlowFixer.DOWN: FlowFixer.UP,
            FlowFixer.LEFT: FlowFixer.RIGHT,
            FlowFixer.RIGHT: FlowFixer.LEFT,
            FlowFixer.UP_LEFT: FlowFixer.DOWN_RIGHT,
            FlowFixer.UP_RIGHT: FlowFixer.DOWN_LEFT,
            FlowFixer.DOWN_LEFT: FlowFixer.UP_RIGHT,
            FlowFixer.DOWN_RIGHT: FlowFixer.UP_LEFT
        }
        return pairs.get(cut, FlowFixer.DOT)

    @staticmethod
    def _is_bad_angle(cut, line, hand):
        # Evita cortes para fora nas extremidades (torção de pulso)
        if hand == FlowFixer.LEFT_HAND:
            if line == 0 and cut in [FlowFixer.RIGHT, FlowFixer.UP_RIGHT, FlowFixer.DOWN_RIGHT]: return True
        if hand == FlowFixer.RIGHT_HAND:
            if line == 3 and cut in [FlowFixer.LEFT, FlowFixer.UP_LEFT, FlowFixer.DOWN_LEFT]: return True
        return False
