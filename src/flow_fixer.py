import math

class FlowFixer:
    """
    Simulador de Paridade e Corretor de Flow (V4 - Híbrido).
    Mantém a liberdade da V2, as bombas da V3 e limpa o meio.
    """

    # Constantes
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = 4, 5, 6, 7
    DOT = 8
    LEFT_HAND, RIGHT_HAND = 0, 1
    BOMB = 3

    @staticmethod
    def fix(notes, bpm):
        if not notes: return []
        notes.sort(key=lambda x: x['_time'])
        
        left = [n for n in notes if n['_type'] == FlowFixer.LEFT_HAND]
        right = [n for n in notes if n['_type'] == FlowFixer.RIGHT_HAND]
        
        fixed_left = FlowFixer._process_hand(left, bpm, FlowFixer.LEFT_HAND)
        fixed_right = FlowFixer._process_hand(right, bpm, FlowFixer.RIGHT_HAND)
        
        all_obj = fixed_left + fixed_right
        all_obj.sort(key=lambda x: x['_time'])
        return all_obj

    @staticmethod
    def _process_hand(hand_notes, bpm, hand_type):
        if not hand_notes: return []

        processed = []
        sec_per_beat = 60 / bpm
        BOMB_THRESHOLD = 0.8 

        prev_ended_up = False 
        prev_note = hand_notes[0]
        
        # Sanitização inicial
        prev_note = FlowFixer._sanitize_note(prev_note, hand_type)
        processed.append(prev_note)

        if prev_note['_cutDirection'] in [0, 4, 5]: prev_ended_up = True
        
        for i in range(1, len(hand_notes)):
            curr_note = hand_notes[i]
            time_diff = (curr_note['_time'] - prev_note['_time']) * sec_per_beat
            
            # --- 1. SANITIZAÇÃO (Ajuste solicitado) ---
            # Remove cortes laterais no meio e cortes ruins nas pontas
            curr_note = FlowFixer._sanitize_note(curr_note, hand_type)

            # --- 2. PARIDADE E RESETS ---
            curr_cut = curr_note['_cutDirection']
            
            # Determina intenção
            curr_goes_up = False
            if curr_cut in [0, 4, 5]: curr_goes_up = True
            elif curr_cut in [1, 6, 7]: curr_goes_up = False
            elif curr_cut == 8: curr_goes_up = curr_note['_lineLayer'] >= 1
            else: curr_goes_up = prev_ended_up # Laterais mantêm estado

            is_reset = (prev_ended_up and curr_goes_up) or (not prev_ended_up and not curr_goes_up)

            if is_reset:
                if time_diff < BOMB_THRESHOLD:
                    # Reset Rápido -> Inverter Nota
                    new_cut = FlowFixer._get_inverse_cut(curr_cut)
                    # Verifica ergonomia da inversão
                    if FlowFixer._is_bad_angle(new_cut, curr_note['_lineIndex'], hand_type):
                        curr_note['_cutDirection'] = 8 # Dot
                    else:
                        curr_note['_cutDirection'] = new_cut
                    curr_goes_up = not prev_ended_up 

                else:
                    # Reset Longo -> PAREDE DE BOMBAS (V3 Style)
                    # Gera bombas nas duas colunas da mão, na layer 0
                    bomb_time = (prev_note['_time'] + curr_note['_time']) / 2
                    
                    cols = [0, 1] if hand_type == 0 else [2, 3]
                    for col in cols:
                        processed.append({
                            "_time": bomb_time,
                            "_lineIndex": col,
                            "_lineLayer": 0, # Sempre no chão
                            "_type": 3,
                            "_cutDirection": 0
                        })
            
            processed.append(curr_note)
            prev_note = curr_note
            prev_ended_up = curr_goes_up

        return processed

    @staticmethod
    def _sanitize_note(note, hand):
        """
        Remove padrões proibidos.
        """
        line = note['_lineIndex']
        cut = note['_cutDirection']
        
        # REGRA CRÍTICA: Remover laterais no meio
        # Se a nota está na coluna 1 ou 2 E o corte é Esquerda (2) ou Direita (3)
        if line in [1, 2] and cut in [2, 3]:
            # Transforma em Dot Note (8)
            # Isso mantém o ritmo mas remove a exigência angular ruim
            note['_cutDirection'] = 8
            return note

        # REGRA: Proibido cortar para fora nas extremidades (Torção de pulso)
        if hand == 0 and line == 0 and cut in [3, 5, 7]: note['_cutDirection'] = 8
        if hand == 1 and line == 3 and cut in [2, 4, 6]: note['_cutDirection'] = 8
        
        return note

    @staticmethod
    def _get_inverse_cut(cut):
        pairs = {0:1, 1:0, 2:3, 3:2, 4:7, 5:6, 6:5, 7:4}
        return pairs.get(cut, 8)

    @staticmethod
    def _is_bad_angle(cut, line, hand):
        if hand == 0 and line == 0 and cut in [3, 5, 7]: return True
        if hand == 1 and line == 3 and cut in [2, 4, 6]: return True
        return False
