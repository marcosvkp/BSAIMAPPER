import math

class FlowFixer:
    """
    Simulador de Paridade e Corretor de Flow (V6 - Resets Longos e Seguros).
    Foco: Bombas apenas em pausas longas e reentrada correta (Down Cut).
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
        
        fixed_left, resets_left = FlowFixer._process_hand(left, bpm, FlowFixer.LEFT_HAND)
        fixed_right, resets_right = FlowFixer._process_hand(right, bpm, FlowFixer.RIGHT_HAND)
        
        # Unifica os tempos de reset
        all_resets = sorted(list(set(resets_left + resets_right)))
        
        # Gera Parede de Bombas Global (4 colunas)
        bombs = []
        for t in all_resets:
            for col in range(4): 
                bombs.append({
                    "_time": t,
                    "_lineIndex": col,
                    "_lineLayer": 0, # Chão
                    "_type": FlowFixer.BOMB,
                    "_cutDirection": 0
                })

        all_obj = fixed_left + fixed_right + bombs
        all_obj.sort(key=lambda x: x['_time'])
        return all_obj

    @staticmethod
    def _process_hand(hand_notes, bpm, hand_type):
        if not hand_notes: return [], []

        processed = []
        resets = []
        sec_per_beat = 60 / bpm
        
        # --- REGRA DE OURO: TEMPO MÍNIMO PARA BOMBA ---
        # Só gera bomba se houver um buraco de pelo menos 3.0 segundos (ex: 1.5s antes, 1.5s depois)
        BOMB_THRESHOLD = 3.0 

        prev_ended_up = False 
        prev_note = hand_notes[0]
        
        prev_note = FlowFixer._sanitize_note(prev_note, hand_type)
        processed.append(prev_note)

        if prev_note['_cutDirection'] in [0, 4, 5]: prev_ended_up = True
        
        for i in range(1, len(hand_notes)):
            curr_note = hand_notes[i]
            time_diff = (curr_note['_time'] - prev_note['_time']) * sec_per_beat
            
            curr_note = FlowFixer._sanitize_note(curr_note, hand_type)
            curr_cut = curr_note['_cutDirection']
            
            # Determina intenção atual
            curr_goes_up = False
            if curr_cut in [0, 4, 5]: curr_goes_up = True
            elif curr_cut in [1, 6, 7]: curr_goes_up = False
            elif curr_cut == 8: curr_goes_up = curr_note['_lineLayer'] >= 1
            else: curr_goes_up = prev_ended_up 

            # Verifica necessidade de Reset (Quebra de Flow)
            is_reset_needed = (prev_ended_up and curr_goes_up) or (not prev_ended_up and not curr_goes_up)

            # Lógica de Decisão: Corrigir ou Resetar?
            if is_reset_needed:
                if time_diff < BOMB_THRESHOLD:
                    # --- CASO 1: Intervalo Curto/Médio -> CORRIGIR NOTA ---
                    # Não há tempo para reset confortável com bomba.
                    # Solução: Inverter a nota para manter o flow contínuo.
                    new_cut = FlowFixer._get_inverse_cut(curr_cut)
                    
                    if FlowFixer._is_bad_angle(new_cut, curr_note['_lineIndex'], hand_type):
                        curr_note['_cutDirection'] = 8 # Dot
                    else:
                        curr_note['_cutDirection'] = new_cut
                    
                    # O flow foi corrigido, então a intenção inverteu
                    curr_goes_up = not prev_ended_up 

                else:
                    # --- CASO 2: Intervalo Longo (> 3s) -> RESET COM BOMBA ---
                    bomb_time = (prev_note['_time'] + curr_note['_time']) / 2
                    resets.append(bomb_time)
                    
                    # --- CORREÇÃO DE REENTRADA (CRÍTICO) ---
                    # Se houve bomba no chão, o jogador levantou a mão.
                    # A próxima nota TEM que ser um corte para BAIXO (ou Dot).
                    # Se for corte para CIMA, é impossível (braço já está em cima).
                    if curr_goes_up: # Se a nota original era pra cima
                        # Força para baixo
                        curr_note['_cutDirection'] = 1 # Down
                        curr_goes_up = False # Agora vai pra baixo
            
            processed.append(curr_note)
            prev_note = curr_note
            prev_ended_up = curr_goes_up

        return processed, resets

    @staticmethod
    def _sanitize_note(note, hand):
        line = note['_lineIndex']
        cut = note['_cutDirection']
        
        # Remove laterais no meio
        if line in [1, 2] and cut in [2, 3]:
            note['_cutDirection'] = 8
            return note

        # Remove cortes para fora nas pontas
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
