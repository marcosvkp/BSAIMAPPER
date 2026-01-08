import math

class FlowFixer:
    """
    Simulador de Paridade e Corretor de Flow (V9 - Stream Safety).
    Foco: Forçar a correção de flow mesmo em streams rápidos.
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
        
        all_resets = sorted(list(set(resets_left + resets_right)))
        
        bombs = []
        for t in all_resets:
            for col in range(4): 
                bombs.append({
                    "_time": t,
                    "_lineIndex": col,
                    "_lineLayer": 0,
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
            
            # Determina a direção de corte da nota atual
            curr_goes_up = False
            if curr_cut in [0, 4, 5]: curr_goes_up = True
            elif curr_cut in [1, 6, 7]: curr_goes_up = False
            elif curr_cut == 8: # Dot notes são ambíguos
                # Assumimos que o jogador continuará o flow, então a direção efetiva é a oposta da anterior.
                curr_goes_up = not prev_ended_up
            
            # Verifica se o flow foi quebrado (ex: cima -> cima ou baixo -> baixo)
            is_reset_needed = (prev_ended_up and curr_goes_up) or (not prev_ended_up and not curr_goes_up)

            if is_reset_needed:
                # Se o intervalo for longo, insere uma bomba para forçar o reset do jogador.
                if time_diff >= BOMB_THRESHOLD:
                    bomb_time = (prev_note['_time'] + curr_note['_time']) / 2
                    resets.append(bomb_time)
                    
                    # Após uma bomba, a próxima nota DEVE ser para baixo.
                    if curr_goes_up:
                        curr_note['_cutDirection'] = 1 # Força corte para baixo
                        curr_goes_up = False
                else:
                    # Se o intervalo for curto (streams, etc.), corrige a nota invertendo o corte.
                    new_cut = FlowFixer._get_inverse_cut(curr_cut)
                    
                    is_inverted = (hand_type == 0 and curr_note['_lineIndex'] > 1) or \
                                  (hand_type == 1 and curr_note['_lineIndex'] < 2)
                    
                    # Se o ângulo invertido for ruim, usa um dot note como último recurso.
                    if FlowFixer._is_bad_angle(new_cut, curr_note['_lineIndex'], hand_type, is_inverted):
                        curr_note['_cutDirection'] = 8
                    else:
                        curr_note['_cutDirection'] = new_cut
                    
                    # A direção do corte foi corrigida.
                    curr_goes_up = not prev_ended_up 
            
            processed.append(curr_note)
            prev_note = curr_note
            prev_ended_up = curr_goes_up

        return processed, resets

    @staticmethod
    def _sanitize_note(note, hand):
        line = int(note['_lineIndex'])
        cut = int(note['_cutDirection'])
        
        # Evita cortes laterais (esquerda/direita) nas colunas centrais (1 e 2)
        if line in [1, 2] and cut in [2, 3]:
            note['_cutDirection'] = 8 # Converte para Dot Note
        
        return note

    @staticmethod
    def _get_inverse_cut(cut):
        pairs = {0:1, 1:0, 2:3, 3:2, 4:7, 5:6, 6:5, 7:4}
        return pairs.get(cut, 8)

    @staticmethod
    def _is_bad_angle(cut, line, hand, is_inverted=False):
        if is_inverted:
            if hand == 0 and line == 3 and cut in [2, 4, 6]: return True
            if hand == 1 and line == 0 and cut in [3, 5, 7]: return True
        else:
            if hand == 0 and line == 0 and cut in [3, 5, 7]: return True
            if hand == 1 and line == 3 and cut in [2, 4, 6]: return True
        return False
