import math

class FlowFixer:
    """
    Simulador de Paridade Estrita (V7).
    Foco: Garantir fluxo ABAB (Cima/Baixo) contínuo e corrigir streams quebrados.
    Removeu-se a geração agressiva de bombas em favor de inverter notas erradas.
    """

    # Constantes de Direção
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = 4, 5, 6, 7
    DOT = 8
    
    LEFT_HAND = 0
    RIGHT_HAND = 1
    BOMB = 3

    @staticmethod
    def fix(notes, bpm):
        if not notes: return []
        
        # Separa por cor/mão
        left_notes = [n for n in notes if n['_type'] == FlowFixer.LEFT_HAND]
        right_notes = [n for n in notes if n['_type'] == FlowFixer.RIGHT_HAND]
        bombs = [n for n in notes if n['_type'] == FlowFixer.BOMB] # Mantém bombas originais se houver
        
        # Processa cada mão independentemente
        fixed_left = FlowFixer._process_hand_strict(left_notes, FlowFixer.LEFT_HAND)
        fixed_right = FlowFixer._process_hand_strict(right_notes, FlowFixer.RIGHT_HAND)
        
        # Junta tudo e reordena
        all_obj = fixed_left + fixed_right + bombs
        all_obj.sort(key=lambda x: x['_time'])
        return all_obj

    @staticmethod
    def _process_hand_strict(hand_notes, hand_type):
        if not hand_notes: return []
        
        hand_notes.sort(key=lambda x: x['_time'])
        processed = []
        
        # Estado inicial: assumimos que a mão começa em baixo (neutro), então a primeira nota idealmente corta pra CIMA.
        # Se a primeira nota for pra BAIXO, ok, aceitamos, mas definimos o estado de acordo.
        
        # True = Mão terminou em CIMA (precisa descer)
        # False = Mão terminou em BAIXO (precisa subir)
        last_ended_high = False 
        
        # Ajuste inicial baseado na primeira nota
        first_cut = hand_notes[0]['_cutDirection']
        if first_cut in [0, 4, 5]: last_ended_high = True # Cortou pra cima, mão tá em cima
        elif first_cut in [1, 6, 7]: last_ended_high = False # Cortou pra baixo, mão tá em baixo
        
        processed.append(hand_notes[0])
        
        for i in range(1, len(hand_notes)):
            prev_note = processed[-1]
            curr_note = hand_notes[i]
            
            # --- Sanitização Básica (Evitar cortes para fora nas pontas) ---
            line = curr_note['_lineIndex']
            cut = curr_note['_cutDirection']
            
            if hand_type == 0 and line == 0 and cut in [3, 5, 7]: curr_note['_cutDirection'] = 8 # Esq canto esq cortando pra esq -> DOT
            if hand_type == 1 and line == 3 and cut in [2, 4, 6]: curr_note['_cutDirection'] = 8 # Dir canto dir cortando pra dir -> DOT

            # --- Análise de Fluxo ---
            # Determina a direção vertical da nota atual
            current_goes_up = False   # Intenção de subir (Corte Cima)
            current_goes_down = False # Intenção de descer (Corte Baixo)
            
            if curr_note['_cutDirection'] in [0, 4, 5]: current_goes_up = True
            elif curr_note['_cutDirection'] in [1, 6, 7]: current_goes_down = True
            
            # Se for lateral ou dot, tentamos inferir baseada na posição da camada (y)
            # Mas na dúvida, forçamos o flow oposto ao anterior
            is_ambiguous = (not current_goes_up) and (not current_goes_down)

            # --- Correção de Paridade ---
            # Se a mão terminou em CIMA (last_ended_high), a próxima nota DEVE ser para baixo (ou neutra que permita descer)
            # Se a mão terminou em BAIXO (!last_ended_high), a próxima nota DEVE ser para cima
            
            must_go_down = last_ended_high
            
            fix_applied = False
            
            if must_go_down:
                # Esperamos um corte para BAIXO.
                if current_goes_up:
                    # ERRO: Mão tá em cima e nota pede pra cortar pra cima.
                    # Correção: Inverter para Baixo
                    curr_note['_cutDirection'] = 1 # Down
                    fix_applied = True
                elif is_ambiguous:
                    # Se é lateral/dot e precisamos descer, forçamos um viés de descida se necessário, 
                    # mas geralmente dots aceitam qualquer fluxo. Mantemos o dot.
                    pass
            else:
                # Esperamos um corte para CIMA.
                if current_goes_down:
                    # ERRO: Mão tá em baixo e nota pede pra cortar pra baixo.
                    # Correção: Inverter para Cima
                    curr_note['_cutDirection'] = 0 # Up
                    fix_applied = True
                elif is_ambiguous:
                    pass

            # --- Atualiza o estado para a próxima iteração ---
            # Recalcula a direção final da nota (pode ter sido alterada pelo fix)
            final_cut = curr_note['_cutDirection']
            
            if final_cut in [0, 4, 5]: last_ended_high = True
            elif final_cut in [1, 6, 7]: last_ended_high = False
            else:
                # Se for DOT ou Lateral, assume que o fluxo continuou e inverteu a posição
                last_ended_high = not last_ended_high 
            
            processed.append(curr_note)

        return processed
