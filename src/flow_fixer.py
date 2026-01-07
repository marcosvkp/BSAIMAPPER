import math

class FlowFixer:
    """
    Simulador de Paridade e Corretor de Flow.
    Segue estritamente as regras de mapas Rankeados (ScoreSaber/BeatLeader).
    """

    # Constantes de Direção (Beat Saber v2)
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
        """
        Executa a revisão completa do mapa.
        Retorna a lista de notas corrigida e com bombas adicionadas.
        """
        if not notes:
            return []

        # Ordenação temporal é obrigatória
        notes.sort(key=lambda x: x['_time'])
        
        # Separação por mão
        left_notes = [n for n in notes if n['_type'] == FlowFixer.LEFT_HAND]
        right_notes = [n for n in notes if n['_type'] == FlowFixer.RIGHT_HAND]
        
        # Processamento individual (Paridade é calculada por mão)
        fixed_left = FlowFixer._process_hand(left_notes, bpm, FlowFixer.LEFT_HAND)
        fixed_right = FlowFixer._process_hand(right_notes, bpm, FlowFixer.RIGHT_HAND)
        
        # Recombinação
        all_objects = fixed_left + fixed_right
        all_objects.sort(key=lambda x: x['_time'])
        
        return all_objects

    @staticmethod
    def _process_hand(hand_notes, bpm, hand_type):
        if not hand_notes:
            return []

        processed_objects = []
        seconds_per_beat = 60 / bpm
        
        # Limiar de Reset (Ranked Standard ~0.25s a 0.5s dependendo do BPM)
        # Se o tempo for menor que isso, o movimento DEVE ser contínuo.
        # Se for maior, pode haver reset (com bomba).
        RESET_THRESHOLD = 0.35 

        # Estado inicial: Assumimos que o braço começa neutro ou em baixo
        # Vamos rastrear a "Intenção do Movimento Anterior"
        # True = Movimento foi para CIMA (braço terminou em cima)
        # False = Movimento foi para BAIXO (braço terminou em baixo)
        prev_ended_up = False 
        prev_note = hand_notes[0]
        processed_objects.append(prev_note)

        # Determina estado inicial baseado na primeira nota
        if prev_note['_cutDirection'] in [FlowFixer.UP, FlowFixer.UP_LEFT, FlowFixer.UP_RIGHT]:
            prev_ended_up = True # Cortou pra cima, braço está em cima
        
        for i in range(1, len(hand_notes)):
            curr_note = hand_notes[i]
            
            time_diff = (curr_note['_time'] - prev_note['_time']) * seconds_per_beat
            
            # Análise da nota atual
            curr_cut = curr_note['_cutDirection']
            is_dot = curr_cut == FlowFixer.DOT
            
            # Determina a intenção da nota atual
            # Se for DOT, tentamos inferir pela posição (Layer 2 = Cima, Layer 0 = Baixo)
            curr_goes_up = False
            if curr_cut in [FlowFixer.UP, FlowFixer.UP_LEFT, FlowFixer.UP_RIGHT]:
                curr_goes_up = True
            elif curr_cut in [FlowFixer.DOWN, FlowFixer.DOWN_LEFT, FlowFixer.DOWN_RIGHT]:
                curr_goes_up = False
            elif is_dot:
                # Dot note: Se está na layer de cima, assume que o braço vai pra cima
                curr_goes_up = curr_note['_lineLayer'] >= 1
            else:
                # Laterais (Left/Right) dependem do contexto, mas geralmente mantêm a altura
                # Vamos assumir alternância simples se for lateral
                curr_goes_up = not prev_ended_up

            # --- DETECÇÃO DE QUEBRA DE FLOW (PARIDADE) ---
            # Erro: Braço estava em cima (prev_ended_up) E tenta cortar pra cima de novo (curr_goes_up)
            # Erro: Braço estava em baixo (not prev_ended_up) E tenta cortar pra baixo de novo (not curr_goes_up)
            is_reset = (prev_ended_up and curr_goes_up) or (not prev_ended_up and not curr_goes_up)

            if is_reset:
                if time_diff < RESET_THRESHOLD:
                    # CASO 1: Reset Invisível Rápido (PROIBIDO) -> Corrigir Nota
                    # Solução: Inverter a direção para manter o flow contínuo
                    
                    new_cut = FlowFixer._get_inverse_cut(curr_cut)
                    
                    # Verifica ergonomia da inversão
                    if FlowFixer._is_bad_angle(new_cut, curr_note['_lineIndex'], hand_type):
                        curr_note['_cutDirection'] = FlowFixer.DOT
                    else:
                        curr_note['_cutDirection'] = new_cut
                    
                    # Atualiza o estado lógico (agora o flow foi corrigido)
                    curr_goes_up = not prev_ended_up 

                else:
                    # CASO 2: Reset Lento (PERMITIDO COM SINALIZAÇÃO) -> Inserir Bomba
                    # O jogador tem tempo de resetar, mas precisamos avisar.
                    # Inserimos uma bomba na posição oposta para forçar o reset.
                    
                    bomb_layer = 2 if prev_ended_up else 0 # Se terminou em cima, bomba em cima pra forçar descer (sem cortar)
                    bomb = {
                        "_time": (prev_note['_time'] + curr_note['_time']) / 2,
                        "_lineIndex": curr_note['_lineIndex'],
                        "_lineLayer": bomb_layer,
                        "_type": FlowFixer.BOMB,
                        "_cutDirection": 0
                    }
                    processed_objects.append(bomb)
                    
                    # O reset é aceito, então o estado do braço "teletransporta" para a posição inicial do próximo corte
                    # Se o próximo corte vai pra cima, o braço reseta pra baixo.
                    pass 

            # Adiciona a nota (possivelmente corrigida)
            processed_objects.append(curr_note)
            
            # Atualiza estado para a próxima iteração
            prev_note = curr_note
            prev_ended_up = curr_goes_up

        return processed_objects

    @staticmethod
    def _get_inverse_cut(cut):
        """Retorna a direção oposta para manter o flow."""
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
        """
        Verifica se o ângulo é desconfortável (ex: torção de pulso).
        Ranked maps evitam cortes 'para fora' nas extremidades opostas.
        """
        # Mão Esquerda (0)
        if hand == FlowFixer.LEFT_HAND:
            # Na extrema esquerda (0), cortar para direita (3) é ruim
            if line == 0 and cut in [FlowFixer.RIGHT, FlowFixer.UP_RIGHT, FlowFixer.DOWN_RIGHT]:
                return True
        
        # Mão Direita (1)
        if hand == FlowFixer.RIGHT_HAND:
            # Na extrema direita (3), cortar para esquerda (2) é ruim
            if line == 3 and cut in [FlowFixer.LEFT, FlowFixer.UP_LEFT, FlowFixer.DOWN_LEFT]:
                return True
                
        return False
