import math

class FlowFixer:
    """
    Responsável por garantir a integridade física e o Flow do mapa.
    Atua como um 'revisor' pós-geração.
    """

    # Definição de direções
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    DOT = 8

    @staticmethod
    def fix(notes, bpm):
        """
        Analisa e corrige a lista de notas para garantir flow contínuo.
        """
        if not notes:
            return []

        # Ordenar por tempo é crucial
        notes.sort(key=lambda x: x['_time'])
        
        # Separar notas por mão para análise de física individual
        left_hand_notes = [n for n in notes if n['_type'] == 0]
        right_hand_notes = [n for n in notes if n['_type'] == 1]

        # Corrigir cada mão individualmente
        fixed_left = FlowFixer._process_hand(left_hand_notes, bpm, hand_type=0)
        fixed_right = FlowFixer._process_hand(right_hand_notes, bpm, hand_type=1)

        # Recombinar e reordenar
        all_notes = fixed_left + fixed_right
        all_notes.sort(key=lambda x: x['_time'])
        
        return all_notes

    @staticmethod
    def _process_hand(hand_notes, bpm, hand_type):
        if not hand_notes:
            return []

        seconds_per_beat = 60 / bpm
        # Tempo mínimo para permitir um reset (ex: pausa para levantar o braço)
        # 0.5s é um valor seguro para jogadores medianos/bons
        reset_threshold_time = 0.4 

        prev_note = hand_notes[0]
        
        for i in range(1, len(hand_notes)):
            curr_note = hand_notes[i]
            
            time_diff_beats = curr_note['_time'] - prev_note['_time']
            time_diff_seconds = time_diff_beats * seconds_per_beat

            # Se for muito rápido, o flow TEM que ser contínuo
            if time_diff_seconds < reset_threshold_time:
                prev_cut = prev_note['_cutDirection']
                curr_cut = curr_note['_cutDirection']

                # REGRA CRÍTICA: Verificar Resets Invisíveis
                if FlowFixer._is_bad_flow(prev_cut, curr_cut):
                    # Detectou problema! (Ex: Cima -> Cima)
                    
                    # Tenta corrigir invertendo a direção (Cima -> Baixo)
                    new_cut = FlowFixer._get_inverse_cut(prev_cut)
                    
                    # Se a inversão criar um ângulo estranho (ex: cruzar braços), usa DOT
                    if FlowFixer._is_awkward_angle(new_cut, curr_note['_lineIndex'], hand_type):
                        curr_note['_cutDirection'] = FlowFixer.DOT
                    else:
                        curr_note['_cutDirection'] = new_cut
            
            prev_note = curr_note
            
        return hand_notes

    @staticmethod
    def _is_bad_flow(prev, curr):
        """
        Retorna True se a transição quebra o flow (Reset Invisível).
        Ex: Cima (0) seguido de Cima (0) ou Cima-Esq (4).
        """
        if prev == FlowFixer.DOT: return False # Dot reseta o flow suavemente
        if curr == FlowFixer.DOT: return False

        # Grupos de direção "Para Cima"
        up_group = [FlowFixer.UP, FlowFixer.UP_LEFT, FlowFixer.UP_RIGHT]
        # Grupos de direção "Para Baixo"
        down_group = [FlowFixer.DOWN, FlowFixer.DOWN_LEFT, FlowFixer.DOWN_RIGHT]
        # Grupos laterais
        left_group = [FlowFixer.LEFT, FlowFixer.UP_LEFT, FlowFixer.DOWN_LEFT]
        right_group = [FlowFixer.RIGHT, FlowFixer.UP_RIGHT, FlowFixer.DOWN_RIGHT]

        # Regra: Se cortou pra cima, não pode cortar pra cima de novo
        if prev in up_group and curr in up_group: return True
        if prev in down_group and curr in down_group: return True
        
        # Regra: Se cortou pra esquerda, não pode cortar pra esquerda de novo (geralmente)
        if prev in left_group and curr in left_group: return True
        if prev in right_group and curr in right_group: return True

        return False

    @staticmethod
    def _get_inverse_cut(cut):
        """Retorna a direção oposta natural."""
        mapping = {
            FlowFixer.UP: FlowFixer.DOWN,
            FlowFixer.DOWN: FlowFixer.UP,
            FlowFixer.LEFT: FlowFixer.RIGHT,
            FlowFixer.RIGHT: FlowFixer.LEFT,
            FlowFixer.UP_LEFT: FlowFixer.DOWN_RIGHT,
            FlowFixer.UP_RIGHT: FlowFixer.DOWN_LEFT,
            FlowFixer.DOWN_LEFT: FlowFixer.UP_RIGHT,
            FlowFixer.DOWN_RIGHT: FlowFixer.UP_LEFT,
            FlowFixer.DOT: FlowFixer.DOT
        }
        return mapping.get(cut, FlowFixer.DOT)

    @staticmethod
    def _is_awkward_angle(cut, line_index, hand):
        """
        Evita movimentos que torcem o pulso ou cruzam braços desnecessariamente.
        """
        # Mão Esquerda (0) na extrema direita (3) cortando pra esquerda (2) = OK
        # Mão Esquerda (0) na extrema esquerda (0) cortando pra direita (3) = Ruim (pulso)
        
        if hand == 0: # Esquerda
            # Evitar cortar para a direita se estiver muito na esquerda (rotação de pulso)
            if line_index <= 1 and cut in [FlowFixer.RIGHT, FlowFixer.UP_RIGHT, FlowFixer.DOWN_RIGHT]:
                return True
        else: # Direita
            # Evitar cortar para a esquerda se estiver muito na direita
            if line_index >= 2 and cut in [FlowFixer.LEFT, FlowFixer.UP_LEFT, FlowFixer.DOWN_LEFT]:
                return True
                
        return False
