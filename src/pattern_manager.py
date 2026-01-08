import random

class FlowState:
    """Mantém o estado da última nota para cada mão para ajudar no fluxo."""
    def __init__(self, hand):
        self.hand = hand
        self.x = 1 if hand == 0 else 2
        self.y = 0
        self.cut = 1 # Começa cortando para baixo

    def update(self, x, y, cut):
        self.x = x
        self.y = y
        self.cut = cut

class PatternManager:
    """
    Classe V2 que decide qual padrão de nota usar e, crucialmente,
    utiliza a previsão de direção de corte do modelo para um fluxo mais inteligente.
    """
    def __init__(self, difficulty="ExpertPlus"):
        self.left = FlowState(0)
        self.right = FlowState(1)
        self.parity = 1 # 1=Direita, 0=Esquerda
        self.difficulty = difficulty
        
        self.patterns = {
            0: ['single_flow', 'stack_simple'],
            1: ['wide_flow', 'diagonal_cross', 'window_wipe'],
            2: ['stream_burst', 'tech_angle', 'double_down', 'tower_stack', 'burst_fill', 'super_stream']
        }

    def get_pattern(self, intensity, complexity_idx, vertical_idx, time_gap, energy_level=0.5, predicted_cut_direction=None):
        """
        Seleciona um padrão de nota com base nas previsões da IA, energia da música e regras de dificuldade.
        Agora inclui a previsão de corte no metadado.
        """
        max_complexity = 2
        allow_bursts = True
        
        # Restrições baseadas na dificuldade selecionada
        if self.difficulty == "Easy":
            max_complexity = 0
            allow_bursts = False
        elif self.difficulty == "Normal":
            max_complexity = 1
            allow_bursts = False
        elif self.difficulty == "Hard":
            max_complexity = 1
            if energy_level > 0.8: max_complexity = 2
            allow_bursts = False
        
        # Ajusta a complexidade com base na energia e nos limites da dificuldade
        if energy_level < 0.3: complexity_idx = 0
        complexity_idx = min(complexity_idx, max_complexity)

        # Cooldown para evitar notas muito próximas em seções de baixa energia
        min_gap_map = {"Easy": 0.5, "Normal": 0.3, "Hard": 0.2}
        min_gap = min_gap_map.get(self.difficulty, 0.12)
        if time_gap < min_gap: return None
        
        # Lógica para iniciar streams/bursts em momentos de alta energia
        if self.difficulty in ["Expert", "ExpertPlus"]:
            if time_gap < 0.22 and energy_level > 0.5:
                return {'type': 'stream_burst', 'vert': vertical_idx, 'predicted_cut': predicted_cut_direction}
            
        options = self.patterns.get(complexity_idx, self.patterns[1])
        
        if not allow_bursts:
            options = [p for p in options if 'burst' not in p and 'stream' not in p]
            if not options: options = self.patterns[0]

        # Lógica para padrões de alta densidade (drops)
        if allow_bursts and energy_level > 0.65:
            if energy_level > 0.85 and random.random() < 0.3:
                return {'type': 'super_stream', 'vert': vertical_idx, 'predicted_cut': predicted_cut_direction}
            if random.random() < 0.2 + (energy_level - 0.65) * 1.3:
                return {'type': 'burst_fill', 'vert': vertical_idx, 'predicted_cut': predicted_cut_direction}
            if random.random() < 0.25:
                return {'type': 'double_down', 'vert': vertical_idx, 'predicted_cut': predicted_cut_direction}

        chosen_type = random.choice(options)
        
        return {
            'type': chosen_type,
            'vert': vertical_idx,
            'predicted_cut': predicted_cut_direction # Passa a previsão de corte adiante
        }

    def apply_pattern(self, meta, time, bpm):
        """Aplica o padrão selecionado, usando a previsão de corte do modelo."""
        if not meta: return []
        
        ptype = meta['type']
        vert_bias = meta['vert']
        predicted_cut = meta.get('predicted_cut') # Obtém a previsão de corte
        
        # Mapeia o tipo de padrão para a função de geração correspondente
        pattern_functions = {
            'single_flow': self._gen_flow, 'stack_simple': self._gen_flow,
            'wide_flow': self._gen_wide, 'diagonal_cross': self._gen_wide, 'window_wipe': self._gen_wide,
            'stream_burst': self._gen_flow,
            'double_down': self._gen_double,
            'tech_angle': self._gen_tech,
            'tower_stack': self._gen_tower,
            'burst_fill': lambda t, v, pc: self._gen_burst(t, bpm, v, length=4),
            'super_stream': lambda t, v, pc: self._gen_burst(t, bpm, v, length=8)
        }
        
        gen_func = pattern_functions.get(ptype, self._gen_flow)
        notes = gen_func(time, vert_bias, predicted_cut)
            
        if ptype != 'double_down':
            self.parity = 1 - self.parity
        return notes

    def _get_fallback_cut(self, hand, target_y, state):
        """Lógica de corte V1, usada como fallback se o modelo não prever um ângulo."""
        if target_y > state.y: cut = 0 # Cima
        else: cut = 1 # Baixo
        
        # Adiciona variedade com cortes diagonais
        if random.random() > 0.7:
            if hand == 0: cut = 6 if cut == 1 else 4 # Diagonal baixo-esq / cima-esq
            else: cut = 7 if cut == 1 else 5 # Diagonal baixo-dir / cima-dir
        return cut

    def _gen_flow(self, time, v_bias, predicted_cut=None):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        # Define posição alvo
        target_y = random.choice([0, 1, 2])
        if v_bias == 2: target_y = max(1, target_y) # Bias para cima
        if v_bias == 0: target_y = min(1, target_y) # Bias para baixo
        
        # **LÓGICA DE CORTE V2**
        if predicted_cut is not None and predicted_cut != 8: # 8 = 'any', então usamos fallback
            cut = predicted_cut
        else:
            cut = self._get_fallback_cut(hand, target_y, state)
            
        line = random.choice([0, 1]) if hand == 0 else random.choice([2, 3])
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_wide(self, time, v_bias, predicted_cut=None):
        hand = self.parity
        line = 0 if hand == 0 else 3
        layer = 0 if v_bias == 0 else 1
        
        if predicted_cut is not None and predicted_cut != 8:
            cut = predicted_cut
        else:
            cut = 7 if hand == 0 else 6 # Fallback para corte diagonal para dentro
            
        return self._create_note(hand, time, line, layer, cut)

    def _gen_double(self, time, v_bias, predicted_cut=None):
        cut = predicted_cut if predicted_cut is not None and predicted_cut != 8 else 1 # Fallback para corte baixo
        l_note = self._create_note(0, time, 1, 0, cut)[0]
        r_note = self._create_note(1, time, 2, 0, cut)[0]
        return [l_note, r_note]

    def _gen_tech(self, time, v_bias, predicted_cut=None):
        hand = self.parity
        line = 1 if hand == 0 else 2
        layer = 1
        
        if predicted_cut is not None and predicted_cut != 8:
            cut = predicted_cut
        else:
            cut = 2 if hand == 0 else 3 # Fallback para corte lateral
            
        return self._create_note(hand, time, line, layer, cut)

    def _gen_tower(self, time, v_bias, predicted_cut=None):
        hand = self.parity
        line = 1 if hand == 0 else 2
        cut = predicted_cut if predicted_cut is not None and predicted_cut != 8 else 1 # Fallback para corte baixo
        n1 = self._create_note(hand, time, line, 0, cut)[0]
        n2 = self._create_note(hand, time, line, 2, cut)[0]
        return [n1, n2]

    def _gen_burst(self, time, bpm, v_bias, length=4):
        """Gera um burst de notas com fluxo interno. A previsão de corte não se aplica aqui."""
        notes = []
        step = 0.25 # 1/4 de beat
        current_hand = self.parity
        
        for i in range(length):
            t = time + (i * step)
            line = 1 if current_hand == 0 else 2
            layer = 0
            cut = 1 if i % 2 == 0 else 0 # Alterna cima/baixo
            
            note = self._create_note(current_hand, t, line, layer, cut)[0]
            notes.append(note)
            current_hand = 1 - current_hand
            
        self.parity = current_hand
        return notes

    def _create_note(self, hand, time, line, layer, cut):
        """Cria a estrutura de dicionário para uma única nota e atualiza o estado."""
        state = self.right if hand == 1 else self.left
        state.update(line, layer, cut)
        return [{
            "_time": time,
            "_lineIndex": line,
            "_lineLayer": layer,
            "_type": hand,
            "_cutDirection": cut
        }]
