import random

class FlowState:
    def __init__(self, hand):
        self.hand = hand
        self.x = 1 if hand == 0 else 2
        self.y = 0
        self.cut = 1 # Começa cortando pra baixo

    def update(self, x, y, cut):
        self.x = x
        self.y = y
        self.cut = cut

class PatternManager:
    def __init__(self, difficulty="ExpertPlus"):
        self.left = FlowState(0)
        self.right = FlowState(1)
        self.parity = 1 # 1=Direita
        self.difficulty = difficulty
        
        # Definição de Padrões por Complexidade (tower_stack removido)
        self.patterns = {
            0: ['single_flow', 'stack_simple'],
            1: ['wide_flow', 'diagonal_cross', 'window_wipe', 'flow_inverted'],
            2: ['stream_burst', 'tech_angle', 'double_down', 'burst_fill', 'super_stream', 'diagonal_cross_inverted']
        }

    def get_pattern(self, intensity, complexity_idx, vertical_idx, time_gap, energy_level=0.5):
        """
        Seleciona padrão baseado nas 3 cabeças da IA, energia e posição anterior.
        """
        max_complexity = 2
        allow_bursts = True
        
        if self.difficulty == "Easy":
            max_complexity = 0
            allow_bursts = False
            if time_gap < 0.5: return None
        elif self.difficulty == "Normal":
            max_complexity = 1
            allow_bursts = False
            if time_gap < 0.3: return None
        elif self.difficulty == "Hard":
            max_complexity = 1
            if energy_level > 0.8: max_complexity = 2
            allow_bursts = False
            if time_gap < 0.2: return None
        elif self.difficulty == "Expert":
            max_complexity = 2
            allow_bursts = True

        if energy_level < 0.3:
            complexity_idx = 0
            if time_gap < 0.4 and self.difficulty not in ["Easy", "Normal"]: return None 

        if complexity_idx > max_complexity:
            complexity_idx = max_complexity

        min_gap = 0.12
        if energy_level > 0.7: min_gap = 0.08
        if energy_level < 0.4: min_gap = 0.25
        
        if self.difficulty == "Easy": min_gap = 0.5
        if self.difficulty == "Normal": min_gap = 0.3
        if self.difficulty == "Hard": min_gap = 0.2
        
        if time_gap < min_gap: return None
        
        if self.difficulty in ["Expert", "ExpertPlus"] and time_gap < 0.22 and energy_level > 0.5:
            return {'type': 'stream_burst', 'vert': vertical_idx}
            
        options = self.patterns.get(complexity_idx, self.patterns[1])
        
        if not allow_bursts:
            options = [p for p in options if p not in ['burst_fill', 'super_stream', 'stream_burst']]
            if not options: options = self.patterns[0]

        chosen_type = random.choice(options)

        # --- Lógica Anti-Wideness e Vision Block (Rígida) ---
        current_hand_state = self.right if self.parity == 1 else self.left
        last_line = current_hand_state.x

        # Se a mão já está fora, FORÇA o retorno ao centro.
        if last_line in [0, 3] and chosen_type in ['wide_flow', 'flow_inverted', 'diagonal_cross_inverted']:
            chosen_type = 'single_flow'

        # --- Lógica de Drop Agressiva (Expert+ e Picos) ---
        if self.difficulty == "ExpertPlus" and allow_bursts:
            if energy_level > 0.85 and random.random() < 0.5:
                return {'type': 'super_stream', 'vert': vertical_idx, 'intensity': intensity}
            if energy_level > 0.7:
                burst_prob = 0.4 + (energy_level - 0.7) * 2.0
                if random.random() < burst_prob:
                    return {'type': 'burst_fill', 'vert': vertical_idx, 'intensity': intensity}
                if random.random() < 0.35:
                    return {'type': 'double_down', 'vert': vertical_idx, 'intensity': intensity}
        elif allow_bursts:
            if energy_level > 0.85 and random.random() < 0.3:
                return {'type': 'super_stream', 'vert': vertical_idx, 'intensity': intensity}
            if energy_level > 0.65:
                burst_prob = 0.2 + (energy_level - 0.65) * 1.3
                if random.random() < burst_prob:
                    return {'type': 'burst_fill', 'vert': vertical_idx, 'intensity': intensity}
                if random.random() < 0.25:
                    return {'type': 'double_down', 'vert': vertical_idx, 'intensity': intensity}
        
        return {
            'type': chosen_type,
            'vert': vertical_idx,
            'intensity': intensity
        }

    def apply_pattern(self, meta, time, bpm):
        if not meta: return []
        
        ptype = meta['type']
        vert_bias = meta['vert']
        
        notes = []
        
        func_map = {
            'single_flow': self._gen_flow,
            'flow_inverted': self._gen_flow_inverted,
            'wide_flow': self._gen_wide,
            'stack_simple': self._gen_flow,
            'diagonal_cross': self._gen_diagonal_cross,
            'diagonal_cross_inverted': self._gen_diagonal_cross_inverted,
            'window_wipe': self._gen_wide,
            'stream_burst': lambda t, v: self._gen_flow(t, v, force_inversion=True),
            'double_down': self._gen_double,
            'tech_angle': self._gen_tech,
            'burst_fill': lambda t, v: self._gen_burst(t, bpm, v, length=4),
            'super_stream': lambda t, v: self._gen_burst(t, bpm, v, length=8)
        }
        
        gen_func = func_map.get(ptype, self._gen_flow)
        notes = gen_func(time, vert_bias)
            
        self.parity = 1 - self.parity
        return notes

    # --- Geradores ---
    
    def _gen_flow(self, time, v_bias, force_inversion=False):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        if state.y == 0: target_y = random.choice([1, 2])
        else: target_y = 0
        
        if v_bias == 2: target_y = max(1, target_y)
        if v_bias == 0: target_y = min(1, target_y)
        
        line = random.choice([0, 1]) if hand == 0 else random.choice([2, 3])
        
        # REGRA ANTI-VISION BLOCK: Se a nota for externa, limita a altura máxima.
        if line in [0, 3]:
            target_y = min(target_y, 1)

        if target_y > state.y: cut = 0 
        else: cut = 1 
        
        if random.random() > 0.7:
            if hand == 0: cut = 6 if cut == 1 else 4 
            else: cut = 7 if cut == 1 else 5 
            
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_flow_inverted(self, time, v_bias):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        if state.y == 0: target_y = random.choice([1, 2])
        else: target_y = 0
        
        if v_bias == 2: target_y = max(1, target_y)
        if v_bias == 0: target_y = min(1, target_y)
        
        # Crossover controlado: cruza para a faixa interna adjacente
        line = 2 if hand == 0 else 1
        
        if target_y > state.y: cut = 0 
        else: cut = 1 
        
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_wide(self, time, v_bias):
        hand = self.parity
        line = 0 if hand == 0 else 3
        # REGRA ANTI-VISION BLOCK: Limita a altura de notas amplas.
        layer = 0 if v_bias == 0 else 1
        cut = 1 
        if hand == 0: cut = 7
        else: cut = 6
        return self._create_note(hand, time, line, layer, cut)

    def _gen_diagonal_cross(self, time, v_bias):
        hand = self.parity
        line = 1 if hand == 0 else 2
        layer = v_bias
        cut = 0 if layer > 0 else 1
        return self._create_note(hand, time, line, layer, cut)

    def _gen_diagonal_cross_inverted(self, time, v_bias):
        hand = self.parity
        line = 2 if hand == 0 else 1 # Invertido
        layer = v_bias
        cut = 0 if layer > 0 else 1
        return self._create_note(hand, time, line, layer, cut)

    def _gen_double(self, time, v_bias):
        l_note = self._create_note(0, time, 1, 0, 1)[0]
        r_note = self._create_note(1, time, 2, 0, 1)[0]
        return [l_note, r_note]

    def _gen_tech(self, time, v_bias):
        hand = self.parity
        line = 1 if hand == 0 else 2
        layer = 1
        cut = 2 if hand == 0 else 3 
        return self._create_note(hand, time, line, layer, cut)

    def _gen_burst(self, time, bpm, v_bias, length=4):
        notes = []
        step = 0.25
        current_hand = self.parity
        
        for i in range(length):
            t = time + (i * step)
            line = 1 if current_hand == 0 else 2
            layer = 0
            cut = 1
            if i % 2 == 1: cut = 0
            
            note = {
                "_time": float(t),
                "_lineIndex": int(line),
                "_lineLayer": int(layer),
                "_type": int(current_hand),
                "_cutDirection": int(cut)
            }
            notes.append(note)
            
            state = self.right if current_hand == 1 else self.left
            state.update(line, layer, cut)
            current_hand = 1 - current_hand
            
        self.parity = current_hand
        return notes

    def _create_note(self, hand, time, line, layer, cut):
        state = self.right if hand == 1 else self.left
        state.update(line, layer, cut)
        return [{
            "_time": float(time),
            "_lineIndex": int(line),
            "_lineLayer": int(layer),
            "_type": int(hand),
            "_cutDirection": int(cut)
        }]
