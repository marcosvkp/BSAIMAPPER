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
    def __init__(self):
        self.left = FlowState(0)
        self.right = FlowState(1)
        self.parity = 1 # 1=Direita
        
        # Definição de Padrões por Complexidade
        self.patterns = {
            # 0: Chill (Baixa energia)
            0: ['single_flow', 'stack_simple'],
            # 1: Dance (Média energia)
            1: ['wide_flow', 'diagonal_cross', 'window_wipe'],
            # 2: Tech/Stream (Alta energia)
            2: ['stream_burst', 'tech_angle', 'double_down', 'tower_stack', 'burst_fill', 'super_stream']
        }

    def get_pattern(self, intensity, complexity_idx, vertical_idx, time_gap, energy_level=0.5):
        """
        Seleciona padrão baseado nas 3 cabeças da IA e na ENERGIA GLOBAL.
        """
        # --- 1. Filtro de Energia (Global) ---
        if energy_level < 0.3:
            complexity_idx = 0
            if time_gap < 0.4: return None 

        if energy_level > 0.8:
            complexity_idx = 2 # Força Tech/Stream
            
        # --- 2. Cooldown Dinâmico ---
        min_gap = 0.12
        if energy_level > 0.7: min_gap = 0.08
        if energy_level < 0.4: min_gap = 0.25
        
        if time_gap < min_gap: return None
        
        # Se o gap for muito pequeno, força stream independente da IA
        if time_gap < 0.22 and energy_level > 0.5:
            return {'type': 'stream_burst', 'vert': vertical_idx}
            
        options = self.patterns.get(complexity_idx, self.patterns[1])
        
        # --- Lógica de Drop Agressiva ---
        
        # Se a energia for muito alta (> 0.85), chance de SUPER STREAM (8 notas)
        if energy_level > 0.85:
            if random.random() < 0.3: # 30% de chance de iniciar um stream longo
                return {'type': 'super_stream', 'vert': vertical_idx, 'intensity': intensity}
        
        # Se a energia for alta (> 0.65), chance de BURST FILL (4 notas)
        # Reduzido de 0.85 para 0.65 para pegar mais drops
        if energy_level > 0.65:
            # Probabilidade escala com a energia: 0.65 -> 20%, 0.95 -> 60%
            burst_prob = 0.2 + (energy_level - 0.65) * 1.3
            if random.random() < burst_prob:
                return {'type': 'burst_fill', 'vert': vertical_idx, 'intensity': intensity}
            
            # Chance de Double Down também aumenta
            if random.random() < 0.25:
                return {'type': 'double_down', 'vert': vertical_idx, 'intensity': intensity}

        chosen_type = random.choice(options)
        
        return {
            'type': chosen_type,
            'vert': vertical_idx, # 0=Baixo, 1=Meio, 2=Cima
            'intensity': intensity
        }

    def apply_pattern(self, meta, time, bpm):
        if not meta: return []
        
        ptype = meta['type']
        vert_bias = meta['vert']
        
        notes = []
        
        if ptype == 'single_flow':
            notes = self._gen_flow(time, vert_bias)
            
        elif ptype == 'wide_flow':
            notes = self._gen_wide(time, vert_bias)
            
        elif ptype == 'stream_burst':
            notes = self._gen_flow(time, vert_bias, force_inversion=True)
            
        elif ptype == 'double_down':
            notes = self._gen_double(time, vert_bias)
            
        elif ptype == 'tech_angle':
            notes = self._gen_tech(time, vert_bias)
            
        elif ptype == 'tower_stack':
            notes = self._gen_tower(time, vert_bias)
            
        elif ptype == 'burst_fill':
            # Gera 4 notas em 1/4 de beat (Stream curto)
            notes = self._gen_burst(time, bpm, vert_bias, length=4)
            
        elif ptype == 'super_stream':
            # Gera 8 notas em 1/4 de beat (Stream longo)
            notes = self._gen_burst(time, bpm, vert_bias, length=8)
            
        else:
            notes = self._gen_flow(time, vert_bias)
            
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
        
        if target_y > state.y: cut = 0 
        else: cut = 1 
        
        if random.random() > 0.7:
            if hand == 0: cut = 6 if cut == 1 else 4 
            else: cut = 7 if cut == 1 else 5 
            
        line = random.choice([0, 1]) if hand == 0 else random.choice([2, 3])
        
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_wide(self, time, v_bias):
        hand = self.parity
        line = 0 if hand == 0 else 3
        layer = 0 if v_bias == 0 else 1
        cut = 1 
        if hand == 0: cut = 7
        else: cut = 6
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

    def _gen_tower(self, time, v_bias):
        hand = self.parity
        line = 1 if hand == 0 else 2
        n1 = self._create_note(hand, time, line, 0, 1)[0]
        n2 = self._create_note(hand, time, line, 2, 1)[0]
        return [n1, n2]

    def _gen_burst(self, time, bpm, v_bias, length=4):
        # Gera uma sequência de notas rápidas (1/4 de beat)
        notes = []
        step = 0.25 # 1/4 de beat
        
        current_hand = self.parity
        
        for i in range(length):
            t = time + (i * step)
            
            # Simula lógica de flow simplificada para o burst
            line = 1 if current_hand == 0 else 2
            layer = 0
            cut = 1 # Baixo
            
            # Inverte direção a cada nota
            if i % 2 == 1: cut = 0 # Cima
            
            note = {
                "_time": t,
                "_lineIndex": line,
                "_lineLayer": layer,
                "_type": current_hand,
                "_cutDirection": cut
            }
            notes.append(note)
            
            state = self.right if current_hand == 1 else self.left
            state.update(line, layer, cut)
            
            current_hand = 1 - current_hand # Troca mão
            
        self.parity = current_hand
        return notes

    def _create_note(self, hand, time, line, layer, cut):
        state = self.right if hand == 1 else self.left
        state.update(line, layer, cut)
        return [{
            "_time": time,
            "_lineIndex": line,
            "_lineLayer": layer,
            "_type": hand,
            "_cutDirection": cut
        }]
