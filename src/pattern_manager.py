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
            2: ['stream_burst', 'tech_angle', 'double_down']
        }

    def get_pattern(self, intensity, complexity_idx, vertical_idx, time_gap):
        """
        Seleciona padrão baseado nas 3 cabeças da IA.
        """
        # Cooldown forçado para evitar spam impossível
        if time_gap < 0.12: return None
        
        # Se o gap for muito pequeno, força stream independente da IA
        if time_gap < 0.22:
            return {'type': 'stream_burst', 'vert': vertical_idx}
            
        # Escolhe padrão baseado na complexidade predita pela IA
        options = self.patterns.get(complexity_idx, self.patterns[1])
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
            
        else:
            notes = self._gen_flow(time, vert_bias)
            
        self.parity = 1 - self.parity
        return notes

    # --- Geradores ---
    
    def _gen_flow(self, time, v_bias, force_inversion=False):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        # Lógica de Flow: Se estava em baixo, vai pra cima
        if state.y == 0: target_y = random.choice([1, 2])
        else: target_y = 0
        
        # Bias da IA: Se a IA quer "Cima" (2), forçamos layer mais alta
        if v_bias == 2: target_y = max(1, target_y)
        if v_bias == 0: target_y = min(1, target_y)
        
        # Direção do corte
        if target_y > state.y: cut = 0 # Cima
        else: cut = 1 # Baixo
        
        # Variação angular (Tech)
        if random.random() > 0.7:
            if hand == 0: cut = 6 if cut == 1 else 4 # Diagonais Esq
            else: cut = 7 if cut == 1 else 5 # Diagonais Dir
            
        line = random.choice([0, 1]) if hand == 0 else random.choice([2, 3])
        
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_wide(self, time, v_bias):
        # Padrão amplo: usa as pontas (0 e 3)
        hand = self.parity
        line = 0 if hand == 0 else 3
        layer = 0 if v_bias == 0 else 1
        cut = 1 # Geralmente corte pra baixo em wide jumps
        
        # Adiciona diagonal para dentro
        if hand == 0: cut = 7
        else: cut = 6
        
        return self._create_note(hand, time, line, layer, cut)

    def _gen_double(self, time, v_bias):
        # Duas notas ao mesmo tempo
        l_note = self._create_note(0, time, 1, 0, 1)[0]
        r_note = self._create_note(1, time, 2, 0, 1)[0]
        return [l_note, r_note]

    def _gen_tech(self, time, v_bias):
        # Notas com ângulos estranhos (Dot notes ou laterais)
        hand = self.parity
        line = 1 if hand == 0 else 2
        layer = 1
        cut = 2 if hand == 0 else 3 # Corte lateral
        return self._create_note(hand, time, line, layer, cut)

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
