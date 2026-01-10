import random
import math

class FlowState:
    def __init__(self, hand):
        self.hand = hand
        self.x = 1 if hand == 0 else 2
        self.y = 0
        self.cut = 1
        self.last_beat_time = -100
        self.last_large_v_jump_beat = -100 # Cooldown para saltos verticais grandes

    def update(self, x, y, cut, beat_time):
        self.x = x
        self.y = y
        self.cut = cut
        self.last_beat_time = beat_time

class PatternManager:
    def __init__(self, difficulty="ExpertPlus"):
        self.left = FlowState(0)
        self.right = FlowState(1)
        self.parity = 1
        self.difficulty = difficulty
        self.patterns = {
            0: ['single_flow', 'stack_simple'],
            1: ['wide_flow', 'diagonal_cross', 'window_wipe', 'flow_inverted'],
            2: ['stream_burst', 'tech_angle', 'double_down', 'burst_fill', 'super_stream', 'diagonal_cross_inverted']
        }

    def get_pattern(self, intensity, complexity_idx, vertical_idx, angle_idx, time_gap, energy_level=0.5):
        max_complexity = 2
        allow_bursts = True
        
        # Ajustes de dificuldade
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
            return {'type': 'stream_burst', 'vert': vertical_idx, 'angle': angle_idx}
            
        options = self.patterns.get(complexity_idx, self.patterns[1])
        if not allow_bursts:
            options = [p for p in options if p not in ['burst_fill', 'super_stream', 'stream_burst']]
            if not options: options = self.patterns[0]

        chosen_type = random.choice(options)
        
        return {'type': chosen_type, 'vert': vertical_idx, 'angle': angle_idx, 'intensity': intensity}

    def apply_pattern(self, meta, time, bpm):
        if not meta: return []
        
        ptype = meta['type']
        vert_bias = meta['vert']
        angle_bias = meta.get('angle', 8) # 8 = Any/Dot
        
        notes = []
        
        func_map = {
            'single_flow': self._gen_flow, 'flow_inverted': self._gen_flow_inverted,
            'wide_flow': self._gen_wide, 'stack_simple': self._gen_flow,
            'diagonal_cross': self._gen_diagonal_cross, 'diagonal_cross_inverted': self._gen_diagonal_cross_inverted,
            'window_wipe': self._gen_wide, 'stream_burst': self._gen_flow,
            'double_down': self._gen_double, 'tech_angle': self._gen_tech,
        }
        
        if ptype in func_map:
            notes = func_map[ptype](time, vert_bias, angle_bias)
        elif ptype == 'burst_fill':
            notes = self._gen_burst(time, bpm, vert_bias, length=4)
        elif ptype == 'super_stream':
            notes = self._gen_burst(time, bpm, vert_bias, length=8)
        else:
            notes = self._gen_flow(time, vert_bias, angle_bias)
            
        self.parity = 1 - self.parity
        return notes

    def _get_comfortable_y(self, potential_y, state, time):
        VERTICAL_JUMP_COOLDOWN_BEATS = 0.5
        is_large_v_jump = (state.y == 0 and potential_y == 2) or \
                          (state.y == 2 and potential_y == 0)
        
        if is_large_v_jump:
            if time - state.last_large_v_jump_beat < VERTICAL_JUMP_COOLDOWN_BEATS:
                return 1
            else:
                state.last_large_v_jump_beat = time
        return potential_y

    def _apply_safety_rules(self, potential_x, potential_y, state, time):
        target_y = self._get_comfortable_y(potential_y, state, time)

        if state.y == 1 and target_y == 0:
            if potential_x > state.x + 1:
                potential_x = state.x + 1
            elif potential_x < state.x - 1:
                potential_x = state.x - 1
        
        line = max(0, min(3, int(potential_x)))

        # --- ALTERAÇÃO: Permitir Top Layer (y=2) nas laterais em dificuldades altas ---
        # Antes: if line in [0, 3]: target_y = min(target_y, 1)
        # Agora: Só restringe se for Easy/Normal. Hard+ libera o teto.
        if self.difficulty in ["Easy", "Normal"]:
            if line in [0, 3]:
                target_y = min(target_y, 1)
        
        # Regra de corte para cima: Se vou cortar pra cima, idealmente a nota deve estar acima da anterior
        is_up_cut_intent = target_y > state.y
        if target_y == 2 and is_up_cut_intent:
            # Se já estou no topo, não posso subir mais, então mantenho ou desço pra 1 se for muito estranho
            pass 

        return line, target_y

    def _gen_flow(self, time, v_bias, angle_bias):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        # --- ALTERAÇÃO: Lógica de Verticalidade Melhorada ---
        # Antes: Só subia se estivesse no chão (0).
        # Agora: Pode transitar 0->1, 1->2, 2->1, etc.
        
        if state.y == 0:
            potential_y = random.choice([0, 1, 2]) # Pode ficar no chão ou subir
        elif state.y == 1:
            potential_y = random.choice([0, 2]) # Meio -> Extremos
        else: # state.y == 2
            potential_y = random.choice([0, 1]) # Topo -> Desce
            
        # Bias da Rede Neural (DirectorNet)
        if v_bias == 2: potential_y = max(1, potential_y) # Força pra cima
        if v_bias == 0: potential_y = min(1, potential_y) # Força pra baixo
        
        potential_x = random.choice([0, 1]) if hand == 0 else random.choice([2, 3])
        
        line, target_y = self._apply_safety_rules(potential_x, potential_y, state, time)

        # Usa angle_bias se for válido (0-8), senão usa lógica padrão
        if angle_bias < 8:
            cut = angle_bias
        else:
            cut = 0 if target_y > state.y else 1
            
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_flow_inverted(self, time, v_bias, angle_bias):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        potential_y = random.choice([1, 2]) if state.y == 0 else 0
        if v_bias == 2: potential_y = max(1, potential_y)
        if v_bias == 0: potential_y = min(1, potential_y)
        
        potential_x = 2 if hand == 0 else 1
        
        line, target_y = self._apply_safety_rules(potential_x, potential_y, state, time)
        
        cut = 0 if target_y > state.y else 1
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_wide(self, time, v_bias, angle_bias):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        potential_y = 0 if v_bias == 0 else 1
        if v_bias == 2: potential_y = 2 # Permite wide alto
        
        potential_x = 0 if hand == 0 else 3
        
        line, target_y = self._apply_safety_rules(potential_x, potential_y, state, time)
        
        cut = 1 
        if hand == 0: cut = 7
        else: cut = 6
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_diagonal_cross(self, time, v_bias, angle_bias):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        potential_x = 1 if hand == 0 else 2
        line, target_y = self._apply_safety_rules(potential_x, v_bias, state, time)
        
        cut = 0 if target_y > 0 else 1
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_diagonal_cross_inverted(self, time, v_bias, angle_bias):
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        potential_x = 2 if hand == 0 else 1
        line, target_y = self._apply_safety_rules(potential_x, v_bias, state, time)
        
        cut = 0 if target_y > 0 else 1
        return self._create_note(hand, time, line, target_y, cut)

    def _gen_double(self, time, v_bias, angle_bias):
        y = 0
        if v_bias == 2: y = 1
        l_note = self._create_note(0, time, 1, y, 1)[0]
        r_note = self._create_note(1, time, 2, y, 1)[0]
        return [l_note, r_note]

    def _gen_tech(self, time, v_bias, angle_bias):
        hand = self.parity
        line = 1 if hand == 0 else 2
        layer = 1
        if v_bias == 2: layer = 2
        if v_bias == 0: layer = 0

        cut = angle_bias if angle_bias < 8 else (2 if hand == 0 else 3)
        return self._create_note(hand, time, line, layer, cut)

    def _gen_burst(self, time, bpm, v_bias, length=4):
        notes = []
        step = 0.25
        current_hand = self.parity
        for i in range(length):
            t = time + (i * step)
            line = 1 if current_hand == 0 else 2
            layer = 0
            cut = 1 if i % 2 == 0 else 0
            note = self._create_note(current_hand, t, line, layer, cut)[0]
            notes.append(note)
            current_hand = 1 - current_hand
        self.parity = current_hand
        return notes

    def _create_note(self, hand, time, line, layer, cut):
        state = self.right if hand == 1 else self.left
        state.update(line, layer, cut, time)
        return [{
            "_time": float(time), "_lineIndex": int(line), "_lineLayer": int(layer),
            "_type": int(hand), "_cutDirection": int(cut)
        }]
