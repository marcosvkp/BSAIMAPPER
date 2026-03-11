class FlowState:
    """Mantém o controle da posição e direção de corte de cada mão."""

    def __init__(self, hand):
        self.hand = hand
        self.x = 1 if hand == 0 else 2
        self.y = 1
        self.cut = 1

    def update(self, x, y, cut):
        self.x, self.y, self.cut = x, y, cut


class PatternManager:
    """
    Traduz as intenções da IA em notas concretas.
    A diversidade é guiada por contexto (tempo + estado), evitando aleatoriedade pura.
    """

    def __init__(self, difficulty="ExpertPlus"):
        self.left = FlowState(0)
        self.right = FlowState(1)
        self.parity = 1
        self.difficulty = difficulty

    def _context_index(self, modulo, *values):
        seed = 0
        for i, val in enumerate(values, 1):
            seed += int(abs(val) * 1000) * (i * 31)
        return seed % modulo

    def apply_pattern(
        self,
        time,
        bpm,
        complexity_idx,
        vertical_idx,
        time_gap,
        intensity=0.5,
        star_level=6.0,
    ):
        notes = []

        is_stream = complexity_idx == 2 and time_gap < (60 / bpm) * 0.45
        wants_double = complexity_idx >= 1 and intensity > 0.6
        high_density = (star_level >= 6.5 and intensity > 0.7) or complexity_idx == 2

        if is_stream:
            notes = self._gen_stream_note(time, vertical_idx)
        elif high_density and wants_double and self._context_index(100, time, intensity) < 35:
            notes = self._gen_double_note(time, vertical_idx, intensity)
        elif complexity_idx == 0:
            notes = self._gen_single_note(time, vertical_idx, intensity)
        elif complexity_idx == 1:
            if self._context_index(10, time, vertical_idx, intensity) < 3:
                notes = self._gen_wide_note(time, vertical_idx)
            else:
                notes = self._gen_single_note(time, vertical_idx, intensity)
        else:
            if self._context_index(10, time, time_gap, intensity) < 2:
                notes = self._gen_tower_stack(time)
            else:
                notes = self._gen_single_note(time, vertical_idx, intensity)

        if len(notes) == 1:
            self.parity = 1 - self.parity

        return notes

    def _create_note(self, hand, time, line, layer, cut):
        state = self.right if hand == 1 else self.left
        state.update(line, layer, cut)
        return [{
            "_time": time,
            "_lineIndex": line,
            "_lineLayer": layer,
            "_type": hand,
            "_cutDirection": cut,
        }]

    def _pick_layer(self, v_bias, hand, intensity, time):
        if v_bias == 0:
            base = [0, 0, 1]
        elif v_bias == 2:
            base = [1, 2, 2]
        else:
            base = [0, 1, 1, 2]

        idx = self._context_index(len(base), time, intensity, hand, v_bias)
        return base[idx]

    def _pick_line(self, hand, intensity, time, spread_bias):
        if hand == 0:
            pool = [0, 1] if spread_bias < 0.5 else [0, 1, 2]
        else:
            pool = [2, 3] if spread_bias < 0.5 else [1, 2, 3]

        idx = self._context_index(len(pool), time, intensity, spread_bias)
        return pool[idx]

    def _gen_single_note(self, time, v_bias, intensity):
        hand = self.parity
        line = self._pick_line(hand, intensity, time, spread_bias=intensity)
        layer = self._pick_layer(v_bias, hand, intensity, time)

        state = self.right if hand == 1 else self.left
        if layer > state.y:
            cut = 0
        elif layer < state.y:
            cut = 1
        else:
            cut = 0 if self._context_index(2, time, state.cut, hand) == 0 else 1

        if self._context_index(10, time, intensity, hand, v_bias) > 7:
            cut = 4 if hand == 0 else 5
        return self._create_note(hand, time, line, layer, cut)

    def _gen_stream_note(self, time, v_bias):
        hand = self.parity
        state = self.right if hand == 1 else self.left

        if v_bias == 0:
            layer = 0
        elif v_bias == 2:
            layer = 1
        else:
            layer = 1

        line_pool = [1, 2] if state.x in [0, 3] else [state.x, 1 if hand == 0 else 2]
        line = line_pool[self._context_index(len(line_pool), time, state.x, state.y)]
        cut = 1 - state.cut if state.cut in [0, 1] else 1
        return self._create_note(hand, time, line, layer, cut)

    def _gen_wide_note(self, time, v_bias):
        hand = self.parity
        line = 0 if hand == 0 else 3
        layer = 0 if v_bias == 0 else 1 if v_bias == 1 else 2
        cut = 6 if hand == 0 else 7
        return self._create_note(hand, time, line, layer, cut)

    def _gen_double_note(self, time, v_bias, intensity):
        if intensity > 0.78:
            l_line, r_line = 0, 3
        else:
            l_line, r_line = 1, 2

        layer = 0 if v_bias == 0 else 1 if v_bias == 1 else 2
        l_note = self._create_note(0, time, l_line, layer, 1)[0]
        r_note = self._create_note(1, time, r_line, layer, 1)[0]
        return [l_note, r_note]

    def _gen_tower_stack(self, time):
        hand = self.parity
        line = 1 if hand == 0 else 2
        low = self._create_note(hand, time, line, 0, 0)[0]
        high = self._create_note(hand, time, line, 2, 0)[0]
        return [low, high]
