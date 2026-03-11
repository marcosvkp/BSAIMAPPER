import random

class FlowState:
    """Mantém o controle da posição e direção de corte de cada mão."""

    def __init__(self, hand):
        self.hand = hand
        self.x = 1 if hand == 0 else 2
        self.y = 1
        self.cut = 0  # Começa cortando pra cima (UP)

    def update(self, x, y, cut):
        self.x, self.y, self.cut = x, y, cut


class PatternManager:
    """
    Traduz as intenções da IA em notas concretas.

    V3 - Melhorias:
    - Memória real de histórico (últimas N notas) para evitar repetição de posição
    - Alternância de mãos garantida por estado, não por modulo de tempo
    - Padrões especializados para star_level alto (jumps, crossovers, window)
    - Remoção da dependência de `time` como seed (causava loops)
    - _gen_double_note com paridade correta para cada mão
    """

    HISTORY_SIZE = 8  # Quantas notas recentes guardar para verificar repetição

    def __init__(self, difficulty="ExpertPlus"):
        self.left = FlowState(0)
        self.right = FlowState(1)
        self.parity = 1       # 0 = esquerda, 1 = direita
        self.difficulty = difficulty
        self.note_count = 0   # Contador global de notas geradas
        self.position_history = []  # Lista de (hand, line, layer) das últimas notas

        # Semente determinística baseada no estado, não no tempo
        self._rng = random.Random(42)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _record(self, hand, line, layer):
        """Registra uma nota no histórico de posições."""
        self.position_history.append((hand, line, layer))
        if len(self.position_history) > self.HISTORY_SIZE:
            self.position_history.pop(0)

    def _is_repeated_position(self, hand, line, layer, lookback=4):
        """Verifica se a posição (hand, line, layer) apareceu nas últimas N notas."""
        recent = self.position_history[-lookback:] if len(self.position_history) >= lookback else self.position_history
        return (hand, line, layer) in recent

    def _alternate_line(self, hand, preferred_line, intensity):
        """
        Retorna uma linha diferente da atual se a posição estiver repetindo muito.
        """
        state = self.right if hand == 1 else self.left
        if hand == 0:
            candidates = [0, 1, 2]
        else:
            candidates = [1, 2, 3]

        # Remove a posição atual para forçar variação
        if preferred_line in candidates and len(candidates) > 1:
            alternatives = [c for c in candidates if c != state.x]
            if self._is_repeated_position(hand, preferred_line, state.y, lookback=3):
                return self._rng.choice(alternatives)
        return preferred_line

    def _pick_layer(self, v_bias, hand, intensity):
        if v_bias == 0:
            base = [0, 0, 1]
        elif v_bias == 2:
            base = [1, 2, 2]
        else:
            base = [0, 1, 1, 2]
        return self._rng.choice(base)

    def _pick_line(self, hand, intensity, spread_bias):
        if hand == 0:
            pool = [0, 1] if spread_bias < 0.5 else [0, 1, 2]
        else:
            pool = [2, 3] if spread_bias < 0.5 else [1, 2, 3]
        return self._rng.choice(pool)

    def _cut_from_flow(self, state, layer, hand, allow_diagonal=True):
        """
        Determina a direção de corte baseada no fluxo atual (paridade vertical).
        Se a mão terminou em cima, corta pra baixo, e vice-versa.
        """
        if layer > state.y:
            cut = 0  # UP
        elif layer < state.y:
            cut = 1  # DOWN
        else:
            # Mesma altura: alterna entre UP e DOWN baseado no estado atual
            cut = 1 if state.cut in [0, 4, 5] else 0

        # Adiciona diagonais com probabilidade leve
        if allow_diagonal and self._rng.random() < 0.25:
            if cut == 0:
                cut = self._rng.choice([0, 4, 5])  # UP, UP_LEFT, UP_RIGHT
            else:
                cut = self._rng.choice([1, 6, 7])  # DOWN, DOWN_LEFT, DOWN_RIGHT

        return cut

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

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
        self.note_count += 1

        beat_dur = 60.0 / bpm
        is_stream = complexity_idx == 2 and time_gap < beat_dur * 0.45
        wants_double = complexity_idx >= 1 and intensity > 0.65
        is_high_star = star_level >= 6.5
        is_very_high_star = star_level >= 8.0

        # --- Seleção de padrão ---
        if is_stream:
            notes = self._gen_stream_note(time, vertical_idx)

        elif is_very_high_star and wants_double and self._rng.random() < 0.35:
            # Mapas 8★+: jumps amplos e padrões de window
            pattern_roll = self._rng.random()
            if pattern_roll < 0.4:
                notes = self._gen_jump(time, vertical_idx, intensity, wide=True)
            elif pattern_roll < 0.7:
                notes = self._gen_window_pattern(time, vertical_idx)
            else:
                notes = self._gen_double_note(time, vertical_idx, intensity)

        elif is_high_star and wants_double and self._rng.random() < 0.30:
            # Mapas 6.5★–8★: jumps moderados
            if self._rng.random() < 0.5:
                notes = self._gen_jump(time, vertical_idx, intensity, wide=False)
            else:
                notes = self._gen_double_note(time, vertical_idx, intensity)

        elif complexity_idx == 0:
            notes = self._gen_single_note(time, vertical_idx, intensity)

        elif complexity_idx == 1:
            if self._rng.random() < 0.25:
                notes = self._gen_wide_note(time, vertical_idx)
            else:
                notes = self._gen_single_note(time, vertical_idx, intensity)

        else:
            # complexity_idx == 2
            if self._rng.random() < 0.15:
                notes = self._gen_tower_stack(time)
            elif is_high_star and self._rng.random() < 0.20:
                notes = self._gen_jump(time, vertical_idx, intensity, wide=False)
            else:
                notes = self._gen_single_note(time, vertical_idx, intensity)

        # Alterna a mão dominante apenas em notas simples
        if len(notes) == 1:
            self.parity = 1 - self.parity

        return notes

    # ------------------------------------------------------------------
    # Geração de notas
    # ------------------------------------------------------------------

    def _create_note(self, hand, time, line, layer, cut):
        state = self.right if hand == 1 else self.left
        state.update(line, layer, cut)
        self._record(hand, line, layer)
        return [{
            "_time": time,
            "_lineIndex": line,
            "_lineLayer": layer,
            "_type": hand,
            "_cutDirection": cut,
        }]

    def _gen_single_note(self, time, v_bias, intensity):
        hand = self.parity
        state = self.right if hand == 1 else self.left

        line = self._pick_line(hand, intensity, spread_bias=intensity)
        line = self._alternate_line(hand, line, intensity)
        layer = self._pick_layer(v_bias, hand, intensity)
        cut = self._cut_from_flow(state, layer, hand)

        return self._create_note(hand, time, line, layer, cut)

    def _gen_stream_note(self, time, v_bias):
        """Nota de stream: posição central, alterna cima/baixo, evita bordas."""
        hand = self.parity
        state = self.right if hand == 1 else self.left

        layer = 1 if v_bias in [1, 2] else 0

        # Em streams, ficamos na faixa central (cols 1-2) para ser jogável
        if hand == 0:
            pool = [1, 2] if state.x == 0 else [0, 1]
        else:
            pool = [1, 2] if state.x == 3 else [2, 3]

        # Evitar repetir a mesma coluna do stream anterior
        line = self._rng.choice(pool)
        if self._is_repeated_position(hand, line, layer, lookback=2):
            line = pool[0] if line == pool[-1] else pool[-1]

        cut = 1 if state.cut in [0, 4, 5] else 0  # Alterna rigorosamente em streams
        return self._create_note(hand, time, line, layer, cut)

    def _gen_wide_note(self, time, v_bias):
        """Nota na borda exterior (col 0 para esq, col 3 para dir)."""
        hand = self.parity
        line = 0 if hand == 0 else 3
        layer = 0 if v_bias == 0 else 1 if v_bias == 1 else 2
        # Cortes diagonais para fora ficam naturais nas bordas
        cut = 6 if hand == 0 else 7  # DOWN_LEFT / DOWN_RIGHT
        return self._create_note(hand, time, line, layer, cut)

    def _gen_double_note(self, time, v_bias, intensity):
        """
        Nota dupla (ambas as mãos ao mesmo tempo).
        Corrige: cada mão usa a direção correta de acordo com sua paridade.
        """
        if intensity > 0.78:
            l_line, r_line = 0, 3
        else:
            l_line, r_line = 1, 2

        layer = 0 if v_bias == 0 else 1 if v_bias == 1 else 2

        # Direção de corte consistente com o estado de cada mão
        l_cut = self._cut_from_flow(self.left, layer, 0, allow_diagonal=False)
        r_cut = self._cut_from_flow(self.right, layer, 1, allow_diagonal=False)

        l_note = self._create_note(0, time, l_line, layer, l_cut)[0]
        r_note = self._create_note(1, time, r_line, layer, r_cut)[0]
        return [l_note, r_note]

    def _gen_jump(self, time, v_bias, intensity, wide=False):
        """
        Jump: notas em posições opostas/distantes entre as mãos.
        Muito comum em mapas de alta dificuldade.
        """
        if wide:
            l_line, r_line = 0, 3
        else:
            l_line = self._rng.choice([0, 1])
            r_line = self._rng.choice([2, 3])

        # Camadas opostas para criar movimento interessante
        if v_bias == 1:
            l_layer = self._rng.choice([0, 2])
            r_layer = 2 if l_layer == 0 else 0
        else:
            l_layer = 0 if v_bias == 0 else 2
            r_layer = l_layer

        l_cut = self._cut_from_flow(self.left, l_layer, 0, allow_diagonal=True)
        r_cut = self._cut_from_flow(self.right, r_layer, 1, allow_diagonal=True)

        l_note = self._create_note(0, time, l_line, l_layer, l_cut)[0]
        r_note = self._create_note(1, time, r_line, r_layer, r_cut)[0]
        return [l_note, r_note]

    def _gen_window_pattern(self, time, v_bias):
        """
        Window pattern: notas em posições que formam uma 'janela' entre as mãos.
        Comum em ExpertPlus de alta estrela — exige esquivar o sabre de uma mão
        enquanto a outra corta.
        Exemplo: esquerda col 1 layer 0, direita col 2 layer 2
        """
        combos = [
            (1, 0, 2, 2),  # Esq baixo, Dir cima
            (1, 2, 2, 0),  # Esq cima, Dir baixo
            (0, 1, 3, 1),  # Esq centro-esq, Dir centro-dir
            (0, 0, 3, 2),  # Esq baixo-esq, Dir cima-dir
        ]
        l_line, l_layer, r_line, r_layer = self._rng.choice(combos)

        l_cut = self._cut_from_flow(self.left, l_layer, 0, allow_diagonal=True)
        r_cut = self._cut_from_flow(self.right, r_layer, 1, allow_diagonal=True)

        l_note = self._create_note(0, time, l_line, l_layer, l_cut)[0]
        r_note = self._create_note(1, time, r_line, r_layer, r_cut)[0]
        return [l_note, r_note]

    def _gen_tower_stack(self, time):
        """
        Tower stack: duas notas da mesma mão em camadas diferentes.
        Usado com moderação (complexidade alta).
        """
        hand = self.parity
        state = self.right if hand == 1 else self.left

        # Evitar repetir a coluna do stack anterior
        line = 1 if hand == 0 else 2
        if self._is_repeated_position(hand, line, 0, lookback=3):
            line = 0 if hand == 0 else 3

        low = self._create_note(hand, time, line, 0, 0)[0]   # Layer 0, UP
        high = self._create_note(hand, time, line, 2, 1)[0]  # Layer 2, DOWN
        return [low, high]