import copy
import random

class FlowFixer:
    """
    Validador físico V13.

    Ordem de operações:
      1. Remove tower stacks (mesma mão, mesmo timestamp)
      2. Remove notas inatingíveis (distância física vs tempo)
      3. _fix_swing: alternância + consistência de layer
      4. _fix_edge: bordas da grade (roda depois do swing)
      5. _fix_crossed_arms: braços cruzados em simultâneas
      6. _fix_hitbox: doubles sobrepostos
    """

    UP         = 0
    DOWN       = 1
    LEFT       = 2
    RIGHT      = 3
    UP_LEFT    = 4
    UP_RIGHT   = 5
    DOWN_LEFT  = 6
    DOWN_RIGHT = 7
    DOT        = 8

    LEFT_HAND  = 0
    RIGHT_HAND = 1
    BOMB       = 3

    MAX_LINE_JUMP_DIST  = 2
    MIN_BEATS_WIDE_JUMP = 0.4

    # Componente vertical de cada direção
    # 'up'   = braço subindo  (corta de baixo pra cima)
    # 'down' = braço descendo (corta de cima pra baixo)
    # None   = puramente horizontal
    VERT = {
        0: 'up',    # UP
        1: 'down',  # DOWN
        2: None,    # LEFT
        3: None,    # RIGHT
        4: 'up',    # UP_LEFT
        5: 'up',    # UP_RIGHT
        6: 'down',  # DOWN_LEFT
        7: 'down',  # DOWN_RIGHT
    }

    # Após subir → deve descer. Após descer → deve subir.
    VALID_AFTER = {
        'up':   {1, 6, 7},   # DOWN, DOWN_LEFT, DOWN_RIGHT
        'down': {0, 4, 5},   # UP,   UP_LEFT,   UP_RIGHT
    }

    OPPOSITE = {
        0: 1, 1: 0,   # UP ↔ DOWN
        2: 3, 3: 2,   # LEFT ↔ RIGHT
        4: 7, 7: 4,   # UP_LEFT ↔ DOWN_RIGHT
        5: 6, 6: 5,   # UP_RIGHT ↔ DOWN_LEFT
    }

    # Cuts proibidos por posição na grade
    # Calculado dinamicamente em _forbidden_for_pos
    # col 0: sem LEFT, UP_LEFT, DOWN_LEFT
    # col 3: sem RIGHT, UP_RIGHT, DOWN_RIGHT
    # layer 0: sem DOWN, DOWN_LEFT, DOWN_RIGHT
    # layer 2: sem UP, UP_LEFT, UP_RIGHT
    _LEFT_FORBIDDEN   = {2, 4, 6}
    _RIGHT_FORBIDDEN  = {3, 5, 7}
    _BOTTOM_FORBIDDEN = {1, 6, 7}
    _TOP_FORBIDDEN    = {0, 4, 5}

    @staticmethod
    def _forbidden_for_pos(col, layer):
        f = set()
        if col   == 0: f |= FlowFixer._LEFT_FORBIDDEN
        if col   == 3: f |= FlowFixer._RIGHT_FORBIDDEN
        if layer == 0: f |= FlowFixer._BOTTOM_FORBIDDEN
        if layer == 2: f |= FlowFixer._TOP_FORBIDDEN
        return f

    # ─────────────────────────────────────────────────────────────
    # Interface pública
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def fix(notes, bpm):
        if not notes:
            return []

        stats = {
            "same_hand": 0, "edge": 0, "unreachable": 0,
            "swing": 0, "crossed": 0, "hitbox": 0,
        }

        notes = [copy.copy(n) for n in notes]
        notes.sort(key=lambda x: x['_time'])

        left  = [n for n in notes if n['_type'] == FlowFixer.LEFT_HAND]
        right = [n for n in notes if n['_type'] == FlowFixer.RIGHT_HAND]
        bombs = [n for n in notes if n['_type'] == FlowFixer.BOMB]

        left  = FlowFixer._remove_simultaneous(left,  stats)
        right = FlowFixer._remove_simultaneous(right, stats)

        left  = FlowFixer._remove_unreachable(left,  stats)
        right = FlowFixer._remove_unreachable(right, stats)

        left  = FlowFixer._fix_swing(left,  stats)
        right = FlowFixer._fix_swing(right, stats)

        # Borda roda APÓS swing para não destruir intenção do modelo
        for n in left + right:
            FlowFixer._fix_edge(n, stats)

        all_notes = sorted(left + right, key=lambda x: x['_time'])
        all_notes = FlowFixer._fix_crossed_arms(all_notes, stats)
        all_notes = FlowFixer._fix_hitbox(all_notes, stats)

        result = sorted(all_notes + bombs, key=lambda x: x['_time'])

        total = sum(stats.values())
        if total > 0:
            print(
                f"      [FlowFixer V13] "
                f"same_hand={stats['same_hand']} | edge={stats['edge']} | "
                f"unreachable={stats['unreachable']} | swing={stats['swing']} | "
                f"crossed={stats['crossed']} | hitbox={stats['hitbox']}"
            )
        return result

    # ─────────────────────────────────────────────────────────────
    # 1. Tower stacks
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _remove_simultaneous(hand_notes, stats):
        if len(hand_notes) < 2:
            return hand_notes
        result, i = [], 0
        while i < len(hand_notes):
            group = [hand_notes[i]]
            j = i + 1
            while j < len(hand_notes) and abs(hand_notes[j]['_time'] - hand_notes[i]['_time']) < 0.016:
                group.append(hand_notes[j])
                j += 1
            if len(group) > 1:
                result.append(min(group, key=lambda n: n['_lineLayer']))
                stats['same_hand'] += len(group) - 1
            else:
                result.append(group[0])
            i = j
        return result

    # ─────────────────────────────────────────────────────────────
    # 2. Atingibilidade
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _remove_unreachable(hand_notes, stats):
        if not hand_notes:
            return []
        result, prev_line = [], None
        for note in hand_notes:
            if prev_line is not None and result:
                time_beats = note['_time'] - result[-1]['_time']
                line_dist  = abs(note['_lineIndex'] - prev_line)
                if line_dist > FlowFixer.MAX_LINE_JUMP_DIST and time_beats < FlowFixer.MIN_BEATS_WIDE_JUMP:
                    stats['unreachable'] += 1
                    continue
            result.append(note)
            prev_line = note['_lineIndex']
        return result

    # ─────────────────────────────────────────────────────────────
    # 3. Swing + consistência de layer
    #
    # Regra de swing: após componente 'up' → deve descer; após 'down' → deve subir.
    # Criativo: aceita DOWN, DOWN_LEFT, DOWN_RIGHT após UP (não força DOWN puro).
    #
    # Regra de layer (o problema da imagem):
    #   Se a nota atual está numa layer MAIS ALTA que a anterior (dlayer > 0),
    #   o cut não pode ter componente 'up' — o braço chegou subindo nessa posição,
    #   não faz sentido cortar para cima novamente em seguida sem descer antes.
    #   Analogamente, layer mais baixa (dlayer < 0) não pode ter comp 'down'.
    #
    # DOT: consome o swing sem alterar direção.
    #   UP → DOT → UP ✅  (DOT desceu, próxima sobe)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_swing(hand_notes, stats):
        if len(hand_notes) < 2:
            return hand_notes

        result     = list(hand_notes)
        prev_cut   = None
        prev_layer = None

        for note in result:
            cut   = note['_cutDirection']
            layer = note['_lineLayer']
            col   = note['_lineIndex']

            # DOT: avança prev_cut sem corrigir a nota
            if cut == FlowFixer.DOT:
                if prev_cut is not None:
                    prev_cut = FlowFixer.OPPOSITE.get(prev_cut, prev_cut)
                prev_layer = layer
                continue

            curr_vert  = FlowFixer.VERT.get(cut)
            needs_fix  = False

            # ── Verifica swing ────────────────────────────────────
            if prev_cut is not None:
                prev_vert = FlowFixer.VERT.get(prev_cut)
                if prev_vert is not None:
                    if cut not in FlowFixer.VALID_AFTER[prev_vert]:
                        needs_fix = True
                else:
                    # horizontal → oposto exato
                    if cut != FlowFixer.OPPOSITE.get(prev_cut):
                        needs_fix = True

            # ── Verifica consistência de layer ────────────────────
            # Se não vai corrigir por swing, ainda verifica layer
            if not needs_fix and prev_layer is not None:
                dlayer = layer - prev_layer
                if dlayer > 0 and curr_vert == 'up':
                    # Subiu de layer mas quer cortar para cima — inconsistente
                    # (caso da imagem: DOWN layer1 → UP layer2)
                    needs_fix = True
                elif dlayer < 0 and curr_vert == 'down':
                    # Desceu de layer mas quer cortar para baixo — inconsistente
                    needs_fix = True

            if needs_fix:
                cut = FlowFixer._best_cut(prev_cut, prev_layer, col, layer)
                note['_cutDirection'] = cut
                stats['swing'] += 1

            prev_cut   = cut
            prev_layer = layer

        return result

    @staticmethod
    def _best_cut(prev_cut, prev_layer, col, layer):
        """
        Escolhe o melhor cut respeitando:
          1. Swing (componente vertical oposto ao anterior)
          2. Consistência de layer (dlayer > 0 → não pode ser 'up')
          3. Bordas da grade
          4. Preferência pelo oposto exato (preserva criatividade)
          5. Fallback: qualquer válido da lista de swing
          6. Último recurso: DOT (só se não há nenhuma opção válida)
        """
        forbidden = FlowFixer._forbidden_for_pos(col, layer)

        # Candidatos de swing
        if prev_cut is not None:
            prev_vert = FlowFixer.VERT.get(prev_cut)
            if prev_vert is not None:
                candidates = list(FlowFixer.VALID_AFTER[prev_vert])
            else:
                opp = FlowFixer.OPPOSITE.get(prev_cut)
                candidates = [opp] if opp is not None else list(range(8))
        else:
            candidates = list(range(8))

        # Filtra por borda
        grid_ok = [c for c in candidates if c not in forbidden]

        # Filtra por consistência de layer
        if prev_layer is not None:
            dlayer = layer - prev_layer
            if dlayer > 0:
                # Subiu de layer → não pode ter comp 'up'
                layer_ok = [c for c in grid_ok if FlowFixer.VERT.get(c) != 'up']
            elif dlayer < 0:
                # Desceu de layer → não pode ter comp 'down'
                layer_ok = [c for c in grid_ok if FlowFixer.VERT.get(c) != 'down']
            else:
                layer_ok = grid_ok
        else:
            layer_ok = grid_ok

        pool = layer_ok if layer_ok else (grid_ok if grid_ok else candidates)

        if not pool:
            return FlowFixer.DOT

        # Prefere oposto exato
        exact = FlowFixer.OPPOSITE.get(prev_cut)
        if exact in pool:
            return exact

        # Prefere diagonais sobre puros (mais criativo)
        diagonals = [c for c in pool if c in {4, 5, 6, 7}]
        if diagonals:
            return random.choice(diagonals)

        return random.choice(pool)

    # ─────────────────────────────────────────────────────────────
    # 4. Bordas
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_edge(note, stats):
        line, layer = note['_lineIndex'], note['_lineLayer']
        cut = original = note['_cutDirection']

        if line == 0:
            if   cut == 2: note['_cutDirection'] = 8  # LEFT → DOT
            elif cut == 4: note['_cutDirection'] = 0  # UP_LEFT → UP
            elif cut == 6: note['_cutDirection'] = 1  # DOWN_LEFT → DOWN
        if line == 3:
            if   cut == 3: note['_cutDirection'] = 8  # RIGHT → DOT
            elif cut == 5: note['_cutDirection'] = 0  # UP_RIGHT → UP
            elif cut == 7: note['_cutDirection'] = 1  # DOWN_RIGHT → DOWN

        cut = note['_cutDirection']

        if layer == 0:
            if   cut == 1: note['_cutDirection'] = 0  # DOWN → UP
            elif cut == 6: note['_cutDirection'] = 4  # DOWN_LEFT → UP_LEFT
            elif cut == 7: note['_cutDirection'] = 5  # DOWN_RIGHT → UP_RIGHT
        if layer == 2:
            if   cut == 0: note['_cutDirection'] = 1  # UP → DOWN
            elif cut == 4: note['_cutDirection'] = 6  # UP_LEFT → DOWN_LEFT
            elif cut == 5: note['_cutDirection'] = 7  # UP_RIGHT → DOWN_RIGHT

        if note['_cutDirection'] != original:
            stats['edge'] += 1

    # ─────────────────────────────────────────────────────────────
    # 5. Braços cruzados (simultâneas)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_crossed_arms(all_notes, stats):
        THRESH = 0.0625
        result, i = list(all_notes), 0
        while i < len(result):
            g_start = i
            while i < len(result)-1 and abs(result[i+1]['_time'] - result[g_start]['_time']) <= THRESH:
                i += 1
            group = result[g_start:i+1]
            if len(group) >= 2:
                lns = [n for n in group if n['_type'] == FlowFixer.LEFT_HAND]
                rns = [n for n in group if n['_type'] == FlowFixer.RIGHT_HAND]
                if lns and rns:
                    for ln in lns:
                        for rn in rns:
                            if ln['_lineIndex'] >= 2 and rn['_lineIndex'] <= 1:
                                ln['_lineIndex'] = 3 - ln['_lineIndex']
                                rn['_lineIndex'] = 3 - rn['_lineIndex']
                                stats['crossed'] += 1
            i += 1
        return result

    # ─────────────────────────────────────────────────────────────
    # 6. Hitbox doubles
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_hitbox(all_notes, stats):
        result, i = list(all_notes), 0
        while i < len(result) - 1:
            a, b = result[i], result[i+1]
            if (abs(a['_time'] - b['_time']) <= 0.0625
                    and a['_type'] != b['_type']
                    and abs(a['_lineIndex'] - b['_lineIndex']) <= 1
                    and a['_lineLayer'] == b['_lineLayer']):
                if a['_lineLayer'] <= 1:
                    a['_lineLayer'], b['_lineLayer'] = 0, 2
                else:
                    a['_lineLayer'], b['_lineLayer'] = 2, 0
                stats['hitbox'] += 1
            i += 1
        return result