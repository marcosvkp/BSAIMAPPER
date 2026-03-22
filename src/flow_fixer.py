import copy
# random REMOVIDO — V14 e 100% deterministico

class FlowFixer:
    """
    Validador fisico V14 — 100% deterministico, zero aleatoriedade.

    Ordem de operacoes:
      1. Remove tower stacks (mesma mao, mesmo timestamp)
      2. Remove notas inatingiveis (distancia fisica vs tempo)
      3. _fix_edge: bordas da grade  <-- ANTES do swing
      4. _fix_swing: alternancia + consistencia de layer
      5. _fix_crossed_arms: bracos cruzados em simultaneas
      6. _fix_hitbox: doubles sobrepostos

    IMPORTANTE: edge roda ANTES do swing.
    Se edge rodar depois, pode converter um cut valido (ex: DOWN gerado
    pelo swing) de volta para UP em layer=0, quebrando o swing recém corrigido.
    Ex: UP(l=1) > UP(l=0)
        swing corrige 2a para DOWN_RIGHT
        edge veria DOWN_RIGHT em layer=0 e converteria para UP_RIGHT  <-- BUG
    Com edge primeiro: a nota em layer=0 ja chega pre-processada ao swing,
    e _best_cut escolhe um cut valido para aquela posicao + swing.

    ================================================================
    REGRA DE DOT (TRANSPARENTE)

    DOTs NAO alteram o estado de swing. A ultima nota direcional
    define o que a PROXIMA direcional deve ser, independente de
    quantos DOTs existam no meio.

    Garantido no mapa inteiro:
      UP  > DOT > UP                INVALIDO  -> corrigido
      UP  > DOT > DOWN              VALIDO    -> mantido
      UP  > DOT > DOT > DOWN        VALIDO    -> mantido
      UP  > DOT > DOT > UP          INVALIDO  -> corrigido
      UP  > DOT > DOT > DOT > DOWN  VALIDO    -> mantido
      UP  > DOT > DOT > DOT > UP    INVALIDO  -> corrigido
      UP_LEFT > DOT > UP_LEFT       INVALIDO  -> corrigido
      UP_LEFT > DOT > DOWN          VALIDO    -> mantido

    _best_cut() e totalmente deterministico: prioridade fixa,
    sem random.choice em nenhum caminho.
    ================================================================
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

    VERT = {
        0: 'up',    # UP
        1: 'down',  # DOWN
        2: None,    # LEFT  (horizontal puro)
        3: None,    # RIGHT (horizontal puro)
        4: 'up',    # UP_LEFT
        5: 'up',    # UP_RIGHT
        6: 'down',  # DOWN_LEFT
        7: 'down',  # DOWN_RIGHT
    }

    VERT_OPP = {'up': 'down', 'down': 'up'}

    VALID_AFTER = {
        'up':   {1, 6, 7},
        'down': {0, 4, 5},
    }

    OPPOSITE = {
        0: 1, 1: 0,
        2: 3, 3: 2,
        4: 7, 7: 4,
        5: 6, 6: 5,
    }

    # Ordem deterministica de preferencia por required_vert
    # Apos UP   (required='down'): DOWN_RIGHT > DOWN_LEFT > DOWN
    # Apos DOWN (required='up'):   UP_LEFT > UP_RIGHT > UP
    _PREFER = {
        'down': [7, 6, 1],
        'up':   [4, 5, 0],
        None:   [7, 4, 6, 5, 1, 0, 3, 2],
    }

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

    # -----------------------------------------------------------------
    # Interface publica
    # -----------------------------------------------------------------

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

        # Edge ANTES do swing: garante que _best_cut nunca escolha
        # um cut que sera invalidado pela borda logo em seguida.
        for n in left + right:
            FlowFixer._fix_edge(n, stats)

        left  = FlowFixer._fix_swing(left,  stats)
        right = FlowFixer._fix_swing(right, stats)

        all_notes = sorted(left + right, key=lambda x: x['_time'])
        all_notes = FlowFixer._fix_crossed_arms(all_notes, stats)
        all_notes = FlowFixer._fix_hitbox(all_notes, stats)

        result = sorted(all_notes + bombs, key=lambda x: x['_time'])

        total = sum(stats.values())
        if total > 0:
            print(
                f"      [FlowFixer V14] "
                f"same_hand={stats['same_hand']} | edge={stats['edge']} | "
                f"unreachable={stats['unreachable']} | swing={stats['swing']} | "
                f"crossed={stats['crossed']} | hitbox={stats['hitbox']}"
            )
        return result

    # -----------------------------------------------------------------
    # 1. Tower stacks
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 2. Atingibilidade
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 3. Bordas da grade  (roda ANTES do swing)
    # -----------------------------------------------------------------

    @staticmethod
    def _fix_edge(note, stats):
        line  = note['_lineIndex']
        layer = note['_lineLayer']
        cut   = original = note['_cutDirection']
        if cut == FlowFixer.DOT:
            return
        if line == 0:
            if   cut == 2: note['_cutDirection'] = 8   # LEFT -> DOT
            elif cut == 4: note['_cutDirection'] = 0   # UP_LEFT -> UP
            elif cut == 6: note['_cutDirection'] = 1   # DOWN_LEFT -> DOWN
        if line == 3:
            if   cut == 3: note['_cutDirection'] = 8   # RIGHT -> DOT
            elif cut == 5: note['_cutDirection'] = 0   # UP_RIGHT -> UP
            elif cut == 7: note['_cutDirection'] = 1   # DOWN_RIGHT -> DOWN
        cut = note['_cutDirection']
        if layer == 0:
            if   cut == 1: note['_cutDirection'] = 0   # DOWN -> UP
            elif cut == 6: note['_cutDirection'] = 4   # DOWN_LEFT -> UP_LEFT
            elif cut == 7: note['_cutDirection'] = 5   # DOWN_RIGHT -> UP_RIGHT
        if layer == 2:
            if   cut == 0: note['_cutDirection'] = 1   # UP -> DOWN
            elif cut == 4: note['_cutDirection'] = 6   # UP_LEFT -> DOWN_LEFT
            elif cut == 5: note['_cutDirection'] = 7   # UP_RIGHT -> DOWN_RIGHT
        if note['_cutDirection'] != original:
            stats['edge'] += 1

    # -----------------------------------------------------------------
    # 4. Swing — DOT TRANSPARENTE
    #
    # required_vert = componente vertical que a PROXIMA nota direcional DEVE ter.
    #   - Inicia como None (qualquer coisa e valida)
    #   - Apos nota com vert='up':   required_vert = 'down'
    #   - Apos nota com vert='down': required_vert = 'up'
    #   - Apos nota horizontal (None): required_vert nao muda
    #   - Apos DOT (cut=8): required_vert NAO MUDA  <-- regra central
    #
    # Consequencia: independente de quantos DOTs ha entre duas notas
    # direcionais, a regra de alternancia se aplica sempre.
    #
    # _best_cut() ja considera as bordas (_forbidden_for_pos), mas como
    # _fix_edge rodou antes, as notas ja chegam aqui com cuts ajustados
    # a suas posicoes, eliminando qualquer conflito residual.
    # -----------------------------------------------------------------

    @staticmethod
    def _fix_swing(hand_notes, stats):
        if len(hand_notes) < 2:
            return hand_notes

        result = list(hand_notes)
        required_vert = None   # componente vertical que a proxima direcional DEVE ter
        prev_horiz    = None   # ultimo cut horizontal (LEFT=2 / RIGHT=3) visto
        prev_layer    = None

        for note in result:
            cut   = note['_cutDirection']
            layer = note['_lineLayer']
            col   = note['_lineIndex']

            # DOT: transparente — nao altera nenhum estado
            if cut == FlowFixer.DOT:
                prev_layer = layer
                continue

            curr_vert = FlowFixer.VERT.get(cut)
            needs_fix = False

            # Verifica violacao de swing vertical
            if required_vert is not None and curr_vert is not None:
                if curr_vert != required_vert:
                    needs_fix = True

            # Verifica repeticao horizontal: RIGHT>RIGHT ou LEFT>LEFT
            if not needs_fix and curr_vert is None:
                if prev_horiz is not None and cut == prev_horiz:
                    needs_fix = True

            # Verifica consistencia de layer
            if not needs_fix and prev_layer is not None:
                dlayer = layer - prev_layer
                if dlayer > 0 and curr_vert == 'up':
                    needs_fix = True
                elif dlayer < 0 and curr_vert == 'down':
                    needs_fix = True

            if needs_fix:
                cut = FlowFixer._best_cut(required_vert, prev_horiz, prev_layer, col, layer)
                note['_cutDirection'] = cut
                curr_vert = FlowFixer.VERT.get(cut)
                stats['swing'] += 1

            # Atualiza estado
            if curr_vert is not None:
                required_vert = FlowFixer.VERT_OPP[curr_vert]
                prev_horiz = None   # nota com vert reseta o estado horizontal
            else:
                prev_horiz = cut    # registra o horizontal atual

            prev_layer = layer

        return result

    @staticmethod
    def _best_cut(required_vert, prev_horiz, prev_layer, col, layer):
        """
        Escolhe o melhor cut — 100% deterministico, sem random em nenhum caminho.

        required_vert : componente vertical obrigatorio ('up'/'down'/None)
        prev_horiz    : ultimo cut horizontal (2=LEFT / 3=RIGHT / None)
                        se definido, o cut escolhido nao pode ser igual a ele
        prev_layer    : layer da nota anterior (para consistencia de layer)

        Prioridade fixa:
          Passe 1: required_vert + nao repete horiz + borda + layer
          Passe 2: required_vert + nao repete horiz + borda (relaxa layer)
          Passe 3: required_vert + nao repete horiz (relaxa tudo)
          Passe 4: qualquer cut nao proibido
          Fallback: DOT
        """
        forbidden = FlowFixer._forbidden_for_pos(col, layer)

        def layer_ok(c):
            if prev_layer is None:
                return True
            v = FlowFixer.VERT.get(c)
            dlayer = layer - prev_layer
            if dlayer > 0 and v == 'up':
                return False
            if dlayer < 0 and v == 'down':
                return False
            return True

        def horiz_ok(c):
            # Nao pode repetir o mesmo horizontal
            if prev_horiz is not None and FlowFixer.VERT.get(c) is None:
                return c != prev_horiz
            return True

        order = FlowFixer._PREFER.get(required_vert, FlowFixer._PREFER[None])

        # Passe 1: tudo ok
        for c in order:
            if c not in forbidden and layer_ok(c) and horiz_ok(c):
                return c

        # Passe 2: relaxa layer
        for c in order:
            if c not in forbidden and horiz_ok(c):
                return c

        # Passe 3: relaxa bordas mas mantem horiz
        for c in order:
            if horiz_ok(c):
                return c

        # Passe 4: qualquer nao proibido
        for c in FlowFixer._PREFER[None]:
            if c not in forbidden:
                return c

        return FlowFixer.DOT

    # -----------------------------------------------------------------
    # 5. Bracos cruzados (simultaneas)
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 6. Hitbox doubles
    # -----------------------------------------------------------------

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