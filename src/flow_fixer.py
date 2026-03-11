import math
import copy


class FlowFixer:
    """
    Simulador de Paridade e Jogabilidade (V8).

    Melhorias sobre V7:
    - Reset de paridade após pausas longas (mão volta à posição neutra)
    - Preservação de cortes diagonais — só inverte o componente vertical errado
    - Detecção de colisão de hitbox entre mãos no mesmo timestamp (doubles/jumps)
    - Detecção de notas inatingíveis por distância física excessiva
    - Não muta os dicts originais (copia antes de modificar)
    - Log de estatísticas de correção para diagnóstico
    """

    # ── Direções ──────────────────────────────────────────────────
    UP         = 0
    DOWN       = 1
    LEFT       = 2
    RIGHT      = 3
    UP_LEFT    = 4
    UP_RIGHT   = 5
    DOWN_LEFT  = 6
    DOWN_RIGHT = 7
    DOT        = 8

    UP_CUTS   = {UP, UP_LEFT, UP_RIGHT}
    DOWN_CUTS = {DOWN, DOWN_LEFT, DOWN_RIGHT}
    SIDE_CUTS = {LEFT, RIGHT}

    # Inversão vertical de cada direção (para corrigir paridade sem perder diagonal)
    _VERTICAL_FLIP = {
        UP:         DOWN,
        DOWN:       UP,
        UP_LEFT:    DOWN_LEFT,
        UP_RIGHT:   DOWN_RIGHT,
        DOWN_LEFT:  UP_LEFT,
        DOWN_RIGHT: UP_RIGHT,
        LEFT:       LEFT,    # laterais não têm componente vertical — mantém
        RIGHT:      RIGHT,
        DOT:        DOT,
    }

    # ── Tipos ─────────────────────────────────────────────────────
    LEFT_HAND  = 0
    RIGHT_HAND = 1
    BOMB       = 3

    # ── Limites físicos ───────────────────────────────────────────
    # Distância máxima de colunas entre notas consecutivas da mesma mão
    # considerada atingível confortavelmente.
    MAX_SAFE_LINE_JUMP = 2   # col 0 → col 3 = 3 de distância → perigoso
    # Tempo mínimo (em beats) entre notas da mesma mão em posições extremas
    MIN_BEATS_FOR_WIDE_JUMP = 0.4

    # Gap em beats acima do qual consideramos que a mão "resetou" a posição
    # (o jogador teve tempo de voltar ao centro)
    PARITY_RESET_GAP_BEATS = 2.0

    # ─────────────────────────────────────────────────────────────
    # Interface pública
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def fix(notes, bpm):
        """
        Corrige o fluxo de um conjunto de notas geradas pela IA.

        Args:
            notes : list[dict]  — notas brutas do PatternManager
            bpm   : float

        Returns:
            list[dict]  — notas corrigidas (novos dicts, originais intocados)
        """
        if not notes:
            return []

        stats = {"parity_fixes": 0, "hitbox_fixes": 0, "edge_fixes": 0,
                 "parity_resets": 0, "unreachable_removed": 0}

        # Copia profunda para não mutar os originais
        notes_copy = [copy.copy(n) for n in notes]

        left_notes  = [n for n in notes_copy if n['_type'] == FlowFixer.LEFT_HAND]
        right_notes = [n for n in notes_copy if n['_type'] == FlowFixer.RIGHT_HAND]
        bombs       = [n for n in notes_copy if n['_type'] == FlowFixer.BOMB]

        fixed_left  = FlowFixer._process_hand(left_notes,  FlowFixer.LEFT_HAND,  bpm, stats)
        fixed_right = FlowFixer._process_hand(right_notes, FlowFixer.RIGHT_HAND, bpm, stats)

        all_notes = fixed_left + fixed_right
        all_notes.sort(key=lambda x: x['_time'])

        # Passa de hitbox entre mãos (doubles/jumps)
        all_notes = FlowFixer._fix_cross_hand_hitbox(all_notes, stats)

        all_obj = all_notes + bombs
        all_obj.sort(key=lambda x: x['_time'])

        total_fixes = stats["parity_fixes"] + stats["hitbox_fixes"] + stats["edge_fixes"]
        if total_fixes > 0:
            print(f"      [FlowFixer] Correções: paridade={stats['parity_fixes']} | "
                  f"hitbox={stats['hitbox_fixes']} | bordas={stats['edge_fixes']} | "
                  f"resets={stats['parity_resets']} | "
                  f"inatingíveis removidos={stats['unreachable_removed']}")

        return all_obj

    # ─────────────────────────────────────────────────────────────
    # Processamento por mão
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _process_hand(hand_notes, hand_type, bpm, stats):
        if not hand_notes:
            return []

        hand_notes.sort(key=lambda x: x['_time'])

        # True  = mão terminou em CIMA (próxima deve descer)
        # False = mão terminou em BAIXO (próxima deve subir)
        last_ended_high = False
        last_line       = 1 if hand_type == FlowFixer.LEFT_HAND else 2
        processed       = []

        # ── Primeira nota ─────────────────────────────────────────
        first = hand_notes[0]
        FlowFixer._fix_edge_cut(first, hand_type, stats)
        cut = first['_cutDirection']
        if cut in FlowFixer.UP_CUTS:
            last_ended_high = True
        elif cut in FlowFixer.DOWN_CUTS:
            last_ended_high = False
        # laterais/dot: assume neutro (False)
        last_line = first['_lineIndex']
        processed.append(first)

        for i in range(1, len(hand_notes)):
            prev = processed[-1]
            curr = hand_notes[i]

            # ── Correção de borda antes de qualquer análise ────────
            FlowFixer._fix_edge_cut(curr, hand_type, stats)

            # ── Reset de paridade após gap longo ───────────────────
            time_gap_beats = curr['_time'] - prev['_time']
            if time_gap_beats >= FlowFixer.PARITY_RESET_GAP_BEATS:
                # Pausa longa: jogador voltou à posição neutra.
                # Não corrigimos paridade — aceitamos o que a IA escolheu
                # e recalibramos o estado a partir desta nota.
                cut = curr['_cutDirection']
                if cut in FlowFixer.UP_CUTS:
                    last_ended_high = True
                elif cut in FlowFixer.DOWN_CUTS:
                    last_ended_high = False
                else:
                    last_ended_high = False  # neutra após pausa
                last_line = curr['_lineIndex']
                processed.append(curr)
                stats["parity_resets"] += 1
                continue

            # ── Verificação de atingibilidade ──────────────────────
            line_dist = abs(curr['_lineIndex'] - last_line)
            if (line_dist > FlowFixer.MAX_SAFE_LINE_JUMP
                    and time_gap_beats < FlowFixer.MIN_BEATS_FOR_WIDE_JUMP):
                # Nota muito longe em tempo muito curto — remove
                stats["unreachable_removed"] += 1
                continue

            # ── Correção de paridade ───────────────────────────────
            cut = curr['_cutDirection']
            goes_up   = cut in FlowFixer.UP_CUTS
            goes_down = cut in FlowFixer.DOWN_CUTS
            is_side   = cut in FlowFixer.SIDE_CUTS
            is_dot    = cut == FlowFixer.DOT

            must_go_down = last_ended_high

            if must_go_down and goes_up:
                # Inverte só o componente vertical, preservando diagonal
                curr['_cutDirection'] = FlowFixer._VERTICAL_FLIP[cut]
                stats["parity_fixes"] += 1
                goes_up   = False
                goes_down = True

            elif not must_go_down and goes_down:
                curr['_cutDirection'] = FlowFixer._VERTICAL_FLIP[cut]
                stats["parity_fixes"] += 1
                goes_down = False
                goes_up   = True

            # ── Atualiza estado ────────────────────────────────────
            final_cut = curr['_cutDirection']
            if final_cut in FlowFixer.UP_CUTS:
                last_ended_high = True
            elif final_cut in FlowFixer.DOWN_CUTS:
                last_ended_high = False
            elif is_side or is_dot:
                # Corte lateral/dot: o pulso continua na mesma direção vertical
                # lógica mais conservadora — não inverte o estado
                pass

            last_line = curr['_lineIndex']
            processed.append(curr)

        return processed

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_edge_cut(note, hand_type, stats):
        """
        Evita cortes que apontam para fora da grade nas colunas de borda.
        Esquerda (col 0) não pode cortar para esquerda.
        Direita (col 3) não pode cortar para direita.
        """
        line = note['_lineIndex']
        cut  = note['_cutDirection']
        original = cut

        if hand_type == FlowFixer.LEFT_HAND and line == 0:
            # Proibido: LEFT(2), UP_RIGHT(5) apontando pra fora, DOWN_RIGHT(7)
            if cut in {FlowFixer.LEFT, FlowFixer.UP_RIGHT, FlowFixer.DOWN_RIGHT}:
                # Converte para o equivalente sem componente lateral
                if cut == FlowFixer.UP_RIGHT:
                    note['_cutDirection'] = FlowFixer.UP
                elif cut == FlowFixer.DOWN_RIGHT:
                    note['_cutDirection'] = FlowFixer.DOWN
                else:
                    note['_cutDirection'] = FlowFixer.DOT

        elif hand_type == FlowFixer.RIGHT_HAND and line == 3:
            # Proibido: RIGHT(3), UP_LEFT(4), DOWN_LEFT(6)
            if cut in {FlowFixer.RIGHT, FlowFixer.UP_LEFT, FlowFixer.DOWN_LEFT}:
                if cut == FlowFixer.UP_LEFT:
                    note['_cutDirection'] = FlowFixer.UP
                elif cut == FlowFixer.DOWN_LEFT:
                    note['_cutDirection'] = FlowFixer.DOWN
                else:
                    note['_cutDirection'] = FlowFixer.DOT

        if note['_cutDirection'] != original:
            stats["edge_fixes"] += 1

    @staticmethod
    def _fix_cross_hand_hitbox(all_notes, stats):
        """
        Detecta notas de mãos diferentes no mesmo timestamp com posições
        sobrepostas (hitbox overlap) e separa verticalmente quando possível.

        Regra: duas notas no mesmo beat são overlap se estiverem na mesma
        coluna E mesma camada, ou em colunas adjacentes E mesma camada.
        """
        if len(all_notes) < 2:
            return all_notes

        result = list(all_notes)
        i = 0
        while i < len(result) - 1:
            a = result[i]
            b = result[i + 1]

            # Mesmo timestamp (tolerância de 1/16 de beat)
            if abs(a['_time'] - b['_time']) > 0.0625:
                i += 1
                continue

            # Só verifica entre mãos diferentes
            if a['_type'] == b['_type']:
                i += 1
                continue

            col_dist   = abs(a['_lineIndex'] - b['_lineIndex'])
            layer_dist = abs(a['_lineLayer'] - b['_lineLayer'])

            # Overlap: mesma coluna ou colunas adjacentes na mesma camada
            is_overlap = (col_dist <= 1 and layer_dist == 0)

            if is_overlap:
                # Tenta separar verticalmente: move a nota de cima para layer 2,
                # a de baixo para layer 0, se ainda não estiverem separadas
                if a['_lineLayer'] <= 1:
                    a['_lineLayer'] = 0
                    b['_lineLayer'] = 2
                else:
                    a['_lineLayer'] = 2
                    b['_lineLayer'] = 0
                stats["hitbox_fixes"] += 1

            i += 1

        return result