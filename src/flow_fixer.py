import math
import copy
from collections import defaultdict


class FlowFixer:
    """
    Simulador de Paridade e Jogabilidade (V9).

    Melhorias sobre V8:
    - Remoção de notas da mesma mão no mesmo timestamp (tower stacks impossíveis)
    - Detecção e correção de desequilíbrio de mão (colapso para só uma cor)
    - Detecção de vision blocks: nota A tapando nota B da outra mão logo depois
    - Reset de paridade após pausas longas
    - Preservação de cortes diagonais na correção de paridade
    - Hitbox entre mãos simultâneas
    - Notas inatingíveis por distância física
    - Não muta os dicts originais
    - Log de estatísticas por dificuldade gerada
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

    _VERTICAL_FLIP = {
        UP:         DOWN,
        DOWN:       UP,
        UP_LEFT:    DOWN_LEFT,
        UP_RIGHT:   DOWN_RIGHT,
        DOWN_LEFT:  UP_LEFT,
        DOWN_RIGHT: UP_RIGHT,
        LEFT:       LEFT,
        RIGHT:      RIGHT,
        DOT:        DOT,
    }

    # ── Tipos ─────────────────────────────────────────────────────
    LEFT_HAND  = 0
    RIGHT_HAND = 1
    BOMB       = 3

    # ── Limites físicos ───────────────────────────────────────────
    MAX_SAFE_LINE_JUMP      = 2     # distância máxima em colunas entre notas consecutivas
    MIN_BEATS_FOR_WIDE_JUMP = 0.4   # tempo mínimo para jumps de coluna extrema
    PARITY_RESET_GAP_BEATS  = 2.0   # pausa que reseta paridade

    # ── Desequilíbrio de mão ──────────────────────────────────────
    # Se uma mão tiver mais que este percentual das notas, redistribui o excesso.
    MAX_HAND_RATIO = 0.72

    # ── Vision blocks ─────────────────────────────────────────────
    # Janela em beats: nota A bloqueia B se B vier em menos que este valor depois de A
    VISION_BLOCK_WINDOW = 0.35

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

        stats = {
            "parity_fixes":        0,
            "hitbox_fixes":        0,
            "edge_fixes":          0,
            "parity_resets":       0,
            "unreachable_removed": 0,
            "same_hand_removed":   0,
            "hand_rebalanced":     0,
            "vision_block_fixed":  0,
        }

        notes_copy = [copy.copy(n) for n in notes]

        left_notes  = [n for n in notes_copy if n['_type'] == FlowFixer.LEFT_HAND]
        right_notes = [n for n in notes_copy if n['_type'] == FlowFixer.RIGHT_HAND]
        bombs       = [n for n in notes_copy if n['_type'] == FlowFixer.BOMB]

        # ── 1. Remove tower stacks impossíveis (mesma mão, mesmo tempo) ──
        left_notes  = FlowFixer._remove_same_hand_simultaneous(left_notes,  stats)
        right_notes = FlowFixer._remove_same_hand_simultaneous(right_notes, stats)

        # ── 2. Corrige desequilíbrio de mão ───────────────────────────────
        left_notes, right_notes = FlowFixer._rebalance_hands(
            left_notes, right_notes, bpm, stats
        )

        # ── 3. Paridade e atingibilidade por mão ──────────────────────────
        fixed_left  = FlowFixer._process_hand(left_notes,  FlowFixer.LEFT_HAND,  bpm, stats)
        fixed_right = FlowFixer._process_hand(right_notes, FlowFixer.RIGHT_HAND, bpm, stats)

        all_notes = fixed_left + fixed_right
        all_notes.sort(key=lambda x: x['_time'])

        # ── 4. Hitbox entre mãos simultâneas ──────────────────────────────
        all_notes = FlowFixer._fix_cross_hand_hitbox(all_notes, stats)

        # ── 5. Vision blocks ──────────────────────────────────────────────
        all_notes = FlowFixer._fix_vision_blocks(all_notes, stats)

        all_obj = all_notes + bombs
        all_obj.sort(key=lambda x: x['_time'])

        # ── Log ───────────────────────────────────────────────────────────
        if sum(stats.values()) > 0:
            print(
                f"      [FlowFixer V9] "
                f"paridade={stats['parity_fixes']} | "
                f"hitbox={stats['hitbox_fixes']} | "
                f"bordas={stats['edge_fixes']} | "
                f"resets={stats['parity_resets']} | "
                f"same_hand={stats['same_hand_removed']} | "
                f"rebalance={stats['hand_rebalanced']} | "
                f"vision={stats['vision_block_fixed']} | "
                f"inatingíveis={stats['unreachable_removed']}"
            )

        return all_obj

    # ─────────────────────────────────────────────────────────────
    # 1. Remove tower stacks impossíveis
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _remove_same_hand_simultaneous(hand_notes, stats):
        """
        Remove notas da mesma mão com timestamp idêntico (ou quase).

        Em Beat Saber, o mesmo sabre não pode cortar dois blocos ao mesmo
        tempo — a nota extra é impossível de acertar fisicamente.
        Quando há conflito, mantém a nota de layer mais baixa (mais natural).
        """
        if len(hand_notes) < 2:
            return hand_notes

        hand_notes.sort(key=lambda x: x['_time'])
        result = []
        i = 0
        while i < len(hand_notes):
            group = [hand_notes[i]]
            j = i + 1
            while j < len(hand_notes) and abs(hand_notes[j]['_time'] - hand_notes[i]['_time']) < 0.016:
                group.append(hand_notes[j])
                j += 1

            if len(group) > 1:
                best = min(group, key=lambda n: n['_lineLayer'])
                result.append(best)
                stats["same_hand_removed"] += len(group) - 1
            else:
                result.append(group[0])

            i = j

        return result

    # ─────────────────────────────────────────────────────────────
    # 2. Rebalanceamento de mão
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rebalance_hands(left_notes, right_notes, bpm, stats):
        """
        Se uma mão tiver mais que MAX_HAND_RATIO das notas totais,
        converte notas isoladas do excesso para a mão deficitária.

        Prioriza notas sem par próximo na outra mão para não criar
        doubles acidentais no processo de conversão.
        """
        total = len(left_notes) + len(right_notes)
        if total == 0:
            return left_notes, right_notes

        left_ratio = len(left_notes) / total

        if left_ratio > FlowFixer.MAX_HAND_RATIO:
            excess_hand    = left_notes
            deficit_hand   = right_notes
            deficit_type   = FlowFixer.RIGHT_HAND
            excess_is_left = True
        elif (1.0 - left_ratio) > FlowFixer.MAX_HAND_RATIO:
            excess_hand    = right_notes
            deficit_hand   = left_notes
            deficit_type   = FlowFixer.LEFT_HAND
            excess_is_left = False
        else:
            return left_notes, right_notes

        target_excess = int(total * 0.55)
        n_to_convert  = len(excess_hand) - target_excess
        if n_to_convert <= 0:
            return left_notes, right_notes

        beat_dur      = 60.0 / bpm
        deficit_times = {n['_time'] for n in deficit_hand}

        # Candidatos: notas isoladas (sem par próximo na mão deficitária)
        candidates = [
            note for note in excess_hand
            if not any(abs(note['_time'] - dt) < beat_dur * 0.3 for dt in deficit_times)
        ]

        # Converte com passo uniforme para distribuir ao longo da música
        candidates.sort(key=lambda n: n['_time'])
        step      = max(1, len(candidates) // n_to_convert)
        converted = 0

        for idx in range(0, len(candidates), step):
            if converted >= n_to_convert:
                break
            note = candidates[idx]
            note['_type'] = deficit_type
            # Ajusta coluna para o lado correto da mão destino
            if deficit_type == FlowFixer.LEFT_HAND:
                note['_lineIndex'] = min(note['_lineIndex'], 1)
            else:
                note['_lineIndex'] = max(note['_lineIndex'], 2)
            deficit_times.add(note['_time'])
            converted += 1
            stats["hand_rebalanced"] += 1

        # Reconstrói listas separadas por tipo
        all_converted = left_notes + right_notes
        new_left  = [n for n in all_converted if n['_type'] == FlowFixer.LEFT_HAND]
        new_right = [n for n in all_converted if n['_type'] == FlowFixer.RIGHT_HAND]

        return new_left, new_right

    # ─────────────────────────────────────────────────────────────
    # 3. Paridade e atingibilidade por mão
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _process_hand(hand_notes, hand_type, bpm, stats):
        if not hand_notes:
            return []

        hand_notes.sort(key=lambda x: x['_time'])

        last_ended_high = False
        last_line       = 1 if hand_type == FlowFixer.LEFT_HAND else 2
        processed       = []

        # Primeira nota
        first = hand_notes[0]
        FlowFixer._fix_edge_cut(first, hand_type, stats)
        cut = first['_cutDirection']
        if cut in FlowFixer.UP_CUTS:
            last_ended_high = True
        elif cut in FlowFixer.DOWN_CUTS:
            last_ended_high = False
        last_line = first['_lineIndex']
        processed.append(first)

        for i in range(1, len(hand_notes)):
            prev = processed[-1]
            curr = hand_notes[i]

            FlowFixer._fix_edge_cut(curr, hand_type, stats)

            time_gap_beats = curr['_time'] - prev['_time']

            # Reset de paridade após pausa longa
            if time_gap_beats >= FlowFixer.PARITY_RESET_GAP_BEATS:
                cut = curr['_cutDirection']
                if cut in FlowFixer.UP_CUTS:
                    last_ended_high = True
                elif cut in FlowFixer.DOWN_CUTS:
                    last_ended_high = False
                else:
                    last_ended_high = False
                last_line = curr['_lineIndex']
                processed.append(curr)
                stats["parity_resets"] += 1
                continue

            # Atingibilidade
            line_dist = abs(curr['_lineIndex'] - last_line)
            if (line_dist > FlowFixer.MAX_SAFE_LINE_JUMP
                    and time_gap_beats < FlowFixer.MIN_BEATS_FOR_WIDE_JUMP):
                stats["unreachable_removed"] += 1
                continue

            # Paridade
            cut       = curr['_cutDirection']
            goes_up   = cut in FlowFixer.UP_CUTS
            goes_down = cut in FlowFixer.DOWN_CUTS

            if last_ended_high and goes_up:
                curr['_cutDirection'] = FlowFixer._VERTICAL_FLIP[cut]
                stats["parity_fixes"] += 1
                goes_up, goes_down = False, True
            elif not last_ended_high and goes_down:
                curr['_cutDirection'] = FlowFixer._VERTICAL_FLIP[cut]
                stats["parity_fixes"] += 1
                goes_down, goes_up = False, True

            # Atualiza estado
            final_cut = curr['_cutDirection']
            if final_cut in FlowFixer.UP_CUTS:
                last_ended_high = True
            elif final_cut in FlowFixer.DOWN_CUTS:
                last_ended_high = False

            last_line = curr['_lineIndex']
            processed.append(curr)

        return processed

    # ─────────────────────────────────────────────────────────────
    # 4. Hitbox entre mãos simultâneas
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_cross_hand_hitbox(all_notes, stats):
        """
        Notas de mãos diferentes no mesmo timestamp com posições sobrepostas
        são separadas verticalmente (uma vai para layer 0, outra para layer 2).
        """
        if len(all_notes) < 2:
            return all_notes

        result = list(all_notes)
        i = 0
        while i < len(result) - 1:
            a = result[i]
            b = result[i + 1]

            if abs(a['_time'] - b['_time']) > 0.0625:
                i += 1
                continue
            if a['_type'] == b['_type']:
                i += 1
                continue

            col_dist   = abs(a['_lineIndex'] - b['_lineIndex'])
            layer_dist = abs(a['_lineLayer'] - b['_lineLayer'])

            if col_dist <= 1 and layer_dist == 0:
                if a['_lineLayer'] <= 1:
                    a['_lineLayer'] = 0
                    b['_lineLayer'] = 2
                else:
                    a['_lineLayer'] = 2
                    b['_lineLayer'] = 0
                stats["hitbox_fixes"] += 1

            i += 1

        return result

    # ─────────────────────────────────────────────────────────────
    # 5. Vision blocks
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_vision_blocks(all_notes, stats):
        """
        Nota A de uma mão bloqueia a visão de nota B da outra mão se:
          - B vem logo depois de A (< VISION_BLOCK_WINDOW beats)
          - Mesma coluna ou colunas adjacentes (col_dist <= 1)
          - Mesma camada (layer_dist == 0)

        Correção: desloca B para uma camada diferente.
        Nunca remove — apenas reposiciona verticalmente.
        """
        if len(all_notes) < 2:
            return all_notes

        result = list(all_notes)

        for i in range(len(result) - 1):
            a = result[i]
            if a['_type'] == FlowFixer.BOMB:
                continue

            for j in range(i + 1, len(result)):
                b = result[j]

                time_diff = b['_time'] - a['_time']
                if time_diff < 0:
                    continue
                if time_diff >= FlowFixer.VISION_BLOCK_WINDOW:
                    break  # lista ordenada por tempo, não há mais candidatos

                if b['_type'] == FlowFixer.BOMB:
                    continue
                if a['_type'] == b['_type']:
                    continue  # mesma mão não é vision block

                col_dist   = abs(a['_lineIndex'] - b['_lineIndex'])
                layer_dist = abs(a['_lineLayer'] - b['_lineLayer'])

                if col_dist <= 1 and layer_dist == 0:
                    # Desloca B para camada diferente
                    if b['_lineLayer'] == 0:
                        b['_lineLayer'] = 1
                    elif b['_lineLayer'] == 1:
                        b['_lineLayer'] = 0
                    else:
                        b['_lineLayer'] = 1
                    stats["vision_block_fixed"] += 1

        return result

    # ─────────────────────────────────────────────────────────────
    # Helper: cortes de borda
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_edge_cut(note, hand_type, stats):
        """
        Evita cortes que apontam para fora da grade.

        Grade Beat Saber: 4 colunas (0=esquerda, 3=direita), 3 camadas (0=baixo, 2=cima).

        Bordas laterais:
          col 0 → proibido cortar para ESQUERDA: LEFT(2), UP_LEFT(4), DOWN_LEFT(6)
          col 3 → proibido cortar para DIREITA:  RIGHT(3), UP_RIGHT(5), DOWN_RIGHT(7)

        Bordas verticais:
          layer 0 → proibido cortar para BAIXO:  DOWN(1), DOWN_LEFT(6), DOWN_RIGHT(7)
          layer 2 → proibido cortar para CIMA:   UP(0),   UP_LEFT(4),   UP_RIGHT(5)

        Correção: remove apenas o componente inválido, preservando o outro.
          UP_LEFT  na col 0  → UP    (remove LEFT,  mantém UP)
          DOWN_LEFT na col 0 → DOWN  (remove LEFT,  mantém DOWN)
          LEFT     na col 0  → DOT   (sem componente válido)
          etc.
        """
        line     = note['_lineIndex']
        layer    = note['_lineLayer']
        cut      = note['_cutDirection']
        original = cut

        # ── Borda esquerda (col 0): não pode cortar para esquerda ────
        if line == 0:
            if cut == FlowFixer.LEFT:
                note['_cutDirection'] = FlowFixer.DOT
            elif cut == FlowFixer.UP_LEFT:
                note['_cutDirection'] = FlowFixer.UP
            elif cut == FlowFixer.DOWN_LEFT:
                note['_cutDirection'] = FlowFixer.DOWN

        # ── Borda direita (col 3): não pode cortar para direita ──────
        if line == 3:
            if cut == FlowFixer.RIGHT:
                note['_cutDirection'] = FlowFixer.DOT
            elif cut == FlowFixer.UP_RIGHT:
                note['_cutDirection'] = FlowFixer.UP
            elif cut == FlowFixer.DOWN_RIGHT:
                note['_cutDirection'] = FlowFixer.DOWN

        # Re-lê o corte após possível correção lateral
        cut = note['_cutDirection']

        # ── Borda inferior (layer 0): não pode cortar para baixo ─────
        if layer == 0:
            if cut == FlowFixer.DOWN:
                note['_cutDirection'] = FlowFixer.UP
            elif cut == FlowFixer.DOWN_LEFT:
                note['_cutDirection'] = FlowFixer.UP_LEFT
            elif cut == FlowFixer.DOWN_RIGHT:
                note['_cutDirection'] = FlowFixer.UP_RIGHT

        # ── Borda superior (layer 2): não pode cortar para cima ──────
        if layer == 2:
            if cut == FlowFixer.UP:
                note['_cutDirection'] = FlowFixer.DOWN
            elif cut == FlowFixer.UP_LEFT:
                note['_cutDirection'] = FlowFixer.DOWN_LEFT
            elif cut == FlowFixer.UP_RIGHT:
                note['_cutDirection'] = FlowFixer.DOWN_RIGHT

        if note['_cutDirection'] != original:
            stats["edge_fixes"] += 1

        return note