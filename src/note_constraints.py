"""
note_constraints.py — Restrições de Posição e Mão V1

Roda APÓS o NoteFilter e ANTES do AngleNet.

Responsabilidades:
  1. Colunas por mão — garante que cada mão joga no lado correto da grade,
     com probabilidades configuráveis de "crossover" para variedade.

  2. Layer 2 + UP — nota no topo virada para cima é fisicamente desconfortável
     (o jogador precisa erguer o braço muito alto). Aplica 20% de chance de
     manter, e converte as demais para layer 0 (ou ajusta o cut).

  3. Conflito de posição — duas notas de mãos diferentes na mesma (col, layer)
     no mesmo beat causam hitbox clash. Resolve automaticamente.

Configuração de colunas:

  Mão ESQUERDA (hand=0) — zona primária: cols 0, 1
    P(col=0) = 60%  (mais comum — lado esquerdo nativo)
    P(col=1) = 40%  (centro-esquerda)
    --- crossover ---
    P(col=2) = 30%  (centro-direita — crossover leve)
    P(col=3) = 10%  (extrema direita — crossover forte)

  Mão DIREITA (hand=1) — zona primária: cols 2, 3
    P(col=2) = 40%  (centro-direita)
    P(col=3) = 60%  (mais comum — lado direito nativo)
    --- crossover ---
    P(col=1) = 30%  (centro-esquerda — crossover leve)
    P(col=0) = 10%  (extrema esquerda — crossover forte)

Estes pesos são aplicados como REDISTRIBUIÇÃO: se o PlaceNet já colocou a nota
numa coluna válida para a mão, mantemos. Se colocou num lugar impossível
(ex: hand=0 em col=3 sem probabilidade de crossover), redistribuímos.
"""

from __future__ import annotations
import random
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────────────────────────
# Configuração de colunas por mão
# ─────────────────────────────────────────────────────────────────

# Distribuição de colunas para mão ESQUERDA (hand=0)
# Formato: [(coluna, peso_relativo), ...]
LEFT_COL_WEIGHTS = [
    (0, 60),   # primária — esquerda nativa
    (1, 40),   # primária — centro-esquerda
    (2, 30),   # crossover leve
    (3, 10),   # crossover forte
]

# Distribuição de colunas para mão DIREITA (hand=1)
RIGHT_COL_WEIGHTS = [
    (2, 40),   # primária — centro-direita
    (3, 60),   # primária — direita nativa
    (1, 30),   # crossover leve
    (0, 10),   # crossover forte
]

# Probabilidade de manter nota no topo (layer=2) com cut UP (0, 4, 5)
LAYER2_UP_KEEP_PROB = 0.20   # 20% de chance de manter; 80% converte para layer 0

# Cut directions considerados "UP" (violam layer 2)
UP_CUTS = {0, 4, 5}  # UP, UP_LEFT, UP_RIGHT


# ─────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────

def _weighted_choice(weights: List[Tuple[int, int]], exclude: set = None) -> int:
    """Amostra uma coluna com peso, excluindo opções inválidas."""
    pool = [(col, w) for col, w in weights if col not in (exclude or set())]
    if not pool:
        # Fallback: qualquer coluna não excluída
        pool = [(col, 1) for col, _ in weights if col not in (exclude or set())]
    if not pool:
        return weights[0][0]  # último recurso
    total = sum(w for _, w in pool)
    r = random.random() * total
    acc = 0
    for col, w in pool:
        acc += w
        if r <= acc:
            return col
    return pool[-1][0]


def _assign_col(hand: int, exclude_cols: set = None) -> int:
    """Retorna uma coluna para a mão dada, respeitando pesos e exclusões."""
    weights = LEFT_COL_WEIGHTS if hand == 0 else RIGHT_COL_WEIGHTS
    return _weighted_choice(weights, exclude=exclude_cols)


# ─────────────────────────────────────────────────────────────────
# Aplicação das restrições
# ─────────────────────────────────────────────────────────────────

def apply_constraints(notes: List[Dict]) -> List[Dict]:
    """
    Aplica todas as restrições de posição às notas.

    Ordem de operações:
      1. Redistribui colunas de acordo com os pesos por mão
      2. Resolve conflitos de posição entre mãos diferentes no mesmo beat
      3. Converte layer=2 + cut UP para layer=0 (com 80% de probabilidade)

    Args:
        notes: lista de dicts (já ordenada por _time, com _type = 0 ou 1)

    Returns:
        Lista modificada in-place (também retornada para conveniência)
    """
    notes = sorted(notes, key=lambda n: n['_time'])

    col_fixes     = 0
    conflict_fixes = 0
    layer_fixes   = 0

    BEAT_THRESH = 0.0625  # notas dentro de 1/16 beat são "simultâneas"

    # ── Passo 1: redistribui colunas por mão ─────────────────────
    for note in notes:
        hand = note['_type']
        if hand not in (0, 1):
            continue

        col = note['_lineIndex']
        weights = LEFT_COL_WEIGHTS if hand == 0 else RIGHT_COL_WEIGHTS
        valid_cols = {c for c, _ in weights}

        # Coluna está completamente fora do espaço configurado?
        # (não deveria acontecer com grade 0-3, mas protege contra dados corrompidos)
        if col not in valid_cols:
            note['_lineIndex'] = _assign_col(hand)
            col_fixes += 1
            continue

        # Verifica se a coluna atual é "possível" para esta mão com base nos pesos.
        # Se o peso for zero (coluna não listada), redistribui.
        weight_map = dict(weights)
        if weight_map.get(col, 0) == 0:
            note['_lineIndex'] = _assign_col(hand)
            col_fixes += 1

    # ── Passo 2: resolve conflitos de posição ─────────────────────
    # Agrupa notas simultâneas e garante que duas mãos diferentes
    # não ocupam exatamente a mesma (col, layer).
    i = 0
    while i < len(notes):
        # Coleta grupo de notas simultâneas
        group = [notes[i]]
        j = i + 1
        while j < len(notes) and abs(notes[j]['_time'] - notes[i]['_time']) <= BEAT_THRESH:
            group.append(notes[j])
            j += 1

        if len(group) >= 2:
            _resolve_conflicts(group)
            conflict_fixes += sum(1 for n in group if n.get('_conflict_fixed'))
            for n in group:
                n.pop('_conflict_fixed', None)

        i = j

    # ── Passo 3: layer 2 + cut UP ─────────────────────────────────
    for note in notes:
        if (note['_lineLayer'] == 2 and
                note['_cutDirection'] in UP_CUTS):
            if random.random() > LAYER2_UP_KEEP_PROB:
                note['_lineLayer'] = 0
                layer_fixes += 1

    total = col_fixes + conflict_fixes + layer_fixes
    if total > 0:
        print(f"      [Constraints] col={col_fixes} | "
              f"conflict={conflict_fixes} | layer2up={layer_fixes}")
    return notes


def _resolve_conflicts(group: List[Dict]):
    """
    Dentro de um grupo de notas simultâneas, garante que duas notas
    de mãos diferentes não ocupam a mesma (col, layer).

    Estratégia:
      - Detecta pares (hand=0, hand=1) com mesma posição
      - Move a nota de menor prioridade (crossover) para uma coluna livre
    """
    # Posições ocupadas por cada mão
    occupied: Dict[int, set] = {0: set(), 1: set()}

    for note in group:
        hand = note['_type']
        if hand in (0, 1):
            pos = (note['_lineIndex'], note['_lineLayer'])
            occupied[hand].add(pos)

    # Detecta conflitos: mesma posição em mãos diferentes
    conflicts = occupied[0] & occupied[1]
    if not conflicts:
        return

    all_occupied = occupied[0] | occupied[1]

    for pos in conflicts:
        col, layer = pos
        # Encontra a nota da mão que está mais "fora do lugar" (crossover)
        conflicting = [
            n for n in group
            if n['_type'] in (0, 1)
            and n['_lineIndex'] == col
            and n['_lineLayer'] == layer
        ]

        if len(conflicting) < 2:
            continue

        # A nota com menor peso para sua mão nesta coluna é a "invasora"
        def col_weight(note):
            hand = note['_type']
            weights = dict(LEFT_COL_WEIGHTS if hand == 0 else RIGHT_COL_WEIGHTS)
            return weights.get(note['_lineIndex'], 0)

        conflicting.sort(key=col_weight)
        mover = conflicting[0]   # menor peso = mais fora do lugar

        # Tenta mover para uma coluna livre na mesma layer
        hand = mover['_type']
        weights = LEFT_COL_WEIGHTS if hand == 0 else RIGHT_COL_WEIGHTS
        taken_cols = {c for c, l in all_occupied if l == layer}

        for try_col, _ in sorted(weights, key=lambda x: -x[1]):
            if try_col not in taken_cols:
                old_col = mover['_lineIndex']
                mover['_lineIndex'] = try_col
                all_occupied.discard((old_col, layer))
                all_occupied.add((try_col, layer))
                occupied[hand].discard((old_col, layer))
                occupied[hand].add((try_col, layer))
                mover['_conflict_fixed'] = True
                break


# ─────────────────────────────────────────────────────────────────
# Validação (debug)
# ─────────────────────────────────────────────────────────────────

def validate(notes: List[Dict]) -> Dict:
    """
    Verifica se ainda existem violações após apply_constraints.
    Retorna dict com contagens de problemas encontrados.
    """
    BEAT_THRESH = 0.0625
    issues = {
        'wrong_col_left':  0,   # hand=0 em col 2 ou 3 (crossover não esperado)
        'wrong_col_right': 0,   # hand=1 em col 0 ou 1
        'layer2_up':       0,   # layer=2 com cut UP
        'position_conflict': 0, # duas mãos na mesma posição simultânea
    }

    # Colunas "nativas" por mão (sem crossover)
    native = {0: {0, 1}, 1: {2, 3}}

    for note in notes:
        hand = note['_type']
        col  = note['_lineIndex']
        layer = note['_lineLayer']
        cut  = note['_cutDirection']

        if hand == 0 and col in {2, 3}:
            issues['wrong_col_left'] += 1
        if hand == 1 and col in {0, 1}:
            issues['wrong_col_right'] += 1
        if layer == 2 and cut in UP_CUTS:
            issues['layer2_up'] += 1

    # Conflitos de posição
    i = 0
    notes_sorted = sorted(notes, key=lambda n: n['_time'])
    while i < len(notes_sorted):
        group = [notes_sorted[i]]
        j = i + 1
        while j < len(notes_sorted) and \
              abs(notes_sorted[j]['_time'] - notes_sorted[i]['_time']) <= BEAT_THRESH:
            group.append(notes_sorted[j])
            j += 1
        if len(group) >= 2:
            positions_0 = {(n['_lineIndex'], n['_lineLayer'])
                          for n in group if n['_type'] == 0}
            positions_1 = {(n['_lineIndex'], n['_lineLayer'])
                          for n in group if n['_type'] == 1}
            issues['position_conflict'] += len(positions_0 & positions_1)
        i = j

    return issues
