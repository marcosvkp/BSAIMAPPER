import numpy as np


def compute_target_nps(ai_confidence: float, target_stars: float) -> float:
    """Calcula NPS alvo com crescimento suave por estrelas.

    4 estrelas é tratada como baseline estável e dificuldades maiores
    crescem de forma progressiva, com retorno decrescente para evitar
    mapas fisicamente impossíveis.
    """
    ai_confidence = float(np.clip(ai_confidence, 0.0, 1.0))
    base_nps = 1.1 + ai_confidence * 5.8

    if target_stars <= 4.0:
        star_scale = 0.90 + 0.04 * target_stars
    elif target_stars <= 5.5:
        star_scale = 1.06 + (target_stars - 4.0) * 0.06
    else:
        # Após 5.5 estrelas o aumento de densidade é contido
        # e a dificuldade vem mais de padrão/flow do que de spam.
        star_scale = 1.15 + (target_stars - 5.5) * 0.03

    star_scale = float(np.clip(star_scale, 0.9, 1.26))
    return base_nps * star_scale


def compute_min_beat_fraction(target_stars: float) -> float:
    """Define cooldown mínimo em fração de beat.

    4.0~5.5 => 1/4 beat
    5.5+    => até ~1/6 beat (evita 1/8 que costuma quebrar jogabilidade)
    """
    if target_stars <= 5.5:
        return 4.0
    extra = min(target_stars - 5.5, 2.0)
    return 4.0 + extra


def candidate_score(prob: float, energy: float) -> float:
    """Score do candidato considerando confiança da IA e energia local."""
    return float(prob) * (0.85 + 0.45 * float(np.clip(energy, 0.0, 1.0)))
