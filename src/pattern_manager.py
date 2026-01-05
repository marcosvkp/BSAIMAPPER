import numpy as np
import random

class PatternManager:
    """
    Gerenciador de padrões determinísticos e Regras de Flow.
    
    Conceito:
    - A IA decide a INTENSIDADE e o TIMING (Onde e quão forte).
    - Este gerenciador decide o PADRÃO (Stream, Jump, Stack) e a GEOMETRIA (Posição, Ângulo).
    """
    
    def __init__(self):
        # Estado do Flow
        self.left_hand_pos = (0, 0) # (Line, Layer)
        self.right_hand_pos = (3, 0)
        self.last_time = 0
        self.parity = 1 # 1 = Direita, 0 = Esquerda (Começa com a direita)
        
        # Definição de Padrões (Templates Relativos)
        # Formato: Lista de passos. Cada passo pode ter notas para esq/dir.
        self.patterns = {
            'single': [
                {'type': 'single'} 
            ],
            'double_down': [
                {'type': 'double', 'cut': 1} # Corte para baixo
            ],
            'stack': [
                {'type': 'single'}, 
                {'type': 'single', 'offset': 0.05} # Nota rápida logo depois
            ],
            'stream_2': [
                {'type': 'single'},
                {'type': 'single', 'offset': 0.125} # 1/4 beat a 120bpm
            ],
            'slider_diagonal': [
                {'type': 'single', 'cut': 6}, # Diagonal
                {'type': 'single', 'cut': 6, 'offset': 0.1, 'slide': True}
            ]
        }

    def get_pattern_for_intensity(self, intensity, time_gap):
        """
        Seleciona um padrão baseado na intensidade (probabilidade da IA) e tempo desde a última nota.
        """
        # Se for muito rápido (menos de 0.2s), força stream ou ignora para evitar spam impossível
        if time_gap < 0.15:
            return None # Cooldown forçado
            
        if time_gap < 0.25:
            return self.patterns['stream_2']
            
        # Alta intensidade
        if intensity > 0.85:
            choices = ['double_down', 'slider_diagonal', 'stack']
            weights = [0.4, 0.3, 0.3]
            key = random.choices(choices, weights=weights)[0]
            return self.patterns[key]
            
        # Intensidade média/baixa -> Single notes com bom flow
        return self.patterns['single']

    def apply_pattern(self, pattern, base_time, bpm):
        """
        Gera as notas reais aplicando regras de paridade e flow.
        """
        if pattern is None:
            return []
            
        notes = []
        
        for step in pattern:
            current_time = base_time + step.get('offset', 0)
            
            # Lógica de Paridade (Alternância de mãos)
            # Se for 'double', usa as duas mãos
            if step['type'] == 'double':
                # Mão Esquerda
                note_l = self._create_note(0, current_time, step.get('cut', 1))
                # Mão Direita
                note_r = self._create_note(1, current_time, step.get('cut', 1))
                notes.append(note_l)
                notes.append(note_r)
                # Resetar paridade ou manter? Em doubles geralmente reseta ou alterna ambas.
                # Vamos manter a alternância simples para o próximo single.
                
            else: # Single
                hand = self.parity
                note = self._create_note(hand, current_time, step.get('cut', None))
                notes.append(note)
                
                # Alterna a mão para a próxima nota
                self.parity = 1 - self.parity
                
        return notes

    def _create_note(self, hand, time, forced_cut=None):
        """
        Cria uma nota calculando a melhor posição e direção baseada na posição anterior da mão.
        """
        # Posição anterior da mão atual
        prev_line, prev_layer = self.left_hand_pos if hand == 0 else self.right_hand_pos
        
        # Decidir nova posição (Heurística simples de Flow)
        # Se a mão estava na esquerda (line 0/1), tende a ir para o meio ou ficar.
        # Se a mão estava em baixo (layer 0), tende a ir para cima (layer 1/2) para cortar para baixo.
        
        # Regra básica: Reset para posição confortável
        if hand == 0: # Esquerda
            new_line = random.choice([0, 1])
        else: # Direita
            new_line = random.choice([2, 3])
            
        # Alternância de altura
        if prev_layer == 0:
            new_layer = 1 # Vai para o meio
            cut_direction = 1 # Corte para baixo (Down)
        else:
            new_layer = 0 # Vai para baixo
            cut_direction = 0 # Corte para cima (Up)
            
        # Se o padrão forçou um corte (ex: slider), usa ele
        if forced_cut is not None:
            cut_direction = forced_cut
            
        # Atualiza estado
        if hand == 0:
            self.left_hand_pos = (new_line, new_layer)
        else:
            self.right_hand_pos = (new_line, new_layer)
            
        return {
            "_time": time,
            "_lineIndex": new_line,
            "_lineLayer": new_layer,
            "_type": hand,
            "_cutDirection": cut_direction
        }
