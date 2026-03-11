import random

class FlowState:
    """Mantém o controle da posição e direção de corte de cada mão."""
    def __init__(self, hand):
        self.hand = hand
        self.x = 1 if hand == 0 else 2
        self.y = 0
        self.cut = 1  # Começa cortando para baixo

    def update(self, x, y, cut):
        self.x, self.y, self.cut = x, y, cut

class PatternManager:
    """
    Responsável por traduzir as intenções da IA em padrões de notas concretos.
    Não filtra mais as notas, apenas as constrói.
    """
    def __init__(self, difficulty="ExpertPlus"):
        self.left = FlowState(0)
        self.right = FlowState(1)
        self.parity = 1  # 1=Direita, 0=Esquerda
        self.difficulty = difficulty

    def apply_pattern(self, time, bpm, complexity_idx, vertical_idx, time_gap):
        """
        Seleciona e gera um padrão de nota com base nas previsões da IA.
        A decisão de colocar uma nota JÁ FOI TOMADA. Esta função apenas a constrói.
        """
        notes = []
        
        # --- Lógica de Seleção de Padrão ---
        
        # Se a IA prevê alta complexidade e o tempo entre notas é curto, é um stream.
        # Um gap de 0.6 beats é um 1/4 de beat em 100 BPM, ou 1/8 de beat em 200 BPM.
        is_stream = complexity_idx == 2 and time_gap < (60 / bpm) * 0.6 

        if is_stream:
            notes = self._gen_stream_note(time, vertical_idx)
        else:
            # Para gaps maiores, usa a complexidade para decidir o padrão.
            if complexity_idx == 0: # Chill: Notas simples, mais espaçadas
                notes = self._gen_single_note(time, vertical_idx)
            elif complexity_idx == 1: # Dance: Mais movimento, doubles, wide notes
                # Maior chance de doubles ou wide notes para dance
                if random.random() < 0.4: # 40% chance de um double
                    notes = self._gen_double_note(time, vertical_idx)
                else:
                    notes = self._gen_wide_note(time, vertical_idx)
            elif complexity_idx == 2: # Tech: Padrões complexos, towers, doubles
                # Maior chance de patterns mais complexos para tech
                if random.random() < 0.3: # 30% chance de um tower
                    notes = self._gen_tower_stack(time, vertical_idx)
                else:
                    notes = self._gen_double_note(time, vertical_idx) # Doubles também são tech
        
        # Inverte a paridade da mão para a próxima nota (se um padrão de mão única foi gerado)
        if len(notes) == 1:
            self.parity = 1 - self.parity
        # Se um double foi gerado, a paridade não precisa mudar para a próxima batida.

        return notes

    # --- GERADORES DE PADRÕES ---

    def _create_note(self, hand, time, line, layer, cut):
        """Cria uma única nota e atualiza o estado da mão correspondente."""
        state = self.right if hand == 1 else self.left
        state.update(line, layer, cut)
        return [{
            "_time": time,
            "_lineIndex": line,
            "_lineLayer": layer,
            "_type": hand,
            "_cutDirection": cut
        }]

    def _gen_single_note(self, time, v_bias):
        """Gera uma nota simples, focada no fluxo e com viés de amplitude."""
        hand = self.parity
        
        # --- Escolha da Linha (Coluna) com viés de amplitude ---
        if hand == 0: # Mão esquerda
            # Mais chance de ir para a coluna 0 (mais à esquerda)
            line = random.choices([0, 1], weights=[0.6, 0.4], k=1)[0]
        else: # Mão direita
            # Mais chance de ir para a coluna 3 (mais à direita)
            line = random.choices([2, 3], weights=[0.4, 0.6], k=1)[0]
        
        # --- Escolha da Camada (Linha Vertical) baseada no v_bias ---
        if v_bias == 0: # Viés para baixo
            layer = random.choice([0, 1])
        elif v_bias == 2: # Viés para cima
            layer = random.choice([1, 2])
        else: # Viés para o meio
            layer = 1
        
        # --- Direção de Corte ---
        # Tenta manter o fluxo (cima/baixo)
        state = self.right if hand == 1 else self.left
        if layer > state.y: cut = 0 # Subiu, cortar para cima
        elif layer < state.y: cut = 1 # Desceu, cortar para baixo
        else: cut = random.choice([0, 1]) # Mesma altura, aleatório

        # Chance de corte diagonal para variedade
        if random.random() > 0.7:
            if hand == 0: cut = random.choice([4, 6]) # UpLeft, DownLeft
            else: cut = random.choice([5, 7]) # UpRight, DownRight
            
        return self._create_note(hand, time, line, layer, cut)

    def _gen_stream_note(self, time, v_bias):
        """Gera uma nota otimizada para streams, alternando a direção de corte."""
        hand = self.parity
        state = self.right if hand == 1 else self.left
        
        # Em streams, a posição vertical tende a ser mais estável e central
        layer = state.y
        if v_bias == 0: layer = 0
        elif v_bias == 2: layer = 1 # Camada 2 em streams é rara
        
        # Linhas centrais para streams
        line = random.choice([1, 2]) 

        # Alterna o corte (cima/baixo) para manter o fluxo do stream
        cut = 1 - state.cut if state.cut in [0, 1] else random.choice([0, 1])
        
        return self._create_note(hand, time, line, layer, cut)

    def _gen_wide_note(self, time, v_bias):
        """Gera uma nota mais afastada, comum em padrões "dance"."""
        hand = self.parity
        line = 0 if hand == 0 else 3 # Força as colunas externas
        layer = 0 if v_bias == 0 else 1 # Bias para camadas baixas/médias
        cut = random.choice([1, 6, 7]) if hand == 0 else random.choice([1, 4, 5]) # Cortes para fora ou para baixo
        return self._create_note(hand, time, line, layer, cut)

    def _gen_double_note(self, time, v_bias):
        """Gera duas notas simultâneas, uma para cada mão, com chance de ser wide."""
        # Decide se é um double wide ou central
        if random.random() < 0.5: # 50% chance para double wide
            l_line = 0
            r_line = 3
        else:
            l_line = 1
            r_line = 2
        
        layer = 0 if v_bias == 0 else 1 # Bias para camadas baixas/médias para doubles
        cut = 1 # Corte para baixo para doubles (mais comum e confortável)
        
        l_note = self._create_note(0, time, l_line, layer, cut)[0]
        r_note = self._create_note(1, time, r_line, layer, cut)[0]
        return [l_note, r_note]

    def _gen_tower_stack(self, time, v_bias):
        """Gera duas notas empilhadas verticalmente para a mesma mão."""
        hand = self.parity
        line = 1 if hand == 0 else 2 # Torres geralmente são mais centrais
        
        # A mão se move de baixo para cima
        n1 = self._create_note(hand, time, line, 0, 0)[0] # Nota de baixo, corte para cima
        n2 = self._create_note(hand, time, line, 2, 0)[0] # Nota de cima, corte para cima
        
        # Não inverte a paridade, pois a mesma mão foi usada
        return [n1, n2]
