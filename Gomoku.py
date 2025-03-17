class Gomoku:
    def __init__(self, size=15):
        self._size = size
        self._board = [['.' for _ in range(size)] for _ in range(size)]
        self._current_player = 'X'
        self._game_over = False

    @property
    def size(self):
        return self._size

    @property
    def current_player(self):
        return self._current_player

    @property
    def game_over(self):
        return self._game_over

    def imprimir_tabuleiro(self):
        print("\n  " + " ".join(f"{i:2}" for i in range(self.size)))
        for i, row in enumerate(self._board):
            print(f"{i:2} " + " ".join(row))

    def acao_valida(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self._board[row][col] == '.'

    def fazer_jogada(self, row, col): # sucessora
        if self.acao_valida(row, col):
            new_state = Gomoku(self.size)
            new_state._board = [r[:] for r in self._board]  # vai copiar o tabuleiro
            new_state._board[row][col] = self._current_player  # vai aplicar o que foi jogado no tabuleiro e depois
            new_state._current_player = 'O' if self._current_player == 'X' else 'X'  # vai mudar o jogador q joga
            new_state._game_over = self._game_over
            return new_state
        return None

    def verificar_vitoria(self, row, col):
        jogador = self._board[row][col]  # vai ver de quem foi a ultima jogada
        directions = [
            (0, 1),  # hor
            (1, 0),  # vert
            (1, 1),  # diago \
            (1, -1)  # diag /
        ]

        for dr, dc in directions:
            count = 1
            # verifica em uma direção
            i, j = row + dr, col + dc
            while 0 <= i < self.size and 0 <= j < self.size and self._board[i][j] == jogador:
                count += 1
                i += dr
                j += dc

            # verifica na outra
            i, j = row - dr, col - dc
            while 0 <= i < self.size and 0 <= j < self.size and self._board[i][j] == jogador:
                count += 1
                i -= dr
                j -= dc

            if count >= 5:
                return True
        return False

    def calcular_utilidade(self, jogador):

        utilidade = 0
        # verifica padrao
        for linha in range(self.size):
            for coluna in range(self.size):
                if self._board[linha][coluna] == jogador:
                    utilidade += self._avaliar_padroes(linha, coluna, jogador)
        return utilidade

    def _avaliar_padroes(self, linha, coluna, jogador):
        pontuacao = 0
        # verifica padrao nas 4 direcoes
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            i, j = linha + dr, coluna + dc
            while 0 <= i < self.size and 0 <= j < self.size and self._board[i][j] == jogador:
                count += 1
                i += dr
                j += dc
            # atribui pontuação com base no padrão
            if count >= 5:
                pontuacao += 1000  # ganhou
            elif count == 4:
                pontuacao += 100   # alinhou 4 pecas
            elif count == 3:
                pontuacao += 10    # alinhou 3
            elif count == 2:
                pontuacao += 1     # 2
        return pontuacao

    def jogar(self):
        while not self._game_over:
            self.imprimir_tabuleiro()
            print(f"\nJogador {self._current_player}, é sua vez!")
            
            try:
                row = int(input("Digite a linha (0-14): "))
                col = int(input("Digite a coluna (0-14): "))
            except ValueError:
                print("Entrada inválida! Use apenas números.")
                continue

            if not self.acao_valida(row, col):
                print("Jogada inválida! Tente novamente.")
                continue

            jogador_atual = self._current_player #manter ref do jogador


            novo_estado = self.fazer_jogada(row, col) # gerar estado
            if novo_estado:
                self._board = novo_estado._board
                self._current_player = novo_estado._current_player

                # ver se alguem ganhou
                if self.verificar_vitoria(row, col):
                    self.imprimir_tabuleiro()
                    print(f"\nJogador {jogador_atual} venceu! Parabéns!")
                    self._game_over = True

                # utilidade
                utilidade_x = self.calcular_utilidade('X')
                utilidade_o = self.calcular_utilidade('O')
                print(f"\nUtilidade (X): {utilidade_x}")
                print(f"Utilidade (O): {utilidade_o}")

if __name__ == "__main__":
    print("=== Gomoku (5 em linha) ===")
    print("Regras:")
    print("- Tabuleiro 15x15")
    print("- Jogadores alternam entre X e O")
    print("- Digite coordenadas de 0 a 14 para linha e coluna")
    print("- Primeiro a fazer 5 em linha vence!\n")
    
    jogo = Gomoku()
    jogo.jogar()