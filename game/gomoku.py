
from .constants import *

class Gomoku:
    
    def __init__(self, size=BOARD_SIZE):
        self._size = size
        self._board = [[EMPTY_CELL for _ in range(size)] for _ in range(size)]
        self._current_player = PLAYER_X
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

    def get_board(self):
        return self._board

    def imprimir_tabuleiro(self):
        print("\n  " + " ".join(f"{i:2}" for i in range(self.size)))
        for i, row in enumerate(self._board):
            print(f"{i:2} " + " ".join(row))

    def acao_valida(self, row, col):
        return (0 <= row < self.size and 
                0 <= col < self.size and 
                self._board[row][col] == EMPTY_CELL)

    def fazer_jogada(self, row, col):
        if self.acao_valida(row, col):
            new_state = Gomoku(self.size)
            new_state._board = [r[:] for r in self._board]  # Copia o tabuleiro
            new_state._board[row][col] = self._current_player
            new_state._current_player = PLAYER_O if self._current_player == PLAYER_X else PLAYER_X
            new_state._game_over = self._game_over
            return new_state
        return None

    def verificar_vitoria(self, row, col):
        jogador = self._board[row][col]

        for dr, dc in DIRECTIONS:
            count = 1
            
            # Verifica em uma direção
            i, j = row + dr, col + dc
            while (0 <= i < self.size and 
                   0 <= j < self.size and 
                   self._board[i][j] == jogador):
                count += 1
                i += dr
                j += dc

            # Verifica na direção oposta
            i, j = row - dr, col - dc
            while (0 <= i < self.size and 
                   0 <= j < self.size and 
                   self._board[i][j] == jogador):
                count += 1
                i -= dr
                j -= dc

            if count >= WIN_CONDITION:
                return True
        return False

    def calcular_utilidade(self, jogador):
        utilidade = 0
        for linha in range(self.size):
            for coluna in range(self.size):
                if self._board[linha][coluna] == jogador:
                    utilidade += self._avaliar_padroes(linha, coluna, jogador)
        return utilidade

    def _avaliar_padroes(self, linha, coluna, jogador):
        pontuacao = 0
        
        for dr, dc in DIRECTIONS:
            count = 1
            i, j = linha + dr, coluna + dc
            
            while (0 <= i < self.size and 
                   0 <= j < self.size and 
                   self._board[i][j] == jogador):
                count += 1
                i += dr
                j += dc
            
            if count >= 5:
                pontuacao += SCORE_WIN
            elif count == 4:
                pontuacao += SCORE_FOUR
            elif count == 3:
                pontuacao += SCORE_THREE
            elif count == 2:
                pontuacao += SCORE_TWO
                
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

            jogador_atual = self._current_player
            novo_estado = self.fazer_jogada(row, col)
            
            if novo_estado:
                self._board = novo_estado._board
                self._current_player = novo_estado._current_player

                if self.verificar_vitoria(row, col):
                    self.imprimir_tabuleiro()
                    print(f"\nJogador {jogador_atual} venceu! Parabéns!")
                    self._game_over = True

                # Mostra utilidades
                utilidade_x = self.calcular_utilidade(PLAYER_X)
                utilidade_o = self.calcular_utilidade(PLAYER_O)
                print(f"\nUtilidade (X): {utilidade_x}")
                print(f"Utilidade (O): {utilidade_o}")