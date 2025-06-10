import pygame
import sys
from game.gomoku import Gomoku
from game.constants import BOARD_SIZE, PLAYER_X, PLAYER_O
from .colors import *

class GomokuGUI: 

    def __init__(self):
        pygame.init()

        self.CELL_SIZE = 30
        self.MARGIN = 50
        self.INPUT_HEIGHT = 100

        self.WIDTH = BOARD_SIZE * self.CELL_SIZE + 2 * self.MARGIN
        self.HEIGHT = BOARD_SIZE * self.CELL_SIZE + 2 * self.INPUT_HEIGHT

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Gomoku - 5 em linha")

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 32)

        self.game = Gomoku()

        self.input_row = ""
        self.input_col = ""
        self.active_input = None
        self.show_error = False
        self.error_message = ""

        self._setup_ui_elements()

    def _setup_ui_elements(self):
        input_y = self.HEIGHT - 80
        self.row_input_rect = pygame.Rect(50, input_y, 100, 30)
        self.col_input_rect = pygame.Rect(200, input_y, 100, 30)
        self.button_rect = pygame.Rect(350, input_y, 80, 30)
        self.reset_button_rect = pygame.Rect(450, input_y, 80, 30)

    def draw_board(self):
        # Fundo do tabuleiro
        board_rect = pygame.Rect(
            self.MARGIN - 10, 
            self.MARGIN - 10, 
            BOARD_SIZE * self.CELL_SIZE + 20, 
            BOARD_SIZE * self.CELL_SIZE + 20
        )
        pygame.draw.rect(self.screen, BROWN, board_rect)
        
        # Linhas do tabuleiro
        for i in range(BOARD_SIZE + 1):
            # Linhas horizontais
            start_pos = (self.MARGIN, self.MARGIN + i * self.CELL_SIZE)
            end_pos = (self.MARGIN + BOARD_SIZE * self.CELL_SIZE, self.MARGIN + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 1)
            
            # Linhas verticais
            start_pos = (self.MARGIN + i * self.CELL_SIZE, self.MARGIN)
            end_pos = (self.MARGIN + i * self.CELL_SIZE, self.MARGIN + BOARD_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 1)
        
        self._draw_pieces()

    def _draw_pieces(self):
        board = self.game.get_board()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] != '.':
                    self._draw_piece(row, col, board[row][col])

    def _draw_piece(self, row, col, player):
        center_x = self.MARGIN + col * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.MARGIN + row * self.CELL_SIZE + self.CELL_SIZE // 2
        radius = self.CELL_SIZE // 2 - 3
        
        if player == PLAYER_X:
            pygame.draw.circle(self.screen, PLAYER_X_COLOR, (center_x, center_y), radius)
        elif player == PLAYER_O:
            pygame.draw.circle(self.screen, PLAYER_O_COLOR, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, PLAYER_O_BORDER, (center_x, center_y), radius, 2)

    def draw_inputs(self):
        row_label = self.font.render("Linha:", True, BLACK)
        col_label = self.font.render("Coluna:", True, BLACK)
        self.screen.blit(row_label, (50, self.HEIGHT - 100))
        self.screen.blit(col_label, (200, self.HEIGHT - 100))
     
        self._draw_input_field(self.row_input_rect, self.input_row, self.active_input == 'row')
        self._draw_input_field(self.col_input_rect, self.input_col, self.active_input == 'col')
      
        self._draw_button(self.button_rect, "Jogar", BUTTON_COLOR)
        self._draw_button(self.reset_button_rect, "Reset", ERROR)

    def _draw_input_field(self, rect, text, is_active):
        color = INPUT_ACTIVE if is_active else INPUT_INACTIVE
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, INPUT_BORDER, rect, 2)
        
        text_surface = self.font.render(text, True, BLACK)
        self.screen.blit(text_surface, (rect.x + 5, rect.y + 5))

    def _draw_button(self, rect, text, color):
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, BLACK, rect, 2)
        
        text_surface = self.font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw_info(self):
        # Título
        title = self.title_font.render("GOMOKU - 5 em Linha", True, BLACK)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 25))
        self.screen.blit(title, title_rect)
        
        # Status do jogo
        self._draw_game_status()
        
        # Utilidades
        self._draw_utilities()
        
        # Mensagem de erro
        if self.show_error:
            error_surface = self.font.render(self.error_message, True, ERROR)
            self.screen.blit(error_surface, (50, self.HEIGHT - 40))

    def _draw_game_status(self):
        if not self.game.game_over:
            status_text = f"Jogador atual: {self.game.current_player}"
            color = PLAYER_X_COLOR if self.game.current_player == PLAYER_X else BLUE
        else:
            winner = PLAYER_O if self.game.current_player == PLAYER_X else PLAYER_X
            status_text = f"Jogador {winner} venceu! 🎉"
            color = SUCCESS
            
        status_surface = self.font.render(status_text, True, color)
        self.screen.blit(status_surface, (10, 50))

    def _draw_utilities(self):
        util_x = self.game.calcular_utilidade(PLAYER_X)
        util_o = self.game.calcular_utilidade(PLAYER_O)
        
        util_text_x = self.small_font.render(f"Utilidade X: {util_x}", True, BLACK)
        util_text_o = self.small_font.render(f"Utilidade O: {util_o}", True, BLUE)
        
        self.screen.blit(util_text_x, (10, 75))
        self.screen.blit(util_text_o, (10, 95))

    def handle_click(self, pos):
        if self.row_input_rect.collidepoint(pos):
            self.active_input = 'row'
            return
        elif self.col_input_rect.collidepoint(pos):
            self.active_input = 'col'
            return
        elif self.button_rect.collidepoint(pos):
            self.try_make_move()
            return
        elif self.reset_button_rect.collidepoint(pos):
            self.reset_game()
            return
        else:
            self.active_input = None
        
        self._handle_board_click(pos)

    def _handle_board_click(self, pos):
        x, y = pos
        
        if (self.MARGIN <= x <= self.MARGIN + BOARD_SIZE * self.CELL_SIZE and
            self.MARGIN <= y <= self.MARGIN + BOARD_SIZE * self.CELL_SIZE):
            
            col = (x - self.MARGIN) // self.CELL_SIZE
            row = (y - self.MARGIN) // self.CELL_SIZE
            
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                self.make_move(row, col)

    def try_make_move(self):
        try:
            if self.input_row == "" or self.input_col == "":
                self._show_error("Preencha linha e coluna!")
                return
                
            row = int(self.input_row)
            col = int(self.input_col)
            
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                self._show_error(f"Coordenadas devem ser entre 0 e {BOARD_SIZE-1}!")
                return
                
            if self.make_move(row, col):
                self._clear_inputs()
                
        except ValueError:
            self._show_error("Digite apenas números!")

    def make_move(self, row, col):
        if self.game.game_over:
            return False
            
        if not self.game.acao_valida(row, col):
            self._show_error("Posição já ocupada!")
            return False
        
        jogador_atual = self.game.current_player
        novo_estado = self.game.fazer_jogada(row, col)
        
        if novo_estado:
            self.game._board = novo_estado._board
            self.game._current_player = novo_estado._current_player
            
            if self.game.verificar_vitoria(row, col):
                self.game._game_over = True
            
            self.show_error = False
            return True
        
        return False

    def reset_game(self):
        self.game = Gomoku()
        self._clear_inputs()
        self.show_error = False

    def _clear_inputs(self):
        self.input_row = ""
        self.input_col = ""
        self.show_error = False

    def _show_error(self, message):
        self.show_error = True
        self.error_message = message

    def handle_keydown(self, event):
        
        if self.active_input is None:
            return
            
        if event.key == pygame.K_BACKSPACE:
            if self.active_input == 'row':
                self.input_row = self.input_row[:-1]
            elif self.active_input == 'col':
                self.input_col = self.input_col[:-1]
        elif event.key == pygame.K_RETURN:
            self.try_make_move()
        elif event.key == pygame.K_TAB:
            # Alterna entre os campos
            self.active_input = 'col' if self.active_input == 'row' else 'row'
        elif event.unicode.isdigit():
            self._handle_digit_input(event.unicode)

    def _handle_digit_input(self, digit):
        if self.active_input == 'row' and len(self.input_row) < 2:
            self.input_row += digit
        elif self.active_input == 'col' and len(self.input_col) < 2:
            self.input_col += digit

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Processa eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)
            
            # Desenha tudo
            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_inputs()
            self.draw_info()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()
