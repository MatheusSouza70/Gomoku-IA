# Importações necessárias
import random
import pickle  # Para salvar/carregar a Q-table
import math  # Para cálculos matemáticos (infinito, etc.)
import os  # Para manipulação de arquivos e diretórios
import threading  # Para salvar a Q-table em segundo plano

import numpy as np  # Para operações numéricas


class QLearningAgent:
    """Agente de Q-Learning para jogar Gomoku."""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.9995, min_epsilon=0.1):
        """Inicializa o agente com os parâmetros de aprendizado.
        
        Args:
            alpha (float): Taxa de aprendizado (default: 0.1).
            gamma (float): Fator de desconto (default: 0.9).
            epsilon (float): Probabilidade inicial de exploração (default: 1.0).
            decay (float): Decaimento de epsilon (default: 0.9995).
            min_epsilon (float): Valor mínimo de epsilon (default: 0.1).
        """
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto (importância de recompensas futuras)
        self.epsilon = epsilon  # Probabilidade de explorar (vs. exploitar)
        self.decay = decay  # Decaimento de epsilon após cada episódio
        self.min_epsilon = min_epsilon  # Valor mínimo de epsilon
        self.q_table = {}  # Tabela Q (dicionário: (estado, ação) → valor Q)

    def get_state_key(self, board):
        """Converte o tabuleiro em uma string única para usar como chave na Q-table.
        
        Args:
            board (list): Matriz representando o tabuleiro.
        
        Returns:
            str: String concatenada representando o estado.
        """
        return ''.join(cell for row in board for cell in row)

    def available_actions(self, board):
        """Retorna todas as ações possíveis (posições vazias) no tabuleiro.
        
        Args:
            board (list): Matriz do tabuleiro.
        
        Returns:
            list: Lista de tuplas (i, j) representando posições livres.
        """
        return [(i, j) for i in range(len(board)) for j in range(len(board[0])) if board[i][j] == '.']

    def choose_action(self, env, explore=True):
        """Escolhe uma ação usando ε-greedy ou softmax.
        
        Args:
            env (Gomoku): Ambiente do jogo.
            explore (bool): Se True, permite exploração (default: True).
        
        Returns:
            tuple: Ação escolhida (i, j).
        """
        moves = env.available_moves()  # Obtém jogadas válidas
        state = self.get_state_key(env._board)  # Obtém estado atual

        # Exploração: escolhe ação baseada em softmax (probabilística)
        if explore and np.random.rand() < self.epsilon:
            q_vals = np.array([self.q_table.get((state, a), 0.0) for a in moves])
            probs = self.softmax(q_vals)
            return moves[np.random.choice(len(moves), p=probs)]

        # Exploitation: escolhe ação com maior Q-value (greedy)
        q_vals = {a: self.q_table.get((state, a), 0.0) for a in moves}
        max_q = max(q_vals.values(), default=0.0)
        best = [a for a, v in q_vals.items() if v == max_q]
        return best[np.random.randint(len(best))] if best else moves[np.random.randint(len(moves))]

    def softmax(self, q_vals, tau=1.0):
        """Calcula probabilidades usando softmax.
        
        Args:
            q_vals (np.array): Valores Q das ações.
            tau (float): Temperatura (controla aleatoriedade).
        
        Returns:
            np.array: Probabilidades normalizadas.
        """
        q_vals = np.array(q_vals)
        exp_q = np.exp((q_vals - np.max(q_vals)) / tau)  # Evita overflow
        return exp_q / np.sum(exp_q)

    def learn(self, prev_board, action, reward, next_board, done, env):
        """Atualiza a Q-table com a experiência (s, a, r, s').
        
        Args:
            prev_board (list): Tabuleiro anterior.
            action (tuple): Ação tomada.
            reward (float): Recompensa recebida.
            next_board (list): Novo tabuleiro.
            done (bool): Se o episódio terminou.
            env (Gomoku): Ambiente do jogo.
        """
        s = self.get_state_key(prev_board)
        ns = self.get_state_key(next_board)
        old_q = self.q_table.get((s, action), 0.0)  # Valor Q antigo
        next_moves = self.available_actions(next_board)
        future = 0.0

        # Calcula o máximo Q-value futuro (se não for estado terminal)
        if not done and next_moves:
            future = max(self.q_table.get((ns, a), 0.0) for a in env.available_moves())

        # Atualiza Q-value usando a equação de Bellman
        self.q_table[(s, action)] = old_q + self.alpha * (reward + self.gamma * future - old_q)

        # Decai epsilon se o episódio terminou
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def save(self, path='qtable.pkl'):
        """Salva a Q-table em um arquivo (assíncrono)."""
        table_copy = dict(self.q_table)  # Cria cópia para evitar problemas de thread

        def _save(snapshot, save_path):
            """Função interna para salvar em segundo plano."""
            dir_path = os.path.dirname(save_path)
            if dir_path:  # só cria diretório se um caminho for especificadoa
                os.makedirs(dir_path, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(snapshot, f)
            print(f"Q-table salva em {save_path} (entradas: {len(snapshot)})")

        # Usa threading para salvar sem bloquear a execução
        threading.Thread(target=_save, args=(table_copy, path), daemon=True).start()

    def load(self, path='qtable.pkl'):
        """Carrega a Q-table de um arquivo."""
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table carregado de {path} (entradas: {len(self.q_table)})")


class Gomoku:
    """Classe que representa o jogo Gomoku (5 em linha)."""

    def __init__(self, size=7):
        """Inicializa o tabuleiro vazio.
        
        Args:
            size (int): Tamanho do tabuleiro (default: 7x7).
        """
        self.size = size
        self._board = [['.' for _ in range(size)] for _ in range(size)]  # Tabuleiro vazio

    def reset(self):
        """Reinicia o tabuleiro."""
        self._board = [['.' for _ in range(self.size)] for _ in range(self.size)]

    def imprimir_tabuleiro(self):
        """Imprime o tabuleiro formatado."""
        print("\n  " + " ".join(f"{i:2}" for i in range(self.size)))  # Cabeçalho colunas
        for i, row in enumerate(self._board):
            print(f"{i:2} " + " ".join(row))  # Linhas com índices

    def available_moves(self):
        """Retorna todas as jogadas válidas (posições vazias)."""
        return [(i, j) for i in range(self.size) for j in range(self.size) if self._board[i][j] == '.']

    def fazer_jogada(self, i, j, player):
        """Executa uma jogada no tabuleiro.
        
        Args:
            i (int): Linha.
            j (int): Coluna.
            player (str): 'X' ou 'O'.
        """
        self._board[i][j] = player

    def verificar_vitoria(self, i, j):
        """Verifica se a última jogada resultou em vitória.
        
        Args:
            i (int): Linha da jogada.
            j (int): Coluna da jogada.
        
        Returns:
            bool: True se houve vitória.
        """
        p = self._board[i][j]
        # Verifica 4 direções: horizontal, vertical, 2 diagonais
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            cnt = 1  # Conta a peça atual
            for d in (1, -1):  # Verifica ambas direções
                x, y = i + dr * d, j + dc * d
                while 0 <= x < self.size and 0 <= y < self.size and self._board[x][y] == p:
                    cnt += 1
                    x += dr * d
                    y += dc * d
            if cnt >= 5:  # 5 em linha
                return True
        return False

    def minimax(self, depth, maximizing, alpha=-math.inf, beta=math.inf):
        """Implementa o algoritmo Minimax com poda alpha-beta.
        
        Args:
            depth (int): Profundidade máxima de busca.
            maximizing (bool): True se é o turno do maximizador ('O').
            alpha (float): Valor alpha para poda.
            beta (float): Valor beta para poda.
        
        Returns:
            tuple: (valor_heurístico, melhor_jogada)
        """
        def evaluate():
            """Função de avaliação heurística simples."""
            return sum(self._board[r][c] == 'O' for r in range(self.size) for c in range(self.size)) - \
                   sum(self._board[r][c] == 'X' for r in range(self.size) for c in range(self.size))

        if depth == 0:  # Folha da árvore de busca
            return evaluate(), None

        best_move = None
        if maximizing:  # Turno do maximizador ('O')
            max_eval = -math.inf
            for i, j in self.available_moves():
                self._board[i][j] = 'O'
                if self.verificar_vitoria(i, j):  # Vitória instantânea
                    self._board[i][j] = '.'
                    return math.inf, (i, j)
                eval, _ = self.minimax(depth - 1, False, alpha, beta)
                self._board[i][j] = '.'  # Desfaz jogada
                if eval > max_eval:
                    max_eval, best_move = eval, (i, j)
                alpha = max(alpha, eval)
                if beta <= alpha:  # Poda beta
                    break
            return max_eval, best_move
        else:  # Turno do minimizador ('X')
            min_eval = math.inf
            for i, j in self.available_moves():
                self._board[i][j] = 'X'
                if self.verificar_vitoria(i, j):  # Derrota instantânea
                    self._board[i][j] = '.'
                    return -math.inf, (i, j)
                eval, _ = self.minimax(depth - 1, True, alpha, beta)
                self._board[i][j] = '.'  # Desfaz jogada
                if eval < min_eval:
                    min_eval, best_move = eval, (i, j)
                beta = min(beta, eval)
                if beta <= alpha:  # Poda alpha
                    break
            return min_eval, best_move

    def train(self, agent, episodes=10000, report_interval=1000, checkpoint_interval=1000):
        """Treina o agente contra Minimax.
        
        Args:
            agent (QLearningAgent): Agente a ser treinado.
            episodes (int): Número de episódios de treino.
            report_interval (int): Frequência de relatórios.
            checkpoint_interval (int): Frequência de salvamento.
        """
        win_q = win_mm = ties = 0  # Contadores
        logs = []  # Armazena histórico para gráficos

        for ep in range(1, episodes + 1):
            self.reset()
            prev = [row[:] for row in self._board]  # Copia o tabuleiro
            done = False

            while not done:
                # Agente Q-Learning joga ('O')
                act = agent.choose_action(self, explore=True)
                self.fazer_jogada(*act, 'O')

                # Verifica vitória do Q-Learning
                if self.verificar_vitoria(*act):
                    agent.learn(prev, act, +100, self._board, True, self)
                    win_q += 1
                    done = True
                    break

                # Minimax joga ('X')
                _, ua = self.minimax(1, False)
                if not ua:  # Empate
                    agent.learn(prev, act, 0, self._board, True, self)
                    ties += 1
                    done = True
                    break
                self.fazer_jogada(*ua, 'X')

                # Verifica vitória do Minimax
                if self.verificar_vitoria(*ua):
                    agent.learn(prev, act, -200, self._board, True, self)
                    win_mm += 1
                    done = True
                    break

                # Atualiza Q-value para passo não-terminal
                agent.learn(prev, act, -0.1, self._board, False, self)
                prev = [row[:] for row in self._board]  # Atualiza estado anterior

            # Armazena logs para gráficos
            logs.append((win_q, win_mm, ties, agent.epsilon))

            # Relatório periódico
            if ep % report_interval == 0:
                print(f"E{ep}: Q={win_q}, MM={win_mm}, ties={ties}, eps={agent.epsilon:.3f}")

            # Checkpoint periódico
            if ep % checkpoint_interval == 0:
                agent.save()
        print("Treino completo.")

    def evaluate_agent(self, agent, games=100):
        """Avalia o agente treinado contra Minimax.
        
        Args:
            agent (QLearningAgent): Agente a ser avaliado.
            games (int): Número de partidas de avaliação.
        """
        wq = wm = t = 0  # Vitórias Q, vitórias Minimax, empates

        for _ in range(games):
            self.reset()
            turn = 0
            done = False

            while not done:
                if turn % 2 == 0:  # Agente Q joga ('O')
                    a = agent.choose_action(self, explore=False)
                    self.fazer_jogada(*a, 'O')
                    if self.verificar_vitoria(*a):
                        wq += 1
                        done = True
                        break
                else:  # Minimax joga ('X')
                    _, a = self.minimax(2, False)
                    if not a:  # Empate
                        t += 1
                        done = True
                        break
                    self.fazer_jogada(*a, 'X')
                    if self.verificar_vitoria(*a):
                        wm += 1
                        done = True
                        break
                turn += 1

            if not done:  # Empate por tabuleiro cheio
                t += 1

        print(f"Aval: Q={wq}/{games}, MM={wm}/{games}, ties={t}/{games}")

    def play_human_vs_human(self):
        """Modo humano vs humano."""
        self.reset()
        turn = 0

        while True:
            self.imprimir_tabuleiro()
            p = 'X' if turn % 2 == 0 else 'O'
            r = int(input("Linha: "))
            c = int(input("Coluna: "))

            if (r, c) not in self.available_moves():
                continue

            self.fazer_jogada(r, c, p)

            if self.verificar_vitoria(r, c):
                self.imprimir_tabuleiro()
                print(f"{p} venceu!")
                break

            turn += 1

            if not self.available_moves():
                print("Empate!")
                break

    def play_human_vs_minimax(self, depth=2):
        """Modo humano vs Minimax."""
        self.reset()
        turn = 0

        while True:
            self.imprimir_tabuleiro()

            if turn % 2 == 0:  # Humano ('X')
                r = int(input("Linha: "))
                c = int(input("Coluna: "))

                if (r, c) not in self.available_moves():
                    continue

                self.fazer_jogada(r, c, 'X')

                if self.verificar_vitoria(r, c):
                    self.imprimir_tabuleiro()
                    print("Você venceu!")
                    break
            else:  # Minimax ('O')
                _, m = self.minimax(depth, True)
                print(f"Minimax joga {m}")
                self.fazer_jogada(*m, 'O')

                if self.verificar_vitoria(*m):
                    self.imprimir_tabuleiro()
                    print("Minimax venceu!")
                    break

            turn += 1

            if not self.available_moves():
                print("Empate!")
                break

    def play_human_vs_q(self, agent):
        """Modo humano vs Q-Learning."""
        agent.load()  # Carrega Q-table treinada
        self.reset()
        turn = 0

        while True:
            self.imprimir_tabuleiro()

            if turn % 2 == 0:  # Humano ('X')
                r = int(input("Linha: "))
                c = int(input("Coluna: "))

                if (r, c) not in self.available_moves():
                    continue

                self.fazer_jogada(r, c, 'X')

                if self.verificar_vitoria(r, c):
                    self.imprimir_tabuleiro()
                    print("Você venceu!")
                    break
            else:  # Q-Learning ('O')
                m = agent.choose_action(self, explore=False)
                print(f"Q joga {m}")
                self.fazer_jogada(*m, 'O')

                if self.verificar_vitoria(*m):
                    self.imprimir_tabuleiro()
                    print("Q-Agent venceu!")
                    break

            turn += 1

            if not self.available_moves():
                print("Empate!")
                break

    def play_minimax_vs_q(self, agent, depth=2):
        """Modo Minimax vs Q-Learning (para testes)."""
        agent.load()  # Carrega Q-table treinada
        self.reset()
        turn = 0

        while True:
            self.imprimir_tabuleiro()

            if turn % 2 == 0:  # Minimax ('X')
                _, m = self.minimax(depth, True)
                print(f"Minimax joga {m}")
                self.fazer_jogada(*m, 'X')

                if self.verificar_vitoria(*m):
                    self.imprimir_tabuleiro()
                    print("Minimax venceu!")
                    break
            else:  # Q-Learning ('O')
                m = agent.choose_action(self, explore=False)
                print(f"Q joga {m}")
                self.fazer_jogada(*m, 'O')

                if self.verificar_vitoria(*m):
                    self.imprimir_tabuleiro()
                    print("Q-Agent venceu!")
                    break

            turn += 1

            if not self.available_moves():
                print("Empate!")
                break


if __name__ == "__main__":
    # Inicializa agente e jogo
    agent = QLearningAgent()
    jogo = Gomoku(size=5)  # Tabuleiro 5x5 (pode ser ajustado)

    # Menu interativo
    while True:
        print("\n=== Menu ===")
        print("1: Humano vs Humano")
        print("2: Humano vs Minimax")
        print("3: Humano vs Q-Learning")
        print("4: Minimax vs Q-Learning")
        print("5: Treinar Q-Learning")
        print("0: Sair")
        opt = input("Opcao: ")

        if opt == '1':
            jogo.play_human_vs_human()
        elif opt == '2':
            jogo.play_human_vs_minimax(depth=2)
        elif opt == '3':
            jogo.play_human_vs_q(agent)
        elif opt == '4':
            jogo.play_minimax_vs_q(agent, depth=2)
        elif opt == '5':
            # Carrega Q-table existente (se houver)
            qpath = os.path.join('qtable.pkl')
            if os.path.exists(qpath):
                agent.load(qpath)
            else:
                print("Nenhuma Q-table existente encontrada. Iniciando treino do zero.")
            jogo.train(agent, episodes=10000)
            agent.save(qpath)
        elif opt == '0':
            break
        else:
            print("Opção inválida")