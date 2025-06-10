
from gui.game_gui import GomokuGUI

def main():
    print("=== Gomoku (5 em linha) ===")
    print("Iniciando interface gráfica...")
    
    game = GomokuGUI()
    game.run()

if __name__ == "__main__":
    main()