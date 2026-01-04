import random
from go_game import GoGame

class RandomAI:
    def __init__(self, color):
        self.color = color

    def get_move(self, game):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)

def main():
    game = GoGame(size=9)
    ai = RandomAI(GoGame.WHITE)
    
    print("Welcome to Go (Jungo rules)!")
    print("You are Black (X), AI is White (O).")
    
    while not game.is_over():
        print("\nCurrent board:")
        print(game)
        
        if game.current_player == GoGame.BLACK:
            move_str = input("Your move (r c) or 'p' to pass: ").strip().lower()
            if move_str == 'p':
                game.play(None, None)
            else:
                try:
                    r, c = map(int, move_str.split())
                    if not game.play(r - 1, c - 1):
                        print("Invalid move, try again.")
                except ValueError:
                    print("Invalid input, enter 'r c' or 'p'.")
        else:
            print("AI is thinking...")
            move = ai.get_move(game)
            if move is None:
                game.play(None, None)
                print("AI passes.")
            else:
                r, c = move
                game.play(r, c)
                if r is None:
                    print("AI passes.")
                else:
                    print(f"AI plays at {r + 1} {c + 1}")
                    
    print("\nGame over!")
    print(game)
    b_score, w_score = game.score()
    print(f"Final Score - Black: {b_score}, White: {w_score}")
    if b_score > w_score:
        print("Black wins!")
    elif w_score > b_score:
        print("White wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
