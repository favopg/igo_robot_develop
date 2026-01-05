import numpy as np

class GoGame:
    BLACK = 1
    WHITE = -1
    EMPTY = 0

    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = self.BLACK
        self.ko_square = None
        self.pass_count = 0
        self.history = []
        self.resigned_player = None

    def get_liberties(self, r, c):
        color = self.board[r, c]
        if color == self.EMPTY:
            return set()
        
        group = set()
        liberties = set()
        stack = [(r, c)]
        
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) in group:
                continue
            group.add((curr_r, curr_c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.board[nr, nc] == self.EMPTY:
                        liberties.add((nr, nc))
                    elif self.board[nr, nc] == color:
                        stack.append((nr, nc))
        return liberties, group

    def play(self, r, c):
        if r is None and c is None:  # Pass
            self.pass_count += 1
            self.history.append((self.current_player, None))
            self.current_player *= -1
            self.ko_square = None
            return True

        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.board[r, c] != self.EMPTY:
            return False
        if (r, c) == self.ko_square:
            return False

        # Place stone temporarily
        self.board[r, c] = self.current_player
        
        # Check for captured stones
        captured_any = False
        captured_groups = []
        opponent = self.current_player * -1
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == opponent:
                libs, group = self.get_liberties(nr, nc)
                if not libs:
                    captured_groups.append(group)
        
        for group in captured_groups:
            for gr, gc in group:
                self.board[gr, gc] = self.EMPTY
            captured_any = True

        # Check for suicide
        libs, group = self.get_liberties(r, c)
        if not libs:
            # Revert
            self.board[r, c] = self.EMPTY
            return False

        # KO handling
        # If one stone was captured and the placed stone now has exactly 1 liberty and is alone
        # Simplified KO: check if the board state repeats. For basic Go, we often just check immediate KO.
        # But let's do a simple one-stone KO check.
        new_ko_square = None
        if captured_any and len(captured_groups) == 1 and len(captured_groups[0]) == 1:
            if len(group) == 1:
                # Potential KO
                new_ko_square = list(captured_groups[0])[0]

        self.ko_square = new_ko_square
        self.pass_count = 0
        self.history.append((self.current_player, (r, c)))
        self.current_player *= -1
        return True

    def resign(self, player):
        self.resigned_player = player
        return True

    def is_over(self):
        return self.pass_count >= 2 or self.resigned_player is not None

    def score(self):
        if self.resigned_player is not None:
            # 投了した場合、投了した側の石数を0、相手を1（または適当な勝利判定可能な値）にするか
            # あるいはスコア計算自体は通常通り行い、UI側で判定する。
            # 今回は純碁ルール（石数）なので、投了した場合は相手の勝ちであることを明確にするため
            # 極端なスコアを返すか、別途勝者判定を設けるのが良い。
            # ここではスコア計算はそのままにし、is_overとresigned_playerで判定することにする。
            pass
        # Jungo rules: count stones on board
        black_stones = np.sum(self.board == self.BLACK)
        white_stones = np.sum(self.board == self.WHITE)
        return black_stones, white_stones

    def get_valid_moves(self):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                # We need to simulate the move to check if it's valid (suicide/ko)
                # But for performance, we can skip occupied squares and KO square
                if self.board[r, c] == self.EMPTY and (r, c) != self.ko_square:
                    # Temporary play
                    orig_board = self.board.copy()
                    if self.play_temp(r, c):
                        moves.append((r, c))
                    self.board = orig_board
        moves.append(None) # Pass
        return moves

    def play_temp(self, r, c):
        # Simplified play check for valid moves
        self.board[r, c] = self.current_player
        opponent = self.current_player * -1
        captured = False
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == opponent:
                libs, _ = self.get_liberties(nr, nc)
                if not libs:
                    captured = True
                    break
        
        if not captured:
            libs, _ = self.get_liberties(r, c)
            if not libs:
                return False
        return True

    def copy(self):
        new_game = GoGame(size=self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.ko_square = self.ko_square
        new_game.pass_count = self.pass_count
        new_game.history = list(self.history)
        new_game.resigned_player = self.resigned_player
        return new_game

    def __str__(self):
        res = "  " + " ".join(str(i + 1) for i in range(self.size)) + "\n"
        for r in range(self.size):
            res += str(r + 1) + " "
            for c in range(self.size):
                if self.board[r, c] == self.BLACK:
                    res += "X "
                elif self.board[r, c] == self.WHITE:
                    res += "O "
                else:
                    res += ". "
            res += "\n"
        return res
