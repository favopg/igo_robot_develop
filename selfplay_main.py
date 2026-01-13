import os
import time
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from sgfmill import sgf, boards
from go_game import GoGame

# --- モデル定義 (app.py からコピー) ---

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class SimpleGoNet(nn.Module):
    def __init__(self, size=9, channels=32, num_blocks=10):
        super(SimpleGoNet, self).__init__()
        self.size = size
        self.channels = channels
        self.num_blocks = num_blocks
        
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * size * size, size * size + 1)
        
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(size * size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        
        p = self.policy_conv(x)
        p = p.view(-1, 2 * self.size * self.size)
        p = self.policy_fc(p)
        
        v = F.relu(self.value_conv(x))
        v = v.view(-1, self.size * self.size)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

# --- MCTS ロジック (app.py からコピー) ---

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

    def expand(self, action_probs):
        for move, prob in action_probs:
            if move not in self.children:
                new_game = self.game.copy()
                new_game.play(move[0], move[1]) if move else new_game.play(None, None)
                self.children[move] = MCTSNode(new_game, parent=self, move=move)
                self.children[move].prior = prob

    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            u = c_puct * child.prior * (np.sqrt(self.visit_count) / (1 + child.visit_count))
            score = (child.value_sum / child.visit_count if child.visit_count > 0 else 0) + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.update(-value)

def board_to_tensor(board_obj, color):
    size = board_obj.side
    tensor = np.zeros((3, size, size), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            p = board_obj.get(r, c)
            if p == color:
                tensor[0, r, c] = 1.0
            elif p is not None:
                tensor[1, r, c] = 1.0
    if color == 'b':
        tensor[2, :, :] = 1.0
    return tensor

def game_to_sgfmill_board(game):
    board = boards.Board(game.size)
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] == GoGame.BLACK:
                board.play(r, c, 'b')
            elif game.board[r, c] == GoGame.WHITE:
                board.play(r, c, 'w')
    return board

def run_mcts(game, model, device, num_simulations=800, is_selfplay=False):
    root = MCTSNode(game)
    
    # ② ディリクレ・ノイズの追加 (自戦対局時のみ)
    # ルートノイズとしてポリシーにわずかなランダムノイズを加える
    if is_selfplay and not game.is_over():
        color_char = 'b' if game.current_player == GoGame.BLACK else 'w'
        btensor = board_to_tensor(game_to_sgfmill_board(game), color_char)
        input_tensor = torch.tensor(np.array([btensor]), dtype=torch.float32).to(device)
        with torch.no_grad():
            policy_logits, _ = model(input_tensor)
            probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        valid_moves = game.get_valid_moves()
        action_probs = []
        
        # ディリクレノイズの生成
        epsilon = 0.25
        alpha = 0.03 # 9x9盤面向けの小さな値
        noise = np.random.dirichlet([alpha] * len(valid_moves))
        
        for i, mv in enumerate(valid_moves):
            idx = 81 if mv is None else mv[0] * 9 + mv[1]
            p = (1 - epsilon) * probs[idx] + epsilon * noise[i]
            action_probs.append((mv, p))
        root.expand(action_probs)

    for _ in range(num_simulations):
        node = root
        while node.children:
            move, node = node.select_child()
        
        if not node.game.is_over():
            color_char = 'b' if node.game.current_player == GoGame.BLACK else 'w'
            btensor = board_to_tensor(game_to_sgfmill_board(node.game), color_char)
            input_tensor = torch.tensor(np.array([btensor]), dtype=torch.float32).to(device)
            with torch.no_grad():
                policy_logits, value = model(input_tensor)
                probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
                v = value.item()
            valid_moves = node.game.get_valid_moves()
            action_probs = []
            for mv in valid_moves:
                idx = 81 if mv is None else mv[0] * 9 + mv[1]
                action_probs.append((mv, probs[idx]))
            node.expand(action_probs)
        else:
            b, w = node.game.score()
            if node.game.current_player == GoGame.BLACK:
                v = 1.0 if b > w else (-1.0 if w > b else 0)
            else:
                v = 1.0 if w > b else (-1.0 if b > w else 0)
        node.update(v)
    
    # ① ソフトマックス・サンプリングの導入
    # 自戦対局かつ序盤（30手まで）は探索回数に基づいた確率的な選択を行う
    if is_selfplay and len(game.history) < 30:
        items = list(root.children.items())
        moves = [item[0] for item in items]
        visits = np.array([item[1].visit_count for item in items], dtype=np.float32)
        
        # 温度パラメータ（1.0に設定して訪問回数に比例させる）
        # 合計が0にならないように微小値を加算
        probs = visits / np.sum(visits)
        
        # 確率に基づいて手を選択
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]
    else:
        # 通常対局や中盤以降は最も訪問回数が多い手を選択
        return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

def self_play_worker(model_state, num_games, worker_id, selfplay_dir):
    import torch
    # 各プロセスでシード値を変更して多様性を確保
    np.random.seed(int((time.time() * 1000) % 2**32) + worker_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGoNet(size=9).to(device)
    model.load_state_dict(model_state)
    model.eval()

    results = []
    for i in range(num_games):
        game_obj = GoGame(size=9)
        sgf_game = sgf.Sgf_game(size=9)
        while not game_obj.is_over():
            color_char = 'b' if game_obj.current_player == GoGame.BLACK else 'w'
            
            # ③ 初手のランダム化
            # 最初の数手（例：2手まで）は完全にランダムまたは有力候補からランダムに選ぶ
            # ここではシンプルに、3手目まではMCTSではなくランダムに有効な手から選ぶ
            if len(game_obj.history) < 3:
                valid_moves = game_obj.get_valid_moves()
                # パスは除外（序盤なので）
                non_pass_moves = [m for m in valid_moves if m is not None]
                if non_pass_moves:
                    move = non_pass_moves[np.random.choice(len(non_pass_moves))]
                else:
                    move = None
            else:
                move = run_mcts(game_obj, model, device, num_simulations=800, is_selfplay=True)
            
            game_obj.play(move[0], move[1]) if move else game_obj.play(None, None)
            node = sgf_game.extend_main_sequence()
            node.set_move(color_char, move)
            if len(game_obj.history) > 100:
                break
        
        # 勝敗結果をSGFに保存
        b_score, w_score = game_obj.score()
        if b_score > w_score:
            sgf_game.root.set("RE", "B+" + str(b_score - w_score))
        elif w_score > b_score:
            sgf_game.root.set("RE", "W+" + str(w_score - b_score))
        else:
            sgf_game.root.set("RE", "Draw")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selfplay_{timestamp}_w{worker_id}_{i}.sgf"
        save_path = os.path.join(selfplay_dir, filename)
        with open(save_path, "wb") as f:
            f.write(sgf_game.serialise())
        results.append(save_path)
    return results

def start_selfplay(model_path, num_games=100, num_processes=4, selfplay_dir="selfplay_sgf"):
    if not os.path.exists(selfplay_dir):
        os.makedirs(selfplay_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGoNet(size=9).to(device)
    
    # .txt.gz の場合は .pt を探す (app.pyのロジックを流用)
    if model_path.endswith(".txt.gz"):
        pt_path = model_path.replace(".txt.gz", ".pt")
        if os.path.exists(pt_path):
            model_path = pt_path
        else:
            raise ValueError(f"Could not find .pt file for model: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    games_per_process = num_games // num_processes
    remaining_games = num_games % num_processes

    tasks = []
    for i in range(num_processes):
        n = games_per_process + (1 if i < remaining_games else 0)
        if n > 0:
            tasks.append((model_state, n, i, selfplay_dir))

    print(f"Starting parallel self-play with {num_processes} processes for {num_games} games...")
    print(f"Model: {model_path}")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(self_play_worker, tasks)
    
    print("Parallel self-play completed.")
    return [item for sublist in results for item in sublist]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone Self-play script for Go AI")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pt model file")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--outdir", type=str, default="selfplay_sgf", help="Output directory for SGF files")
    
    args = parser.parse_args()
    start_selfplay(args.model, args.games, args.processes, args.outdir)
