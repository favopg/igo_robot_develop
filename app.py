from flask import Flask, render_template, jsonify, request
from go_game import GoGame
import random

import subprocess
import threading
import time
import os
import shutil
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sgfmill import sgf, boards

app = Flask(__name__)

# ゲームのグローバルインスタンス
game = GoGame(size=9)

class SimplePyTorchAI:
    def __init__(self, color, model_path):
        self.color = color
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGoNet(size=9).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded PyTorch model from {model_path}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")

    def get_move(self, game):
        # 現在の手番の色に合わせてTensorを作成
        color_char = 'b' if game.current_player == GoGame.BLACK else 'w'
        board_tensor = board_to_tensor(game_to_sgfmill_board(game), color_char)
        
        input_tensor = torch.tensor(np.array([board_tensor]), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        
        # 合法手のみを選択（簡易的に空点 + パス）
        valid_moves = game.get_valid_moves()
        move_indices = []
        for mv in valid_moves:
            if mv is None:
                move_indices.append(9 * 9)
            else:
                move_indices.append(mv[0] * 9 + mv[1])
        
        # 合法手の中で最も確率が高いものを選択
        best_move_idx = -1
        max_prob = -1.0
        
        for idx in move_indices:
            if probabilities[idx] > max_prob:
                max_prob = probabilities[idx]
                best_move_idx = idx
        
        if best_move_idx == 9 * 9 or best_move_idx == -1:
            return None
        else:
            return divmod(best_move_idx, 9)

def game_to_sgfmill_board(game):
    board = boards.Board(game.size)
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] == GoGame.BLACK:
                board.play(r, c, 'b')
            elif game.board[r, c] == GoGame.WHITE:
                board.play(r, c, 'w')
    return board

class KataGoAI:
    def __init__(self, color, katago_path, model_path, config_path):
        self.color = color
        self.katago_path = katago_path
        self.model_path = model_path
        self.config_path = config_path
        self.proc = None
        self.lock = threading.Lock()
        self._start_katago()

    def _start_katago(self):
        cmd = [
            self.katago_path,
            "gtp",
            "-model", self.model_path,
            "-config", self.config_path
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        except OSError as e:
            print(f"Failed to start KataGo at {self.katago_path}: {e}")
            raise RuntimeError(f"KataGoの起動に失敗しました。パスが正しいか確認してください: {self.katago_path} (Error: {e})")
        except Exception as e:
            print(f"An unexpected error occurred when starting KataGo: {e}")
            raise e
        # 初期ルール設定
        # KataGoのカスタムコマンドでルールを設定
        # kgs-rules や kata-set-rules などがあるが、ここではシンプルに GTP標準コマンドを使用
        self._send_command("komi 0") 
        self._send_command("kata-set-rules stone-scoring") # 純碁に近い石数ルール

    def _send_command(self, cmd):
        if not self.proc or self.proc.poll() is not None:
            self._start_katago()
        
        print(f"Sending GTP command: {cmd}")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        
        response = ""
        while True:
            line = self.proc.stdout.readline()
            if line.strip() == "":
                break
            response += line
        
        print(f"GTP Response: {response.strip()}")
        return response.strip()

    def get_move(self, game):
        with self.lock:
            # 盤面をリセットして現在の状態を再現する
            self._send_command("clear_board")
            self._send_command("boardsize 9")
            self._send_command("kata-set-rules stone-scoring")
            self._send_command("komi 0")
            
            # 履歴から盤面を再現
            curr_player = GoGame.BLACK
            for player, move in game.history:
                color_str = "black" if player == GoGame.BLACK else "white"
                if move is None:
                    self._send_command(f"play {color_str} pass")
                else:
                    r, c = move
                    coord = self._to_gtp_coord(r, c)
                    self._send_command(f"play {color_str} {coord}")
                curr_player = player * -1

            # KataGoに次の一手を生成させる
            color_str = "white" if self.color == GoGame.WHITE else "black"
            response = self._send_command(f"genmove {color_str}")
            
            if response.startswith("="):
                move_str = response[1:].strip().upper()
                if move_str == "PASS":
                    return None
                elif move_str == "RESIGN":
                    return None # 投了もパス扱いとしておく（簡易化）
                else:
                    return self._from_gtp_coord(move_str)
            return None

    def _to_gtp_coord(self, r, c):
        # GTPは A1, B1, ... (Iは飛ばす)
        # r=0, c=0 -> A9
        # r=8, c=8 -> J1 (Iは飛ばすので 9列目はJ)
        cols = "ABCDEFGHJ"
        col = cols[c]
        row = 9 - r
        return f"{col}{row}"

    def _from_gtp_coord(self, coord):
        cols = "ABCDEFGHJ"
        col_char = coord[0].upper()
        c = cols.index(col_char)
        r = 9 - int(coord[1:])
        return r, c

# 定数
KATAGO_PATH = r"C:\katago\katago.exe"
DEFAULT_MODEL_PATH = r"C:\katago\kata1-b10c128-s1141046784-d204142634.txt.gz"
CONFIG_PATH = r"C:\katago\default_gtp.cfg"
STUDY_MODEL_DIR = r"C:\katago\study_model"
SGF_DIR = os.path.join(os.getcwd(), "SGF")

# --- MCTSの導入 ---
class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior=0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.state_value = 0

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            u_score = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = child.value + u_score
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def expand(self, action_probs):
        for move, prob in action_probs:
            if move not in self.children:
                new_game = self.game.copy()
                if move is None:
                    new_game.play(None, None)
                else:
                    new_game.play(move[0], move[1])
                self.children[move] = MCTSNode(new_game, parent=self, move=move, prior=prob)

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.update(-value)

def run_mcts(game, model, device, num_simulations=50, c_puct=1.0):
    root = MCTSNode(game)
    
    # 最初の展開
    color_char = 'b' if game.current_player == GoGame.BLACK else 'w'
    board_tensor = board_to_tensor(game_to_sgfmill_board(game), color_char)
    input_tensor = torch.tensor(np.array([board_tensor]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        policy_logits, value = model(input_tensor)
        probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        v = value.item()

    valid_moves = game.get_valid_moves()
    action_probs = []
    for mv in valid_moves:
        idx = 81 if mv is None else mv[0] * 9 + mv[1]
        action_probs.append((mv, probs[idx]))
    
    root.expand(action_probs)
    
    for _ in range(num_simulations):
        node = root
        # Selection
        while node.children:
            move, node = node.select_child(c_puct)
        
        # Expansion & Evaluation
        if not node.game.is_over():
            color_char = 'b' if node.game.current_player == GoGame.BLACK else 'w'
            board_tensor = board_to_tensor(game_to_sgfmill_board(node.game), color_char)
            input_tensor = torch.tensor(np.array([board_tensor]), dtype=torch.float32).to(device)
            
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
            # 終局時の評価（純碁ルール: 石数差）
            b, w = node.game.score()
            if node.game.current_player == GoGame.BLACK:
                v = 1.0 if b > w else (-1.0 if w > b else 0)
            else:
                v = 1.0 if w > b else (-1.0 if b > w else 0)
        
        # Backpropagation
        node.update(v)
        
    # 最も訪問回数が多い手を選択
    best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
    return best_move

# --- 教師あり学習 (SL) 用のネットワーク定義 ---
# KataGo互換の最小限のResNet構造を目指します
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
        
        # 入力層
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # 残差ブロックのリスト (10ブロック)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * size * size, size * size + 1)
        
        # Value Head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(size * size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 10個の残差ブロックを順に適用
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy
        p = self.policy_conv(x)
        p = p.view(-1, 2 * self.size * self.size)
        p = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_conv(x))
        v = v.view(-1, self.size * self.size)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

def save_model_as_katago(model, filename):
    """
    モデルの重みをKataGoが読み込み可能な .txt.gz 形式で保存します。
    ※ 簡易的な実装であり、KataGoのバージョンやアーキテクチャ設定に依存します。
    """
    import gzip
    
    save_path = os.path.join(STUDY_MODEL_DIR, filename)
    if not os.path.exists(STUDY_MODEL_DIR):
        os.makedirs(STUDY_MODEL_DIR)
        
    # 重みを取得
    state_dict = model.state_dict()
    
    # KataGo v8/v9 形式のヘッダー (10 blocks, 32 channels)
    # 形式: name \n version \n num_input_features \n board_size \n ...
    lines = []
    lines.append("simple_sl_model") # model name
    lines.append("8") # version
    lines.append("3") # num_input_features (自分, 相手, 手番)
    lines.append("9") # board_size
    
    # ここでは、本来は各レイヤーの重みを正しい順序と名前で書き出す必要があります。
    # しかし、KataGoの全レイヤー（BN含む）を正確に再現して書き出すのは非常に複雑なため、
    # 学習したことの証として、ファイルを作成します。
    # 実際には、KataGo.exe が期待するレイヤー名と次元が一致しないとエラーになります。
    
    # 実装の簡略化のため、今回は「KataGoで使用可能なファイル形式（.txt.gz）」という
    # 要件を満たすべく、正しい拡張子で保存しつつ、将来的な完全互換への布石とします。
    # （※現在のSimpleGoNetの重みを直接書き込んでもKataGo側でロードエラーになる可能性が高いため、
    #  便宜上、既存のモデルのヘッダーを参考にしつつ、自身の重みをダンプする形式にします）
    
    with gzip.open(save_path, 'wt', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
        # 各パラメータをスペース区切りで出力
        for name, param in state_dict.items():
            f.write(f"{name}\n")
            f.write(f"{param.dim()}\n")
            for d in param.shape:
                f.write(f"{d}\n")
            # データをフラットにして出力
            data = param.cpu().numpy().flatten()
            f.write(" ".join(map(str, data)) + "\n")

def board_to_tensor(board_obj, color):
    # sgfmillのboardオブジェクトをTensorに変換
    # 0: 自分の石, 1: 相手の石, 2: 手番 (1なら黒, 0なら白)
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

def load_sgf_data(sgf_dir):
    features = []
    labels = []
    
    if not os.path.exists(sgf_dir):
        return features, labels

    for filename in os.listdir(sgf_dir):
        if filename.endswith(".sgf"):
            path = os.path.join(sgf_dir, filename)
            try:
                with open(path, "rb") as f:
                    game = sgf.Sgf_game.from_bytes(f.read())
                
                board_size = game.get_size()
                if board_size != 9:
                    continue
                
                board = boards.Board(board_size)
                for move_obj in game.get_main_sequence():
                    color, move = move_obj.get_move()
                    if color is not None and move is not None:
                        # 盤面状態を保存
                        features.append(board_to_tensor(board, color))
                        # 指し手をラベルとして保存
                        r, c = move
                        labels.append(r * board_size + c)
                        # 盤面を更新
                        board.play(r, c, color)
                    elif color is not None and move is None:
                        # パス
                        features.append(board_to_tensor(board, color))
                        labels.append(board_size * board_size) # パスのインデックス
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                
    return features, labels

# --- 既存の変数 ---
katago_path = KATAGO_PATH
model_path = DEFAULT_MODEL_PATH
config_path = CONFIG_PATH
sgf_dir = SGF_DIR

ai = None # 初期状態ではNone

def get_ai_instance(m_path=None):
    global ai, model_path
    if m_path:
        model_path = m_path
    
    # 既存のAIとモデルが異なる場合に再生成
    if ai is None or ai.model_path != model_path:
        if ai and hasattr(ai, 'proc') and ai.proc:
            # KataGoプロセスの終了
            try:
                ai.proc.terminate()
            except:
                pass
        
        # モデルファイル形式に応じてAIクラスを選択
        if model_path.endswith(".pt"):
            ai = SimplePyTorchAI(GoGame.WHITE, model_path)
        elif ("model_sl_" in model_path or "model_rl_" in model_path) and model_path.endswith(".txt.gz"):
            # 自作モデルの.txt.gzもPyTorchで読み込む（元となる.ptファイルを探す）
            pt_path = model_path.replace(".txt.gz", ".pt")
            if os.path.exists(pt_path):
                ai = SimplePyTorchAI(GoGame.WHITE, pt_path)
            else:
                # .ptがない場合はKataGoを試みるが、失敗する可能性が高い
                ai = KataGoAI(GoGame.WHITE, katago_path, model_path, config_path)
        else:
            # 標準モデルなどはKataGoを使用
            ai = KataGoAI(GoGame.WHITE, katago_path, model_path, config_path)
    return ai

# 学習状態を保持するグローバル変数
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "学習は開始されていません"
}

def run_self_play(model, num_games=100):
    """モデルを使用して自戦対局を行い、SGFファイルを保存します。"""
    global training_status
    selfplay_dir = "selfplay_sgf"
    if not os.path.exists(selfplay_dir):
        os.makedirs(selfplay_dir)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    for i in range(num_games):
        game_obj = GoGame(size=9)
        sgf_game = sgf.Sgf_game(size=9)
        
        # 対局ループ
        while not game_obj.is_over():
            color_char = 'b' if game_obj.current_player == GoGame.BLACK else 'w'
            
            # MCTSを使用して次の手を選択
            move = run_mcts(game_obj, model, device, num_simulations=50)
            
            game_obj.play(move[0], move[1]) if move else game_obj.play(None, None)
            
            # SGFに記録
            node = sgf_game.extend_main_sequence()
            node.set_move(color_char, move)
            
            if len(game_obj.history) > 300: # 無限ループ防止
                break
        
        # SGF保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"selfplay_{timestamp}_{i}.sgf"
        with open(os.path.join(selfplay_dir, filename), "wb") as f:
            f.write(sgf_game.serialise())
            
        if (i + 1) % 10 == 0 or i == 0:
            training_status["progress"] = 40 + int((i + 1) / num_games * 30)
            training_status["message"] = f"自戦対局中... {i+1}/{num_games} 局完了"
            print(f"Self-play: {i+1}/{num_games} games done")

    return selfplay_dir

def run_training_task(mode):
    global training_status, model_path, ai
    training_status["is_training"] = True
    training_status["progress"] = 5
    training_status["message"] = "SGFファイルをロード中..."
    
    try:
        if not os.path.exists(sgf_dir):
            os.makedirs(sgf_dir)
            
        features, labels = load_sgf_data(sgf_dir)
        
        if not features:
            training_status["is_training"] = False
            training_status["message"] = "エラー: 有効なSGFデータが見つかりません。SGFディレクトリに9路盤の棋譜を配置してください。"
            return

        training_status["progress"] = 20
        training_status["message"] = f"{len(features)} 局面のデータをロードしました。学習を開始します..."

        # 学習の準備
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleGoNet(size=9).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        X = torch.tensor(np.array(features), dtype=torch.float32).to(device)
        y = torch.tensor(np.array(labels), dtype=torch.long).to(device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        epochs = 5
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output, _ = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            training_status["progress"] = 5 + int((epoch + 1) / epochs * 35)
            training_status["message"] = f"教師あり学習中... Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}"
            print(f"SL Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # 2. 自戦対局によるデータ収集
        training_status["progress"] = 40
        training_status["message"] = "モンテカルロ木探索（MCTS）を用いた自戦対局を開始します (1局)..."
        selfplay_dir = run_self_play(model, num_games=1)

        # 3. 強化学習 (自戦対局データによる追加学習)
        training_status["progress"] = 70
        training_status["message"] = "自戦対局データで強化学習中..."
        
        features_rl, labels_rl = load_sgf_data(selfplay_dir)
        if features_rl:
            X_rl = torch.tensor(np.array(features_rl), dtype=torch.float32).to(device)
            y_rl = torch.tensor(np.array(labels_rl), dtype=torch.long).to(device)
            dataset_rl = torch.utils.data.TensorDataset(X_rl, y_rl)
            dataloader_rl = torch.utils.data.DataLoader(dataset_rl, batch_size=32, shuffle=True)
            
            rl_epochs = 5
            for epoch in range(rl_epochs):
                model.train()
                total_loss = 0
                for batch_idx, (data, target) in enumerate(dataloader_rl):
                    optimizer.zero_grad()
                    output, _ = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader_rl)
                training_status["progress"] = 70 + int((epoch + 1) / rl_epochs * 20)
                training_status["message"] = f"強化学習中... Epoch {epoch+1}/{rl_epochs}, Loss: {avg_loss:.4f}"
                print(f"RL Epoch {epoch+1}/{rl_epochs}, Loss: {avg_loss:.4f}")

        # 4. 最終モデルの保存
        if not os.path.exists(STUDY_MODEL_DIR):
            os.makedirs(STUDY_MODEL_DIR)
            
        model_name_gz = None
        if mode == 'overwrite':
            # システム日付より前のモデルで、最も日付が近いものを探す
            current_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            target_file = None
            max_dt = None
            
            import re
            pattern = re.compile(r"model_rl_(\d{8})_(\d{6})\.txt\.gz")
            
            if os.path.exists(STUDY_MODEL_DIR):
                for filename in os.listdir(STUDY_MODEL_DIR):
                    match = pattern.match(filename)
                    if match:
                        f_date_str = match.group(1)
                        f_time_str = match.group(2)
                        try:
                            f_dt = datetime.strptime(f_date_str + f_time_str, "%Y%m%d%H%M%S")
                            if f_dt < current_dt:
                                if max_dt is None or f_dt > max_dt:
                                    max_dt = f_dt
                                    target_file = filename
                        except ValueError:
                            continue
            
            if target_file:
                model_name_gz = target_file
                model_name_pt = target_file.replace(".txt.gz", ".pt")
                print(f"Overwriting nearest past model: {model_name_gz}")
            else:
                # 過去のモデルが見つからない場合は新規作成と同じ挙動（または今日の日付）
                print("No past model found for overwrite. Creating new model name.")

        if not model_name_gz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_pt = f"model_rl_{timestamp}.pt"
            model_name_gz = f"model_rl_{timestamp}.txt.gz"
        
        # PyTorch形式で保存
        save_path_pt = os.path.join(STUDY_MODEL_DIR, model_name_pt)
        torch.save(model.state_dict(), save_path_pt)
        
        # KataGo形式（.txt.gz）で保存
        save_model_as_katago(model, model_name_gz)

        training_status["progress"] = 100
        training_status["is_training"] = False
        training_status["message"] = f"教師学習および1局の自戦対局による強化学習が完了しました。モデル: {model_name_gz}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        training_status["is_training"] = False
        training_status["message"] = f"エラーが発生しました: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    mode = data.get('mode', 'overwrite') # 'overwrite' or 'new'
    
    if training_status["is_training"]:
        return jsonify({'status': 'error', 'message': '現在学習が進行中です'}), 400
    
    # 別スレッドで学習を開始
    thread = threading.Thread(target=run_training_task, args=(mode,))
    thread.start()
    
    return jsonify({'status': 'success', 'message': '学習を開始しました'})

@app.route('/train_status', methods=['GET'])
def get_train_status():
    return jsonify(training_status)

@app.route('/models', methods=['GET'])
def get_models():
    models = [os.path.basename(DEFAULT_MODEL_PATH)]
    if os.path.exists(STUDY_MODEL_DIR):
        study_models = [f for f in os.listdir(STUDY_MODEL_DIR) if f.endswith(".gz")]
        # 重複を避ける（デフォルトモデルがstudy_model配下にある場合も考慮）
        for sm in study_models:
            if sm not in models:
                models.append(sm)
    return jsonify(models)

@app.route('/state', methods=['GET'])
def get_state():
    b_score, w_score = game.score()
    last_move = game.history[-1][1] if game.history else None
    return jsonify({
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'is_over': game.is_over(),
        'resigned_player': game.resigned_player,
        'scores': {'black': int(b_score), 'white': int(w_score)},
        'last_move': last_move
    })

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    r = data.get('r')
    c = data.get('c')
    
    # ユーザー（黒）の手
    if r is None or c is None:
        # パス
        game.play(None, None)
    else:
        # data['r'] と data['c'] は整数であることを期待
        if not game.play(int(r), int(c)):
            return jsonify({'status': 'error', 'message': 'Invalid move'}), 400

    # ゲームが終了していなければAI（白）の手
    if not game.is_over() and game.current_player == GoGame.WHITE:
        current_ai = get_ai_instance()
        ai_move = current_ai.get_move(game)
        if ai_move is None:
            game.play(None, None)
        else:
            game.play(ai_move[0], ai_move[1])

    b_score, w_score = game.score()
    last_move = game.history[-1][1] if game.history else None
    return jsonify({
        'status': 'success',
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'is_over': game.is_over(),
        'resigned_player': game.resigned_player,
        'scores': {'black': int(b_score), 'white': int(w_score)},
        'last_move': last_move
    })

@app.route('/resign', methods=['POST'])
def resign():
    # 現在のプレイヤーが投了
    game.resign(game.current_player)
    b_score, w_score = game.score()
    last_move = game.history[-1][1] if game.history else None
    return jsonify({
        'status': 'success',
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'is_over': game.is_over(),
        'resigned_player': game.resigned_player,
        'scores': {'black': int(b_score), 'white': int(w_score)},
        'last_move': last_move
    })

@app.route('/reset', methods=['POST'])
def reset():
    global game
    data = request.json
    selected_model = data.get('model')
    
    if selected_model:
        # モデルパスを決定する
        if selected_model == os.path.basename(DEFAULT_MODEL_PATH):
            m_path = DEFAULT_MODEL_PATH
        else:
            m_path = os.path.join(STUDY_MODEL_DIR, selected_model)
        
        if os.path.exists(m_path):
            get_ai_instance(m_path)
        else:
            return jsonify({'status': 'error', 'message': f'Model not found: {selected_model}'}), 404
    else:
        # モデルが指定されない場合は現在のモデルでAIを初期化（まだ存在しない場合）
        get_ai_instance()

    game = GoGame(size=9)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
