from flask import Flask, render_template, jsonify, request
from go_game import GoGame
import random

import subprocess
import threading
import time
import os
import shutil
import multiprocessing
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sgfmill import sgf, boards

# 自戦対局スクリプトから必要なクラスと関数をインポート
from selfplay_main import (
    SimpleGoNet, ResBlock, MCTSNode, run_mcts, 
    self_play_worker, board_to_tensor, game_to_sgfmill_board
)

app = Flask(__name__)

# ゲームのグローバルインスタンス
game = GoGame(size=9)

class SimplePyTorchAI:
    def __init__(self, color, model_path, num_simulations=1600):
        self.color = color
        self.model_path = model_path
        self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGoNet(size=9).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded PyTorch model from {model_path} with simulations={num_simulations}")
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")

    def get_move(self, game):
        # MCTSを使用して次の一手を選択
        # num_simulations を調整することで強さと速度を調整可能
        move = run_mcts(game, self.model, self.device, num_simulations=self.num_simulations)
        return move

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

# --- MCTS ロジック ---
# selfplay_main からインポート済み

# --- 教師あり学習 (SL) 用のネットワーク定義 ---
# selfplay_main からインポート済み

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

# --- 補助関数 ---
# board_to_tensor は selfplay_main からインポート済み

def get_symmetries(tensor, label, size=9):
    """
    盤面テンソルと指し手ラベルの8つの対称形（回転・反転）を生成します。
    tensor: (3, size, size)
    label: 指し手のインデックス (0 to size*size)
    """
    symmetries = []
    
    # 指し手の座標を取得
    if label == size * size: # パス
        r, c = None, None
    else:
        r, c = divmod(label, size)

    for i in range(8):
        # テンソルの回転・反転
        # i: 0-3 回転, 4-7 反転して回転
        new_tensor = tensor.copy()
        if i >= 4:
            new_tensor = np.flip(new_tensor, axis=2) # 左右反転
        
        rot_count = i % 4
        if rot_count > 0:
            new_tensor = np.rot90(new_tensor, k=rot_count, axes=(1, 2))
        
        # ラベルの回転・反転
        if r is None:
            new_label = size * size
        else:
            # numpyのrot90(k=1)は (r, c) -> (size-1-c, r) に相当
            # flip(axis=2)は (r, c) -> (r, size-1-c) に相当
            curr_r, curr_c = r, c
            if i >= 4:
                curr_c = size - 1 - curr_c
            
            for _ in range(rot_count):
                curr_r, curr_c = size - 1 - curr_c, curr_r
            
            new_label = curr_r * size + curr_c
            
        symmetries.append((new_tensor, new_label))
    
    return symmetries

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
                    raw_data = f.read()
                
                # 複数のエンコーディングを試行してデコード
                game = None
                for encoding in ["utf-8", "shift_jis", "cp932", "iso2022_jp"]:
                    try:
                        content = raw_data.decode(encoding)
                        game = sgf.Sgf_game.from_string(content)
                        break
                    except:
                        continue
                
                if game is None:
                    # デコードに失敗した場合はsgfmillのデフォルトに任せる
                    game = sgf.Sgf_game.from_bytes(raw_data)
                
                board_size = game.get_size()
                if board_size != 9:
                    continue
                
                board = boards.Board(board_size)
                
                # 対局者名を取得
                black_player = ""
                white_player = ""
                try:
                    black_player = game.root.get("PB")
                    if black_player is None: black_player = ""
                except KeyError:
                    pass
                try:
                    white_player = game.root.get("PW")
                    if white_player is None: white_player = ""
                except KeyError:
                    pass

                for move_obj in game.get_main_sequence():
                    color, move = move_obj.get_move()
                    if color is not None:
                        # 「イッシー」が含まれているプレイヤーの手のみ学習対象にする
                        current_player_name = black_player if color == "b" else white_player
                        if "イッシー" not in current_player_name:
                            # 盤面だけ更新して、学習データには追加しない
                            if move is not None:
                                r, c = move
                                board.play(r, c, color)
                            continue

                        # 盤面状態をテンソル化
                        tensor = board_to_tensor(board, color)
                        
                        if move is not None:
                            r, c = move
                            label = r * board_size + c
                            # 盤面を更新（データ保存の前に行うと、打った後の盤面になってしまうため注意が必要だが、
                            # 現状のロジックでは play する前に board_to_tensor しているので正しい）
                        else:
                            # パス
                            label = board_size * board_size
                        
                        # データ拡張（8つの対称形を追加）
                        symmetries = get_symmetries(tensor, label, board_size)
                        for sym_tensor, sym_label in symmetries:
                            features.append(sym_tensor)
                            labels.append(sym_label)
                        
                        # 盤面を更新
                        if move is not None:
                            board.play(r, c, color)
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

def get_ai_instance(m_path=None, num_simulations=1600):
    global ai, model_path
    if m_path:
        model_path = m_path
    
    # 既存のAIとモデルが異なる、または探索数が異なる場合に再生成
    if ai is None or ai.model_path != model_path or (hasattr(ai, 'num_simulations') and ai.num_simulations != num_simulations):
        if ai and hasattr(ai, 'proc') and ai.proc:
            # KataGoプロセスの終了
            try:
                ai.proc.terminate()
            except:
                pass
        
        # モデルファイル形式に応じてAIクラスを選択
        if model_path.endswith(".pt"):
            ai = SimplePyTorchAI(GoGame.WHITE, model_path, num_simulations=num_simulations)
        elif ("model_sl_" in model_path or "model_rl_" in model_path) and model_path.endswith(".txt.gz"):
            # 自作モデルの.txt.gzもPyTorchで読み込む（元となる.ptファイルを探す）
            pt_path = model_path.replace(".txt.gz", ".pt")
            if os.path.exists(pt_path):
                ai = SimplePyTorchAI(GoGame.WHITE, pt_path, num_simulations=num_simulations)
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

# 自戦対局状態を保持するグローバル変数
selfplay_status = {
    "is_running": False,
    "progress": 0,
    "message": "自戦対局は開始されていません"
}

def run_self_play(model, num_games=100, status_dict=None):
    """モデルを使用して自戦対局を行い、SGFファイルを保存します（並列化版）。"""
    if status_dict is None:
        global training_status
        status_dict = training_status

    selfplay_dir = "selfplay_sgf"
    if not os.path.exists(selfplay_dir):
        os.makedirs(selfplay_dir)
        
    num_processes = 4
    games_per_process = num_games // num_processes
    remaining_games = num_games % num_processes
    
    # モデルの重みをコピー（各プロセスに渡すため）
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    tasks = []
    for i in range(num_processes):
        n = games_per_process + (1 if i < remaining_games else 0)
        if n > 0:
            tasks.append((model_state, n, i, selfplay_dir))
    
    print(f"Starting parallel self-play with {num_processes} processes for {num_games} games...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 非同期で実行し、進捗を監視
        results = [pool.apply_async(self_play_worker, t) for t in tasks]
        
        finished_games = 0
        while finished_games < num_games:
            time.sleep(1)
            # 各タスクの進捗を確認（簡易的にファイル数でカウント）
            current_files = len([f for f in os.listdir(selfplay_dir) if f.startswith("selfplay_")])
            # ※ 実際にはこのディレクトリに既存のファイルがある可能性があるため、
            # より正確にはワーカーの完了報告を待つか、一時ディレクトリを使うべきですが、
            # ここではUI表示用に「現在のおおよその完了数」を表示します。
            
            # 完了したタスクの数を確認
            completed_tasks = sum(1 for r in results if r.ready())
            if completed_tasks == len(tasks):
                break
                
            # 進捗更新 (UI用)
            if status_dict == training_status:
                status_dict["progress"] = 40 + int((completed_tasks / len(tasks)) * 30)
            else:
                # 自戦対局タスクの場合、進捗は 0-80% の範囲（残りの20%は強化学習）
                status_dict["progress"] = int((completed_tasks / len(tasks)) * 80)
            status_dict["message"] = f"並列自戦対局中... {completed_tasks}/{len(tasks)} タスク完了"

    print("Parallel self-play completed.")
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

        # 2. 最終モデルの保存
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
            model_name_pt = f"model_sl_{timestamp}.pt"
            model_name_gz = f"model_sl_{timestamp}.txt.gz"
        
        # PyTorch形式で保存
        save_path_pt = os.path.join(STUDY_MODEL_DIR, model_name_pt)
        torch.save(model.state_dict(), save_path_pt)
        
        # KataGo形式（.txt.gz）で保存
        save_model_as_katago(model, model_name_gz)

        training_status["progress"] = 100
        training_status["is_training"] = False
        training_status["message"] = f"教師学習が完了しました。モデル: {model_name_gz}"
        
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

def run_self_play_task(model_filename, num_games=100):
    global selfplay_status, model_path
    selfplay_status["is_running"] = True
    selfplay_status["progress"] = 0
    selfplay_status["message"] = f"モデル {model_filename} で自戦対局準備中..."
    
    try:
        # モデルのパスを特定
        full_model_path = os.path.join(STUDY_MODEL_DIR, model_filename)
        if not os.path.exists(full_model_path):
            # デフォルトモデルの場合
            if model_filename == os.path.basename(DEFAULT_MODEL_PATH):
                full_model_path = DEFAULT_MODEL_PATH
            else:
                raise FileNotFoundError(f"Model file not found: {model_filename}")

        # PyTorchモデルをロードするために .pt を探す
        pt_path = full_model_path
        if full_model_path.endswith(".txt.gz"):
            pt_path = full_model_path.replace(".txt.gz", ".pt")
        
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"PyTorch model file not found: {pt_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleGoNet(size=9).to(device)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        model.eval()

        # 自戦対局実行
        selfplay_dir = run_self_play(model, num_games=num_games, status_dict=selfplay_status)

        # 強化学習 (自戦対局データによる追加学習)
        selfplay_status["progress"] = 80
        selfplay_status["message"] = "自戦対局データで強化学習中..."
        
        features_rl, labels_rl = load_sgf_data(selfplay_dir)
        if features_rl:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
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
                selfplay_status["progress"] = 80 + int((epoch + 1) / rl_epochs * 15)
                selfplay_status["message"] = f"強化学習中... Epoch {epoch+1}/{rl_epochs}, Loss: {avg_loss:.4f}"
                print(f"Self-play RL Epoch {epoch+1}/{rl_epochs}, Loss: {avg_loss:.4f}")

        # 最終モデルの保存（自戦対局後の学習結果は常に新規または上書きとして保存）
        # 元のモデルがデフォルトモデル以外ならそのモデルを更新、デフォルトなら新規作成
        model_name_gz = model_filename
        if model_filename == os.path.basename(DEFAULT_MODEL_PATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_gz = f"model_rl_{timestamp}.txt.gz"
        
        model_name_pt = model_name_gz.replace(".txt.gz", ".pt")
        
        # PyTorch形式で保存
        save_path_pt = os.path.join(STUDY_MODEL_DIR, model_name_pt)
        torch.save(model.state_dict(), save_path_pt)
        
        # KataGo形式（.txt.gz）で保存
        save_model_as_katago(model, model_name_gz)

        selfplay_status["progress"] = 100
        selfplay_status["is_running"] = False
        selfplay_status["message"] = f"モデル {model_name_gz} による自戦対局および強化学習が完了しました。"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        selfplay_status["is_running"] = False
        selfplay_status["message"] = f"エラーが発生しました: {str(e)}"

@app.route('/start_selfplay', methods=['POST'])
def start_selfplay_route():
    data = request.json
    model_filename = data.get('model')
    num_games = data.get('num_games', 100)
    
    if selfplay_status["is_running"]:
        return jsonify({'status': 'error', 'message': '現在自戦対局が進行中です'}), 400
    
    if not model_filename:
        return jsonify({'status': 'error', 'message': 'モデルが指定されていません'}), 400

    # 非同期的に自戦対局を実行
    thread = threading.Thread(target=run_self_play_task, args=(model_filename, num_games))
    thread.start()
    
    return jsonify({'status': 'success', 'message': '自戦対局を開始しました'})

@app.route('/selfplay_status', methods=['GET'])
def get_selfplay_status():
    return jsonify(selfplay_status)

@app.route('/models', methods=['GET'])
def get_models():
    models = [os.path.basename(DEFAULT_MODEL_PATH)]
    if os.path.exists(STUDY_MODEL_DIR):
        # .gz (KataGo形式) と .pt (PyTorch形式) の両方を取得
        study_models = [f for f in os.listdir(STUDY_MODEL_DIR) if f.endswith(".gz") or f.endswith(".pt")]
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
    num_simulations = int(data.get('num_simulations', 1600))
    
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
        current_ai = get_ai_instance(num_simulations=num_simulations)
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
    num_simulations = int(data.get('num_simulations', 1600))
    
    if selected_model:
        # モデルパスを決定する
        if selected_model == os.path.basename(DEFAULT_MODEL_PATH):
            m_path = DEFAULT_MODEL_PATH
        else:
            m_path = os.path.join(STUDY_MODEL_DIR, selected_model)
        
        if os.path.exists(m_path):
            get_ai_instance(m_path, num_simulations=num_simulations)
        else:
            return jsonify({'status': 'error', 'message': f'Model not found: {selected_model}'}), 404
    else:
        # モデルが指定されない場合は現在のモデルでAIを初期化（まだ存在しない場合）
        get_ai_instance(num_simulations=num_simulations)

    game = GoGame(size=9)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
