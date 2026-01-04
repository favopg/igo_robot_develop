from flask import Flask, render_template, jsonify, request
from go_game import GoGame
import random

import subprocess
import threading
import time
import os

app = Flask(__name__)

# ゲームのグローバルインスタンス
game = GoGame(size=9)

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
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
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

katago_path = r"C:\katago\katago.exe"
model_path = r"C:\katago\kata1-b10c128-s1141046784-d204142634.txt.gz"
config_path = r"C:\katago\default_gtp.cfg"

ai = KataGoAI(GoGame.WHITE, katago_path, model_path, config_path)

@app.route('/')
def index():
    return render_template('index.html')

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
        ai_move = ai.get_move(game)
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
    game = GoGame(size=9)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
