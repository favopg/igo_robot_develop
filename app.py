from flask import Flask, render_template, jsonify, request
from go_game import GoGame
import random

app = Flask(__name__)

# ゲームのグローバルインスタンス（シンプルにするため）
game = GoGame(size=9)

class RandomAI:
    def __init__(self, color):
        self.color = color

    def get_move(self, game):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)

ai = RandomAI(GoGame.WHITE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state', methods=['GET'])
def get_state():
    b_score, w_score = game.score()
    return jsonify({
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'is_over': game.is_over(),
        'scores': {'black': int(b_score), 'white': int(w_score)}
    })

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    r = data.get('r')
    c = data.get('c')
    
    # ユーザー（黒）の手
    if r is None and c is None:
        # パス
        game.play(None, None)
    else:
        if not game.play(r, c):
            return jsonify({'status': 'error', 'message': 'Invalid move'}), 400

    # ゲームが終了していなければAI（白）の手
    if not game.is_over() and game.current_player == GoGame.WHITE:
        ai_move = ai.get_move(game)
        if ai_move is None:
            game.play(None, None)
        else:
            game.play(ai_move[0], ai_move[1])

    b_score, w_score = game.score()
    return jsonify({
        'status': 'success',
        'board': game.board.tolist(),
        'current_player': game.current_player,
        'is_over': game.is_over(),
        'scores': {'black': int(b_score), 'white': int(w_score)}
    })

@app.route('/reset', methods=['POST'])
def reset():
    global game
    game = GoGame(size=9)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
