document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('intersections');
    const hoshiContainer = document.getElementById('hoshi-container');
    const turnIndicator = document.getElementById('turn-indicator');
    const scoreIndicator = document.getElementById('score');
    const messageElement = document.getElementById('message');
    const passBtn = document.getElementById('pass-btn');
    const resignBtn = document.getElementById('resign-btn');
    const resetBtn = document.getElementById('reset-btn');
    const modelSelect = document.getElementById('model-select');
    const simSelect = document.getElementById('sim-select');
    const startGameBtn = document.getElementById('start-game-btn');

    let size = 9;
    let currentPlayer = 1; // 1: 黒, -1: 白

    let isWaiting = false;

    function createBoard() {
        boardElement.innerHTML = '';
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                const intersection = document.createElement('div');
                intersection.className = 'intersection';
                intersection.dataset.r = r;
                intersection.dataset.c = c;
                intersection.addEventListener('click', () => {
                    if (!isWaiting) {
                        handleMove(r, c);
                    }
                });
                
                // プレビュー機能の追加
                intersection.addEventListener('mouseenter', () => showPreview(intersection));
                intersection.addEventListener('mouseleave', () => removePreview(intersection));
                
                boardElement.appendChild(intersection);
            }
        }

        // 星（星打ち）の配置
        hoshiContainer.innerHTML = '';
        const hoshiPositions = [
            [2, 2], [2, 6], [6, 2], [6, 6], [4, 4]
        ];
        hoshiPositions.forEach(([r, c]) => {
            const hoshi = document.createElement('div');
            hoshi.className = 'hoshi';
            hoshi.style.top = `${r * 40}px`;
            hoshi.style.left = `${c * 40}px`;
            hoshiContainer.appendChild(hoshi);
        });
    }

    async function loadModels() {
        try {
            const response = await fetch('/models');
            const models = await response.json();
            modelSelect.innerHTML = '';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load models', error);
        }
    }

    async function updateState() {
        const response = await fetch('/state');
        const data = await response.json();
        renderBoard(data.board, data.last_move);
        updateStatus(data);
    }

    function renderBoard(board, lastMove = null) {
        const intersections = boardElement.querySelectorAll('.intersection');
        intersections.forEach(inter => {
            const r = parseInt(inter.dataset.r);
            const c = parseInt(inter.dataset.c);
            const value = board[r][c];
            
            // Remove existing stone and any highlights
            const existingStone = inter.querySelector('.stone');
            if (existingStone) existingStone.remove();

            if (value !== 0) {
                const stone = document.createElement('div');
                stone.className = `stone ${value === 1 ? 'black' : 'white'}`;
                
                // 最新の手を強調表示 (lastMoveが指定されている場合のみ)
                if (lastMove && lastMove[0] === r && lastMove[1] === c) {
                    stone.classList.add('last-move');
                }
                
                inter.appendChild(stone);
            }
        });
    }

    function updateStatus(data) {
        currentPlayer = data.current_player;
        turnIndicator.textContent = data.is_over ? '終局' : (data.current_player === 1 ? '黒の番です' : '白の番です');
        scoreIndicator.textContent = `黒: ${data.scores.black}, 白: ${data.scores.white}`;
        if (data.is_over) {
            let winner = '';
            if (data.resigned_player !== null) {
                winner = data.resigned_player === 1 ? '白の勝ち (黒の投了)' : '黒の勝ち (白の投了)';
            } else {
                winner = data.scores.black > data.scores.white ? '黒の勝ち' : (data.scores.white > data.scores.black ? '白の勝ち' : '引き分け');
            }
            messageElement.textContent = `ゲーム終了！ ${winner}`;
        } else {
            messageElement.textContent = '';
        }
    }

    function showPreview(intersection) {
        // すでに石がある場合や、自分の番でない（AIの思考中など）場合は表示しない
        if (intersection.querySelector('.stone') || currentPlayer !== 1) {
            return;
        }
        const stone = document.createElement('div');
        stone.className = `stone black preview`;
        intersection.appendChild(stone);
    }

    function removePreview(intersection) {
        const previewStone = intersection.querySelector('.stone.preview');
        if (previewStone) {
            previewStone.remove();
        }
    }

    async function handleMove(r, c) {
        if (isWaiting) return;

        // 着手時にプレビューを削除
        const intersections = boardElement.querySelectorAll('.intersection');
        intersections.forEach(inter => removePreview(inter));

        // 既存の赤いマーク(last-move)をすべて削除
        document.querySelectorAll('.last-move').forEach(el => el.classList.remove('last-move'));

        // 先行描画 (黒石を置く)
        if (r !== null && c !== null) {
            const targetInter = Array.from(intersections).find(inter => 
                parseInt(inter.dataset.r) === r && parseInt(inter.dataset.c) === c
            );
            if (targetInter && !targetInter.querySelector('.stone')) {
                const stone = document.createElement('div');
                stone.className = `stone black last-move`;
                targetInter.appendChild(stone);
                
                // AI思考中のメッセージを表示
                messageElement.textContent = 'AIが考えています...';
                messageElement.style.color = 'blue';
                isWaiting = true;
            }
        } else {
            // パスの場合も待ち状態にする
            messageElement.textContent = 'AIが考えています...';
            messageElement.style.color = 'blue';
            isWaiting = true;
        }

        try {
            const num_simulations = simSelect.value;
            const response = await fetch('/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ r, c, num_simulations })
            });
            const data = await response.json();
            if (data.status === 'success') {
                renderBoard(data.board, data.last_move);
                updateStatus(data);
            } else {
                messageElement.textContent = data.message;
                messageElement.style.color = 'red';
            }
        } catch (error) {
            console.error('Move failed', error);
            messageElement.textContent = '通信エラーが発生しました。';
            messageElement.style.color = 'red';
        } finally {
            isWaiting = false;
        }
    }

    passBtn.addEventListener('click', () => handleMove(null, null));

    resignBtn.addEventListener('click', async () => {
        if (confirm('投了しますか？')) {
            const response = await fetch('/resign', { method: 'POST' });
            const data = await response.json();
            renderBoard(data.board, data.last_move);
            updateStatus(data);
        }
    });

    resetBtn.addEventListener('click', async () => {
        if (confirm('現在の対局をリセットしますか？')) {
            const num_simulations = simSelect.value;
            await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_simulations })
            });
            updateState();
        }
    });

    startGameBtn.addEventListener('click', async () => {
        const model = modelSelect.value;
        const num_simulations = simSelect.value;
        if (confirm(`モデル「${model}」探索数「${num_simulations}」で新しい対局を開始しますか？`)) {
            const response = await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, num_simulations })
            });
            const data = await response.json();
            if (data.status === 'success') {
                updateState();
                messageElement.textContent = '新しい対局を開始しました。';
                messageElement.style.color = 'green';
                setTimeout(() => {
                    messageElement.style.color = '';
                }, 3000);
            } else {
                alert('エラー: ' + data.message);
            }
        }
    });

    // 機械学習機能の制御
    const trainStartBtn = document.getElementById('train-start-btn');
    const trainProgressContainer = document.getElementById('training-progress-container');
    const trainProgressBar = document.getElementById('training-progress-bar');
    const trainMessage = document.getElementById('training-message');

    trainStartBtn.addEventListener('click', async () => {
        const mode = document.querySelector('input[name="train-mode"]:checked').value;
        
        if (!confirm(`${mode === 'new' ? '新規モデル' : '既存モデル上書き'}で学習を開始しますか？`)) {
            return;
        }

        trainStartBtn.disabled = true;
        trainProgressContainer.style.display = 'block';
        
        try {
            const response = await fetch('/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
            const data = await response.json();
            
            if (data.status === 'success') {
                // 進捗ポーリング開始
                pollTrainingStatus();
            } else {
                alert(data.message);
                trainStartBtn.disabled = false;
            }
        } catch (error) {
            console.error('Training start failed', error);
            trainStartBtn.disabled = false;
        }
    });

    function pollTrainingStatus() {
        const interval = setInterval(async () => {
            try {
                const response = await fetch('/train_status');
                const data = await response.json();
                
                trainProgressBar.style.width = `${data.progress}%`;
                trainMessage.textContent = data.message;
                
                if (!data.is_training) {
                    clearInterval(interval);
                    trainStartBtn.disabled = false;
                    if (data.progress === 100) {
                        alert('学習が完了しました！');
                    }
                }
            } catch (error) {
                console.error('Polling failed', error);
                clearInterval(interval);
                trainStartBtn.disabled = false;
            }
        }, 1000);
    }

    // 自戦対局機能の制御
    const selfplayStartBtn = document.getElementById('selfplay-start-btn');
    const selfplayGamesSelect = document.getElementById('selfplay-games-select');
    const selfplayProgressContainer = document.getElementById('selfplay-progress-container');
    const selfplayProgressBar = document.getElementById('selfplay-progress-bar');
    const selfplayMessage = document.getElementById('selfplay-message');

    selfplayStartBtn.addEventListener('click', async () => {
        const model = modelSelect.value;
        const numGames = parseInt(selfplayGamesSelect.value);
        if (!confirm(`モデル「${model}」で自戦対局(${numGames}局)を開始しますか？`)) {
            return;
        }

        selfplayStartBtn.disabled = true;
        selfplayProgressContainer.style.display = 'block';

        try {
            const response = await fetch('/start_selfplay', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model, num_games: numGames })
            });
            const data = await response.json();

            if (data.status === 'success') {
                pollSelfplayStatus();
            } else {
                alert('エラー: ' + data.message);
                selfplayStartBtn.disabled = false;
            }
        } catch (error) {
            console.error('Selfplay execution failed', error);
            alert('通信エラーが発生しました。');
            selfplayStartBtn.disabled = false;
        }
    });

    function pollSelfplayStatus() {
        const interval = setInterval(async () => {
            try {
                const response = await fetch('/selfplay_status');
                const data = await response.json();
                
                selfplayProgressBar.style.width = `${data.progress}%`;
                selfplayMessage.textContent = data.message;
                
                if (!data.is_running) {
                    clearInterval(interval);
                    selfplayStartBtn.disabled = false;
                    if (data.progress === 100) {
                        alert('自戦対局と強化学習が完了しました！');
                    } else if (data.message.includes('エラー')) {
                        alert(data.message);
                    }
                }
            } catch (error) {
                console.error('Selfplay polling failed', error);
                clearInterval(interval);
                selfplayStartBtn.disabled = false;
            }
        }, 1000);
    }

    createBoard();
    loadModels();
    updateState();
});
