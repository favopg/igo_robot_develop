document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('intersections');
    const hoshiContainer = document.getElementById('hoshi-container');
    const turnIndicator = document.getElementById('turn-indicator');
    const scoreIndicator = document.getElementById('score');
    const messageElement = document.getElementById('message');
    const passBtn = document.getElementById('pass-btn');
    const resignBtn = document.getElementById('resign-btn');
    const resetBtn = document.getElementById('reset-btn');

    let size = 9;

    function createBoard() {
        boardElement.innerHTML = '';
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                const intersection = document.createElement('div');
                intersection.className = 'intersection';
                intersection.dataset.r = r;
                intersection.dataset.c = c;
                intersection.addEventListener('click', () => handleMove(r, c));
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

    async function updateState() {
        const response = await fetch('/state');
        const data = await response.json();
        renderBoard(data.board);
        updateStatus(data);
    }

    function renderBoard(board) {
        const intersections = boardElement.querySelectorAll('.intersection');
        intersections.forEach(inter => {
            const r = inter.dataset.r;
            const c = inter.dataset.c;
            const value = board[r][c];
            
            // Remove existing stone
            const existingStone = inter.querySelector('.stone');
            if (existingStone) existingStone.remove();

            if (value !== 0) {
                const stone = document.createElement('div');
                stone.className = `stone ${value === 1 ? 'black' : 'white'}`;
                inter.appendChild(stone);
            }
        });
    }

    function updateStatus(data) {
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

    async function handleMove(r, c) {
        try {
            const response = await fetch('/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ r, c })
            });
            const data = await response.json();
            if (data.status === 'success') {
                renderBoard(data.board);
                updateStatus(data);
            } else {
                messageElement.textContent = data.message;
            }
        } catch (error) {
            console.error('Move failed', error);
        }
    }

    passBtn.addEventListener('click', () => handleMove(null, null));

    resignBtn.addEventListener('click', async () => {
        if (confirm('投了しますか？')) {
            const response = await fetch('/resign', { method: 'POST' });
            const data = await response.json();
            renderBoard(data.board);
            updateStatus(data);
        }
    });

    resetBtn.addEventListener('click', async () => {
        await fetch('/reset', { method: 'POST' });
        updateState();
    });

    createBoard();
    updateState();
});
