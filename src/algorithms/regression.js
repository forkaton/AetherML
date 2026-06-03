// =============================================
// Linear Regression Visualizer (Refactored)
// =============================================

document.addEventListener('DOMContentLoaded', () => {
    const canvasEl = document.getElementById('reg-canvas');
    if (!canvasEl) return;

    const canvas = new CanvasHelper('reg-canvas');
    let chart = null;
    const chartEl = document.getElementById('reg-chart');
    if (chartEl) {
        chart = new MiniChart('reg-chart', {
            title: 'Loss (MSE) over Epochs',
            lineColor: '#06b6d4',
            fillColor: 'rgba(6, 182, 212, 0.1)',
            xLabel: 'Epoch'
        });
    }

    let points = [];
    let m = 0;
    let c = 0.5;
    let epoch = 0;
    let isRunning = false;
    let animationId = null;
    let lossHistory = [];

    // UI Elements
    const lrInput = document.getElementById('reg-lr');
    const lrVal = document.getElementById('reg-lr-val');
    const modeSelect = document.getElementById('reg-mode');
    const epochDisplay = document.getElementById('reg-epoch');

    const btnStep = document.getElementById('reg-step');
    const btnRun = document.getElementById('reg-run');
    const btnReset = document.getElementById('reg-reset');
    const btnClear = document.getElementById('reg-clear');
    const btnPresetLin = document.getElementById('reg-preset-linear');
    const btnPresetNoise = document.getElementById('reg-preset-noise');

    // Math Panel Elements
    const valM = document.getElementById('val-m');
    const valC = document.getElementById('val-c');
    const valLoss = document.getElementById('val-loss');
    const valGradM = document.getElementById('val-grad-m');
    const valGradC = document.getElementById('val-grad-c');

    function updateMathPanel(mse, dm, dc) {
        if (valM) valM.textContent = formatNum(m);
        if (valC) valC.textContent = formatNum(c);
        if (valLoss) valLoss.textContent = formatNum(mse);
        if (valGradM) valGradM.textContent = formatNum(dm);
        if (valGradC) valGradC.textContent = formatNum(dc);
        if (epochDisplay) epochDisplay.textContent = epoch;
    }

    function render() {
        canvas.clear();
        if (points.length > 0) canvas.drawResiduals(points, m, c);
        points.forEach(p => canvas.drawPoint(p.x, p.y, '#f59e0b', 5, true));
        canvas.drawLine(m, c, '#06b6d4', 3);

        if (points.length > 0) {
            const mse = calculateMSE(points, m, c);
            const grads = calculateGradients(points, m, c);
            updateMathPanel(mse, grads.dm, grads.dc);
        } else {
            updateMathPanel(0, 0, 0);
        }
    }

    function stepGD() {
        if (points.length === 0) return;
        const lr = parseFloat(lrInput.value);
        const mode = modeSelect.value;
        let grads = { dm: 0, dc: 0 };

        if (mode === 'batch') {
            grads = calculateGradients(points, m, c);
        } else if (mode === 'sgd') {
            const p = points[Math.floor(Math.random() * points.length)];
            const err = p.y - (m * p.x + c);
            grads.dm = -2 * p.x * err;
            grads.dc = -2 * err;
        } else if (mode === 'minibatch') {
            const batchSize = Math.max(2, Math.floor(points.length * 0.2));
            const batch = shuffleArray(points).slice(0, batchSize);
            grads = calculateGradients(batch, m, c);
        }

        m -= lr * grads.dm;
        c -= lr * grads.dc;
        epoch++;

        const mse = calculateMSE(points, m, c);
        lossHistory.push(mse);
        if (chart) chart.setData(lossHistory);

        render();
    }

    function loop() {
        if (!isRunning) return;
        stepGD();
        animationId = requestAnimationFrame(loop);
    }

    // Interactions
    canvas.canvas.addEventListener('click', (e) => {
        const rect = canvas.canvas.getBoundingClientRect();
        const p = canvas.toDataCoord(e.clientX - rect.left, e.clientY - rect.top);
        points.push(p);
        render();
    });

    lrInput.addEventListener('input', (e) => { lrVal.textContent = e.target.value; });

    btnStep.addEventListener('click', () => stepGD());

    btnRun.addEventListener('click', () => {
        isRunning = !isRunning;
        if (isRunning) {
            btnRun.textContent = '⏸ Pause';
            btnRun.classList.add('running');
            loop();
        } else {
            btnRun.textContent = '▶ Run';
            btnRun.classList.remove('running');
            cancelAnimationFrame(animationId);
        }
    });

    btnReset.addEventListener('click', () => {
        m = 0; c = 0.5; epoch = 0; lossHistory = [];
        if (chart) chart.clear();
        render();
    });

    btnClear.addEventListener('click', () => {
        points = []; m = 0; c = 0.5; epoch = 0; lossHistory = [];
        if (isRunning) btnRun.click();
        if (chart) chart.clear();
        render();
    });

    btnPresetLin.addEventListener('click', () => {
        points = [];
        for (let i = 0.05; i <= 0.95; i += 0.05) {
            points.push({ x: i, y: i * 0.8 + 0.1 + (Math.random() * 0.08 - 0.04) });
        }
        m = 0; c = 0.5; epoch = 0; lossHistory = [];
        if (chart) chart.clear();
        render();
    });

    btnPresetNoise.addEventListener('click', () => {
        points = [];
        for (let i = 0.05; i <= 0.95; i += 0.04) {
            points.push({ x: i, y: 0.3 + i * 0.4 + (Math.random() * 0.3 - 0.15) });
        }
        m = 0; c = 0.5; epoch = 0; lossHistory = [];
        if (chart) chart.clear();
        render();
    });

    render();
});
