// =============================================
// Neural Network — Backpropagation Debugger
// =============================================

window.AetherML = window.AetherML || {};
window.AetherML.nn = {
    initialized: false,
    canvas: null,       // decision boundary canvas
    netCanvas: null,    // network diagram canvas
    chart: null,        // loss chart

    points: [],
    layers: [2, 4, 4, 1],  // architecture: input(2) → hidden(4) → hidden(4) → output(1)
    activation: 'sigmoid',
    lr: 0.5,
    weights: [],
    biases: [],
    epoch: 0,
    lossHistory: [],
    isRunning: false,
    animId: null,

    // Cache for visualization
    lastForward: null,

    init() {
        if (this.initialized) return;
        this.canvas = new CanvasHelper('nn-data-canvas');
        this.netCanvas = new CanvasHelper('nn-net-canvas');
        if (!this.canvas.canvas) return;

        const chartEl = document.getElementById('nn-chart');
        if (chartEl) {
            this.chart = new MiniChart('nn-chart', {
                title: 'Loss (MSE)',
                lineColor: '#8b5cf6',
                fillColor: 'rgba(139,92,246,0.1)',
                xLabel: 'Epoch'
            });
        }

        this.sliderLR = document.getElementById('nn-lr');
        this.valLR = document.getElementById('nn-lr-val');
        this.selectAct = document.getElementById('nn-activation');
        this.btnStep = document.getElementById('nn-step');
        this.btnRun = document.getElementById('nn-run');
        this.btnReset = document.getElementById('nn-reset');
        this.btnClear = document.getElementById('nn-clear');
        this.btnPresetXor = document.getElementById('nn-preset-xor');
        this.btnPresetCircles = document.getElementById('nn-preset-circles');
        this.btnPresetSpirals = document.getElementById('nn-preset-spirals');
        this.selectArch = document.getElementById('nn-arch');

        this.sliderLR.addEventListener('input', (e) => {
            this.lr = parseFloat(e.target.value);
            this.valLR.textContent = this.lr.toFixed(2);
        });
        this.selectAct.addEventListener('change', (e) => {
            this.activation = e.target.value;
            this.initWeights();
            this.render();
        });
        this.selectArch.addEventListener('change', (e) => {
            const archs = {
                '2-4-1': [2, 4, 1],
                '2-4-4-1': [2, 4, 4, 1],
                '2-8-4-1': [2, 8, 4, 1],
                '2-6-6-1': [2, 6, 6, 1]
            };
            this.layers = archs[e.target.value] || [2, 4, 4, 1];
            this.initWeights();
            this.render();
        });

        this.canvas.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.canvas.getBoundingClientRect();
            const p = this.canvas.toDataCoord(e.clientX - rect.left, e.clientY - rect.top);
            // Determine class by which half of canvas
            p.cls = (p.x + p.y > 1) ? 1 : 0;
            this.points.push(p);
            this.render();
        });

        this.btnStep.addEventListener('click', () => this.trainEpoch());
        this.btnRun.addEventListener('click', () => this.toggleRun());
        this.btnReset.addEventListener('click', () => { this.initWeights(); this.epoch = 0; this.lossHistory = []; if (this.chart) this.chart.clear(); this.render(); });
        this.btnClear.addEventListener('click', () => { this.points = []; this.epoch = 0; this.lossHistory = []; if (this.chart) this.chart.clear(); this.initWeights(); this.render(); });

        this.btnPresetXor.addEventListener('click', () => { this.points = DatasetGenerator.xor(80); this.initWeights(); this.render(); });
        this.btnPresetCircles.addEventListener('click', () => { this.points = DatasetGenerator.circles(100); this.initWeights(); this.render(); });
        this.btnPresetSpirals.addEventListener('click', () => { this.points = DatasetGenerator.spirals(150); this.initWeights(); this.render(); });

        this.initWeights();
        this.initialized = true;
        this.render();
    },

    // Xavier initialization
    initWeights() {
        this.weights = [];
        this.biases = [];
        for (let l = 1; l < this.layers.length; l++) {
            const fanIn = this.layers[l - 1];
            const fanOut = this.layers[l];
            const scale = Math.sqrt(2 / (fanIn + fanOut));
            const w = [];
            for (let j = 0; j < fanOut; j++) {
                const row = [];
                for (let i = 0; i < fanIn; i++) {
                    row.push((Math.random() * 2 - 1) * scale);
                }
                w.push(row);
            }
            this.weights.push(w);
            this.biases.push(new Array(fanOut).fill(0).map(() => (Math.random() - 0.5) * 0.1));
        }
        this.epoch = 0;
    },

    // Forward pass (stores intermediate values for backprop)
    forward(input) {
        const act = getActivation(this.activation);
        const zs = [];  // pre-activation
        const as = [input]; // post-activation (first is input)

        let current = input;
        for (let l = 0; l < this.weights.length; l++) {
            const w = this.weights[l];
            const b = this.biases[l];
            const z = [];
            const a = [];
            const isOutput = (l === this.weights.length - 1);

            for (let j = 0; j < w.length; j++) {
                let sum = b[j];
                for (let i = 0; i < current.length; i++) {
                    sum += w[j][i] * current[i];
                }
                z.push(sum);
                // Output layer always uses sigmoid for binary classification
                a.push(isOutput ? sigmoid(sum) : act.fn(sum));
            }
            zs.push(z);
            as.push(a);
            current = a;
        }

        return { zs, as, output: current };
    },

    // Backward pass
    backward(forwardResult, target) {
        const { zs, as } = forwardResult;
        const act = getActivation(this.activation);
        const numLayers = this.weights.length;
        const dW = [];
        const dB = [];

        // Initialize gradients arrays
        for (let l = 0; l < numLayers; l++) {
            dW.push(this.weights[l].map(row => new Array(row.length).fill(0)));
            dB.push(new Array(this.biases[l].length).fill(0));
        }

        // Output layer error: dL/da * da/dz
        const outputA = as[numLayers];
        let delta = [outputA[0] - target]; // MSE derivative * sigmoid derivative baked in for output

        // Actually for sigmoid output with MSE:
        // dL/dz_output = (a - y) * sigmoidDeriv(z)
        delta = [(outputA[0] - target) * sigmoidDeriv(zs[numLayers - 1][0])];

        // Compute gradients for output layer
        for (let j = 0; j < this.weights[numLayers - 1].length; j++) {
            for (let i = 0; i < this.weights[numLayers - 1][j].length; i++) {
                dW[numLayers - 1][j][i] = delta[j] * as[numLayers - 1][i];
            }
            dB[numLayers - 1][j] = delta[j];
        }

        // Hidden layers (backpropagate)
        for (let l = numLayers - 2; l >= 0; l--) {
            const newDelta = [];
            for (let i = 0; i < this.weights[l].length; i++) {
                let err = 0;
                for (let j = 0; j < this.weights[l + 1].length; j++) {
                    err += this.weights[l + 1][j][i] * delta[j];
                }
                newDelta.push(err * act.deriv(zs[l][i]));
            }
            delta = newDelta;

            for (let j = 0; j < this.weights[l].length; j++) {
                for (let i = 0; i < this.weights[l][j].length; i++) {
                    dW[l][j][i] = delta[j] * as[l][i];
                }
                dB[l][j] = delta[j];
            }
        }

        return { dW, dB };
    },

    trainEpoch() {
        if (this.points.length === 0) return;
        let totalLoss = 0;

        // Shuffle points each epoch
        const shuffled = shuffleArray(this.points);

        for (const p of shuffled) {
            const input = [p.x, p.y];
            const target = p.cls;

            const fwd = this.forward(input);
            const { dW, dB } = this.backward(fwd, target);

            // Update weights
            for (let l = 0; l < this.weights.length; l++) {
                for (let j = 0; j < this.weights[l].length; j++) {
                    for (let i = 0; i < this.weights[l][j].length; i++) {
                        this.weights[l][j][i] -= this.lr * dW[l][j][i];
                    }
                    this.biases[l][j] -= this.lr * dB[l][j];
                }
            }

            totalLoss += (fwd.output[0] - target) ** 2;
        }

        totalLoss /= this.points.length;
        this.epoch++;
        this.lossHistory.push(totalLoss);
        if (this.chart) this.chart.setData(this.lossHistory);

        // Store last forward for viz (use first point)
        this.lastForward = this.forward([this.points[0].x, this.points[0].y]);

        this.render();
    },

    toggleRun() {
        this.isRunning = !this.isRunning;
        if (this.isRunning) {
            this.btnRun.textContent = 'Pause';
            this.btnRun.style.background = '#ef4444';
            this.runLoop();
        } else {
            this.btnRun.textContent = 'Run';
            this.btnRun.style.background = '#6366f1';
            cancelAnimationFrame(this.animId);
        }
    },

    runLoop() {
        if (!this.isRunning) return;
        // Train multiple epochs per frame for speed
        for (let i = 0; i < 5; i++) this.trainEpoch();
        this.animId = requestAnimationFrame(() => this.runLoop());
    },

    // Predict single point
    predict(x, y) {
        const fwd = this.forward([x, y]);
        return fwd.output[0];
    },

    // --- Rendering ---
    drawDecisionBoundary() {
        if (this.points.length === 0) return;
        const step = 6;
        const imgData = this.canvas.getImageData();
        const d = imgData.data;

        for (let py = 0; py < this.canvas.height; py += step) {
            for (let px = 0; px < this.canvas.width; px += step) {
                const pt = this.canvas.toDataCoord(px, py);
                const out = this.predict(pt.x, pt.y);
                // Blend between red and blue based on output
                const r = Math.floor(239 * (1 - out) + 59 * out);
                const g = Math.floor(68 * (1 - out) + 130 * out);
                const b = Math.floor(68 * (1 - out) + 246 * out);

                for (let dy = 0; dy < step && py + dy < this.canvas.height; dy++) {
                    for (let dx = 0; dx < step && px + dx < this.canvas.width; dx++) {
                        const idx = ((py + dy) * this.canvas.width + (px + dx)) * 4;
                        d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = 50;
                    }
                }
            }
        }
        this.canvas.putImageData(imgData);
    },

    drawNetworkDiagram() {
        if (!this.netCanvas) return;
        const ctx = this.netCanvas.ctx;
        const w = this.netCanvas.width;
        const h = this.netCanvas.height;

        ctx.clearRect(0, 0, w, h);

        const numLayers = this.layers.length;
        const layerSpacing = w / (numLayers + 1);

        // Positions for each neuron
        const positions = [];
        for (let l = 0; l < numLayers; l++) {
            const layerPos = [];
            const n = this.layers[l];
            const ySpacing = h / (n + 1);
            for (let i = 0; i < n; i++) {
                layerPos.push({
                    x: layerSpacing * (l + 1),
                    y: ySpacing * (i + 1)
                });
            }
            positions.push(layerPos);
        }

        // Draw connections (weights)
        for (let l = 0; l < this.weights.length; l++) {
            for (let j = 0; j < this.weights[l].length; j++) {
                for (let i = 0; i < this.weights[l][j].length; i++) {
                    const wt = this.weights[l][j][i];
                    const absW = Math.min(Math.abs(wt), 3);
                    const lineWidth = 0.5 + absW * 1.5;

                    // Color: green for positive, red for negative
                    const color = wt >= 0
                        ? `rgba(16, 185, 129, ${0.15 + absW * 0.2})`
                        : `rgba(239, 68, 68, ${0.15 + absW * 0.2})`;

                    ctx.beginPath();
                    ctx.moveTo(positions[l][i].x, positions[l][i].y);
                    ctx.lineTo(positions[l + 1][j].x, positions[l + 1][j].y);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = lineWidth;
                    ctx.stroke();
                }
            }
        }

        // Draw neurons
        for (let l = 0; l < numLayers; l++) {
            for (let i = 0; i < this.layers[l]; i++) {
                const pos = positions[l][i];
                const r = 12;

                // Neuron activation color
                let actVal = 0;
                if (this.lastForward && this.lastForward.as[l]) {
                    actVal = this.lastForward.as[l][i] || 0;
                }

                const brightness = Math.floor(40 + actVal * 180);
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
                ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${Math.floor(brightness * 1.3)})`;
                ctx.fill();
                ctx.strokeStyle = '#6366f1';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Label
                ctx.fillStyle = '#fff';
                ctx.font = '8px JetBrains Mono';
                ctx.textAlign = 'center';
                ctx.fillText(actVal.toFixed(1), pos.x, pos.y + 3);
            }
        }

        // Layer labels
        const labels = ['Input', ...this.layers.slice(1, -1).map((_, i) => `Hidden ${i + 1}`), 'Output'];
        for (let l = 0; l < numLayers; l++) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText(labels[l], layerSpacing * (l + 1), h - 8);
        }
    },

    render() {
        this.canvas.clear();
        this.drawDecisionBoundary();

        // Draw points
        const classColors = ['#ef4444', '#3b82f6'];
        this.points.forEach(p => {
            this.canvas.drawPoint(p.x, p.y, classColors[p.cls], 5, true);
        });

        this.drawNetworkDiagram();
        this.updateMathPanel();
    },

    updateMathPanel() {
        const content = document.getElementById('nn-math-content');
        if (!content) return;

        const loss = this.lossHistory.length > 0 ? this.lossHistory[this.lossHistory.length - 1] : 0;
        const archStr = this.layers.join(' → ');
        const totalParams = this.weights.reduce((sum, layerW, l) => {
            return sum + layerW.reduce((s, row) => s + row.length, 0) + this.biases[l].length;
        }, 0);

        content.innerHTML = `
            <div class="math-block"><h4>Forward Pass</h4>
            <code class="math-eq">z = Wx + b</code>
            <code class="math-eq">a = σ(z)</code>
            <div class="live-vars">
                <div class="var-row"><span>Architecture:</span><span class="val">${archStr}</span></div>
                <div class="var-row"><span>Parameters:</span><span class="val">${totalParams}</span></div>
                <div class="var-row"><span>Activation:</span><span class="val">${this.activation}</span></div>
            </div></div>
            <div class="math-block"><h4>Training</h4>
            <code class="math-eq">w ← w - α · ∂L/∂w</code>
            <div class="live-vars">
                <div class="var-row"><span>Epoch:</span><span class="val">${this.epoch}</span></div>
                <div class="var-row highlight"><span>Loss (MSE):</span><span class="val">${formatNum(loss)}</span></div>
                <div class="var-row"><span>Learning Rate:</span><span class="val">${this.lr.toFixed(2)}</span></div>
            </div></div>`;
    }
};
