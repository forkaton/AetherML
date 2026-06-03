// =============================================
// K-Means Clustering — Convergence Analyzer
// =============================================

window.AetherML = window.AetherML || {};
window.AetherML.kmeans = {
    initialized: false,
    canvas: null,
    chart: null,
    points: [],
    centroids: [],
    assignments: [],
    K: 3,
    iteration: 0,
    converged: false,
    useKMeansPP: true,
    isRunning: false,
    animId: null,
    wcssHistory: [],

    clusterColors: ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899', '#f97316'],
    clusterBg: [
        'rgba(239,68,68,0.5)', 'rgba(59,130,246,0.5)', 'rgba(16,185,129,0.5)', 'rgba(245,158,11,0.5)',
        'rgba(139,92,246,0.5)', 'rgba(6,182,212,0.5)', 'rgba(236,72,153,0.5)', 'rgba(249,115,22,0.5)'
    ],

    init() {
        if (this.initialized) return;
        this.canvas = new CanvasHelper('kmeans-canvas');
        if (!this.canvas.canvas) return;

        const chartEl = document.getElementById('kmeans-chart');
        if (chartEl) {
            this.chart = new MiniChart('kmeans-chart', {
                title: 'WCSS (Inertia)',
                lineColor: '#f59e0b',
                fillColor: 'rgba(245,158,11,0.1)',
                xLabel: 'Iteration'
            });
        }

        // Controls
        this.sliderK = document.getElementById('kmeans-k');
        this.valK = document.getElementById('kmeans-k-val');
        this.btnStep = document.getElementById('kmeans-step');
        this.btnRun = document.getElementById('kmeans-run');
        this.btnReset = document.getElementById('kmeans-reset');
        this.btnClear = document.getElementById('kmeans-clear');
        this.toggleKPP = document.getElementById('kmeans-kpp');
        this.btnPresetBlobs = document.getElementById('kmeans-preset-blobs');
        this.btnPresetMoons = document.getElementById('kmeans-preset-moons');
        this.btnPresetCircles = document.getElementById('kmeans-preset-circles');

        // Events
        this.canvas.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.canvas.getBoundingClientRect();
            const p = this.canvas.toDataCoord(e.clientX - rect.left, e.clientY - rect.top);
            this.points.push(p);
            this.render();
        });

        this.sliderK.addEventListener('input', (e) => {
            this.K = parseInt(e.target.value);
            this.valK.textContent = this.K;
        });

        this.btnStep.addEventListener('click', () => this.step());
        this.btnRun.addEventListener('click', () => this.toggleRun());
        this.btnReset.addEventListener('click', () => this.resetCentroids());
        this.btnClear.addEventListener('click', () => this.clearAll());
        this.toggleKPP.addEventListener('change', (e) => { this.useKMeansPP = e.target.checked; });

        this.btnPresetBlobs.addEventListener('click', () => {
            this.points = DatasetGenerator.blobs(120, this.K); this.resetCentroids();
        });
        this.btnPresetMoons.addEventListener('click', () => {
            this.points = DatasetGenerator.moons(120); this.resetCentroids();
        });
        this.btnPresetCircles.addEventListener('click', () => {
            this.points = DatasetGenerator.circles(120); this.resetCentroids();
        });

        this.initialized = true;
        this.render();
    },

    initCentroidsRandom() {
        this.centroids = [];
        const shuffled = shuffleArray(this.points);
        for (let i = 0; i < this.K; i++) {
            if (shuffled[i]) {
                this.centroids.push({ x: shuffled[i].x, y: shuffled[i].y });
            } else {
                this.centroids.push({ x: Math.random(), y: Math.random() });
            }
        }
    },

    initCentroidsKPP() {
        this.centroids = [];
        // First centroid: random point
        const first = this.points[Math.floor(Math.random() * this.points.length)];
        this.centroids.push({ x: first.x, y: first.y });

        for (let c = 1; c < this.K; c++) {
            // For each point, compute min distance to existing centroids
            const dists = this.points.map(p => {
                let minD = Infinity;
                this.centroids.forEach(ct => { minD = Math.min(minD, euclideanDist(p, ct)); });
                return minD * minD; // squared distance as weight
            });
            const totalDist = dists.reduce((a, b) => a + b, 0);
            // Weighted random selection
            let r = Math.random() * totalDist;
            for (let i = 0; i < dists.length; i++) {
                r -= dists[i];
                if (r <= 0) {
                    this.centroids.push({ x: this.points[i].x, y: this.points[i].y });
                    break;
                }
            }
            if (this.centroids.length <= c) {
                // Fallback
                const p = this.points[Math.floor(Math.random() * this.points.length)];
                this.centroids.push({ x: p.x, y: p.y });
            }
        }
    },

    resetCentroids() {
        if (this.points.length < this.K) return;
        if (this.isRunning) this.toggleRun();
        this.iteration = 0;
        this.converged = false;
        this.wcssHistory = [];
        if (this.chart) this.chart.clear();
        if (this.useKMeansPP) {
            this.initCentroidsKPP();
        } else {
            this.initCentroidsRandom();
        }
        this.assignPoints();
        this.render();
    },

    clearAll() {
        if (this.isRunning) this.toggleRun();
        this.points = [];
        this.centroids = [];
        this.assignments = [];
        this.iteration = 0;
        this.converged = false;
        this.wcssHistory = [];
        if (this.chart) this.chart.clear();
        this.render();
    },

    assignPoints() {
        this.assignments = this.points.map(p => {
            let minD = Infinity, bestC = 0;
            this.centroids.forEach((c, i) => {
                const d = euclideanDist(p, c);
                if (d < minD) { minD = d; bestC = i; }
            });
            return bestC;
        });
    },

    moveCentroids() {
        const oldCentroids = this.centroids.map(c => ({ ...c }));
        for (let c = 0; c < this.K; c++) {
            let sx = 0, sy = 0, count = 0;
            this.points.forEach((p, i) => {
                if (this.assignments[i] === c) { sx += p.x; sy += p.y; count++; }
            });
            if (count > 0) {
                this.centroids[c] = { x: sx / count, y: sy / count };
            }
        }
        // Check convergence
        let maxDelta = 0;
        for (let c = 0; c < this.K; c++) {
            maxDelta = Math.max(maxDelta, euclideanDist(oldCentroids[c], this.centroids[c]));
        }
        if (maxDelta < 0.001) this.converged = true;
    },

    calculateWCSS() {
        let wcss = 0;
        this.points.forEach((p, i) => {
            const c = this.centroids[this.assignments[i]];
            if (c) wcss += (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
        });
        return wcss;
    },

    step() {
        if (this.points.length < this.K) return;
        if (this.centroids.length === 0) this.resetCentroids();
        if (this.converged) return;

        this.assignPoints();
        this.moveCentroids();
        this.iteration++;

        const wcss = this.calculateWCSS();
        this.wcssHistory.push(wcss);
        if (this.chart) this.chart.setData(this.wcssHistory);

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
        if (!this.isRunning || this.converged) {
            this.isRunning = false;
            this.btnRun.textContent = 'Run';
            this.btnRun.style.background = '#6366f1';
            return;
        }
        this.step();
        this.animId = setTimeout(() => this.runLoop(), 300); // slow enough to see
    },

    render() {
        this.canvas.clear();

        // Draw connections from points to centroids
        if (this.centroids.length > 0 && this.assignments.length > 0) {
            this.canvas.ctx.globalAlpha = 0.15;
            this.points.forEach((p, i) => {
                const c = this.centroids[this.assignments[i]];
                if (c) this.canvas.drawConnection(p.x, p.y, c.x, c.y, this.clusterColors[this.assignments[i]], 1);
            });
            this.canvas.ctx.globalAlpha = 1;
        }

        // Draw points
        this.points.forEach((p, i) => {
            const color = this.assignments.length > 0 ? this.clusterBg[this.assignments[i]] : '#94a3b8';
            this.canvas.drawPoint(p.x, p.y, color, 5);
        });

        // Draw centroids
        this.centroids.forEach((c, i) => {
            this.canvas.drawPoint(c.x, c.y, this.clusterColors[i], 10, true);
            // Cross marker
            const pos = this.canvas.toCanvasCoord(c.x, c.y);
            this.canvas.ctx.strokeStyle = '#fff';
            this.canvas.ctx.lineWidth = 2;
            this.canvas.ctx.beginPath();
            this.canvas.ctx.moveTo(pos.cx - 6, pos.cy); this.canvas.ctx.lineTo(pos.cx + 6, pos.cy);
            this.canvas.ctx.moveTo(pos.cx, pos.cy - 6); this.canvas.ctx.lineTo(pos.cx, pos.cy + 6);
            this.canvas.ctx.stroke();
        });

        this.updateMathPanel();
    },

    updateMathPanel() {
        const content = document.getElementById('kmeans-math-content');
        if (!content) return;

        const wcss = this.centroids.length > 0 ? this.calculateWCSS() : 0;
        let centroidRows = this.centroids.map((c, i) =>
            `<div class="var-row"><span>C${i + 1}:</span><span class="val">(${formatNum(c.x, 3)}, ${formatNum(c.y, 3)})</span></div>`
        ).join('');

        content.innerHTML = `
            <div class="math-block"><h4>Algorithm State</h4>
            <code class="math-eq">Assign → Move → Repeat</code>
            <div class="live-vars">
                <div class="var-row"><span>Iteration:</span><span class="val">${this.iteration}</span></div>
                <div class="var-row highlight"><span>WCSS:</span><span class="val">${formatNum(wcss)}</span></div>
                <div class="var-row"><span>Converged:</span><span class="val">${this.converged ? '✅ Yes' : '⏳ No'}</span></div>
                <div class="var-row"><span>Init:</span><span class="val">${this.useKMeansPP ? 'K-Means++' : 'Random'}</span></div>
            </div></div>
            <div class="math-block"><h4>Centroid Positions</h4>
            <div class="live-vars">${centroidRows || '<p style="color:#94a3b8;">None</p>'}</div></div>`;
    }
};
