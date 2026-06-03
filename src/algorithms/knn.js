// =============================================
// KNN — K-Nearest Neighbors Decision Boundary Lab
// =============================================

window.AetherML = window.AetherML || {};
window.AetherML.knn = {
    initialized: false,
    canvas: null,
    points: [],
    currentClass: 0,
    K: 3,
    metric: 'euclidean',
    showBoundary: true,
    boundaryRes: 4, // pixel step for boundary (lower = higher quality but slower)
    hoverPoint: null,

    classColors: ['#ef4444', '#3b82f6', '#10b981'],
    classBgColors: ['rgba(239,68,68,0.35)', 'rgba(59,130,246,0.35)', 'rgba(16,185,129,0.35)'],
    classNames: ['Red', 'Blue', 'Green'],

    init() {
        if (this.initialized) return;
        this.canvas = new CanvasHelper('knn-canvas');
        if (!this.canvas.canvas) return;

        this.btnClear = document.getElementById('knn-clear');
        this.btnBoundary = document.getElementById('knn-toggle-boundary');
        this.sliderK = document.getElementById('knn-k');
        this.valK = document.getElementById('knn-k-val');
        this.selectMetric = document.getElementById('knn-metric');
        this.selectClass = document.getElementById('knn-class');
        this.btnPresetLinear = document.getElementById('knn-preset-linear');
        this.btnPresetMoons = document.getElementById('knn-preset-moons');
        this.btnPresetCircles = document.getElementById('knn-preset-circles');

        // Event listeners
        this.canvas.canvas.addEventListener('click', (e) => this.handleClick(e));
        this.canvas.canvas.addEventListener('mousemove', (e) => this.handleHover(e));
        this.btnClear.addEventListener('click', () => { this.points = []; this.render(); });
        this.btnBoundary.addEventListener('click', () => { this.showBoundary = !this.showBoundary; this.render(); });
        this.sliderK.addEventListener('input', (e) => {
            this.K = parseInt(e.target.value);
            this.valK.textContent = this.K;
            this.render();
        });
        this.selectMetric.addEventListener('change', (e) => { this.metric = e.target.value; this.render(); });
        this.selectClass.addEventListener('change', (e) => { this.currentClass = parseInt(e.target.value); });

        this.btnPresetLinear.addEventListener('click', () => { this.points = DatasetGenerator.linear(80); this.render(); });
        this.btnPresetMoons.addEventListener('click', () => { this.points = DatasetGenerator.moons(120); this.render(); });
        this.btnPresetCircles.addEventListener('click', () => { this.points = DatasetGenerator.circles(120); this.render(); });

        this.initialized = true;
        this.render();
    },

    handleClick(e) {
        const rect = this.canvas.canvas.getBoundingClientRect();
        const p = this.canvas.toDataCoord(e.clientX - rect.left, e.clientY - rect.top);
        p.cls = this.currentClass;
        this.points.push(p);
        this.render();
    },

    handleHover(e) {
        if (this.points.length === 0) return;
        const rect = this.canvas.canvas.getBoundingClientRect();
        this.hoverPoint = this.canvas.toDataCoord(e.clientX - rect.left, e.clientY - rect.top);
        this.updateMathPanel();
    },

    classify(queryPt) {
        const distFn = getDistanceFn(this.metric);
        const dists = this.points.map((p, i) => ({ idx: i, dist: distFn(queryPt, p), cls: p.cls }));
        dists.sort((a, b) => a.dist - b.dist);
        const neighbors = dists.slice(0, this.K);
        const votes = [0, 0, 0];
        neighbors.forEach(n => votes[n.cls]++);
        let maxVotes = 0, maxCls = 0;
        votes.forEach((v, i) => { if (v > maxVotes) { maxVotes = v; maxCls = i; } });
        return { cls: maxCls, neighbors, votes };
    },

    drawDecisionBoundary() {
        if (this.points.length < 2) return;
        const imgData = this.canvas.getImageData();
        const d = imgData.data;
        const step = this.boundaryRes;
        const colorMap = [
            [239, 68, 68],   // red
            [59, 130, 246],  // blue
            [16, 185, 129]   // green
        ];

        for (let py = 0; py < this.canvas.height; py += step) {
            for (let px = 0; px < this.canvas.width; px += step) {
                const pt = this.canvas.toDataCoord(px, py);
                const { cls } = this.classify(pt);
                const c = colorMap[cls];
                // Fill step x step block
                for (let dy = 0; dy < step && py + dy < this.canvas.height; dy++) {
                    for (let dx = 0; dx < step && px + dx < this.canvas.width; dx++) {
                        const idx = ((py + dy) * this.canvas.width + (px + dx)) * 4;
                        d[idx] = c[0]; d[idx + 1] = c[1]; d[idx + 2] = c[2]; d[idx + 3] = 40;
                    }
                }
            }
        }
        this.canvas.putImageData(imgData);
    },

    render() {
        this.canvas.clear();
        if (this.showBoundary && this.points.length >= 2) {
            this.drawDecisionBoundary();
        }
        // Draw points
        this.points.forEach(p => {
            this.canvas.drawPoint(p.x, p.y, this.classColors[p.cls], 6, true);
        });
        this.updateMathPanel();
    },

    updateMathPanel() {
        const content = document.getElementById('knn-math-content');
        if (!content) return;

        if (!this.hoverPoint || this.points.length === 0) {
            content.innerHTML = `
                <div class="math-block"><h4>KNN Info</h4>
                <code class="math-eq">Points: ${this.points.length} | K: ${this.K}</code>
                <p style="color:#94a3b8;font-size:0.85rem;">Hover over the canvas to see live KNN classification.</p></div>`;
            return;
        }

        const { cls, neighbors, votes } = this.classify(this.hoverPoint);
        let neighborsHTML = neighbors.map((n, i) =>
            `<div class="var-row"><span>#${i + 1} (Class ${this.classNames[n.cls]})</span><span class="val">${formatNum(n.dist)}</span></div>`
        ).join('');

        content.innerHTML = `
            <div class="math-block"><h4>Query Classification</h4>
            <code class="math-eq">Predicted: ${this.classNames[cls]}</code>
            <div class="live-vars">
                <div class="var-row"><span>Votes R/B/G:</span><span class="val">${votes.join(' / ')}</span></div>
            </div></div>
            <div class="math-block"><h4>K=${this.K} Nearest Neighbors</h4>
            <div class="live-vars">${neighborsHTML}</div></div>`;
    }
};
