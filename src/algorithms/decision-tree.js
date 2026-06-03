// =============================================
// Decision Tree — Split Visualizer
// =============================================

window.AetherML = window.AetherML || {};
window.AetherML.tree = {
    initialized: false,
    canvas: null,
    treeCanvas: null,
    points: [],
    tree: null,
    maxDepth: 4,
    impurityFn: 'gini', // 'gini' or 'entropy'
    classColors: ['#ef4444', '#3b82f6'],
    classNames: ['Red', 'Blue'],

    init() {
        if (this.initialized) return;
        this.canvas = new CanvasHelper('tree-data-canvas');
        this.treeCanvas = new CanvasHelper('tree-struct-canvas');
        if (!this.canvas.canvas) return;

        this.sliderDepth = document.getElementById('tree-depth');
        this.valDepth = document.getElementById('tree-depth-val');
        this.selectImpurity = document.getElementById('tree-impurity');
        this.selectClass = document.getElementById('tree-class');
        this.btnClear = document.getElementById('tree-clear');
        this.btnPresetLinear = document.getElementById('tree-preset-linear');
        this.btnPresetMoons = document.getElementById('tree-preset-moons');
        this.btnPresetXor = document.getElementById('tree-preset-xor');

        this.currentClass = 0;

        this.canvas.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.canvas.getBoundingClientRect();
            const p = this.canvas.toDataCoord(e.clientX - rect.left, e.clientY - rect.top);
            p.cls = this.currentClass;
            this.points.push(p);
            this.buildAndRender();
        });

        this.sliderDepth.addEventListener('input', (e) => {
            this.maxDepth = parseInt(e.target.value);
            this.valDepth.textContent = this.maxDepth;
            this.buildAndRender();
        });

        this.selectImpurity.addEventListener('change', (e) => {
            this.impurityFn = e.target.value;
            this.buildAndRender();
        });

        this.selectClass.addEventListener('change', (e) => { this.currentClass = parseInt(e.target.value); });

        this.btnClear.addEventListener('click', () => {
            this.points = []; this.tree = null; this.render();
        });
        this.btnPresetLinear.addEventListener('click', () => {
            this.points = DatasetGenerator.linear(80); this.buildAndRender();
        });
        this.btnPresetMoons.addEventListener('click', () => {
            this.points = DatasetGenerator.moons(100); this.buildAndRender();
        });
        this.btnPresetXor.addEventListener('click', () => {
            this.points = DatasetGenerator.xor(80); this.buildAndRender();
        });

        this.initialized = true;
        this.render();
    },

    // --- Tree building algorithm ---
    calcImpurity(points) {
        if (points.length === 0) return 0;
        const counts = [0, 0];
        points.forEach(p => counts[p.cls]++);
        return this.impurityFn === 'gini' ? giniImpurity(counts) : entropy(counts);
    },

    findBestSplit(points, depth) {
        if (points.length <= 1 || depth >= this.maxDepth) return null;

        const currentImpurity = this.calcImpurity(points);
        if (currentImpurity === 0) return null; // pure node

        let bestGain = 0, bestAxis = 0, bestThreshold = 0;

        for (let axis = 0; axis < 2; axis++) {
            const key = axis === 0 ? 'x' : 'y';
            const sorted = [...points].sort((a, b) => a[key] - b[key]);

            for (let i = 1; i < sorted.length; i++) {
                if (sorted[i][key] === sorted[i - 1][key]) continue;
                const threshold = (sorted[i][key] + sorted[i - 1][key]) / 2;
                const left = points.filter(p => p[key] <= threshold);
                const right = points.filter(p => p[key] > threshold);

                const wLeft = left.length / points.length;
                const wRight = right.length / points.length;
                const gain = currentImpurity - wLeft * this.calcImpurity(left) - wRight * this.calcImpurity(right);

                if (gain > bestGain) {
                    bestGain = gain;
                    bestAxis = axis;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestGain <= 0) return null;
        return { axis: bestAxis, threshold: bestThreshold, gain: bestGain };
    },

    buildTree(points, depth = 0, bounds = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 }) {
        const counts = [0, 0];
        points.forEach(p => counts[p.cls]++);
        const majority = counts[0] >= counts[1] ? 0 : 1;
        const impurity = this.calcImpurity(points);

        const node = { depth, points: points.length, counts, majority, impurity, bounds: { ...bounds } };

        const split = this.findBestSplit(points, depth);
        if (!split) {
            node.isLeaf = true;
            return node;
        }

        node.axis = split.axis;
        node.threshold = split.threshold;
        node.gain = split.gain;
        node.isLeaf = false;

        const key = split.axis === 0 ? 'x' : 'y';
        const leftPts = points.filter(p => p[key] <= split.threshold);
        const rightPts = points.filter(p => p[key] > split.threshold);

        const leftBounds = { ...bounds };
        const rightBounds = { ...bounds };
        if (split.axis === 0) {
            leftBounds.xMax = split.threshold;
            rightBounds.xMin = split.threshold;
        } else {
            leftBounds.yMax = split.threshold;
            rightBounds.yMin = split.threshold;
        }

        node.left = this.buildTree(leftPts, depth + 1, leftBounds);
        node.right = this.buildTree(rightPts, depth + 1, rightBounds);

        return node;
    },

    predictPoint(node, point) {
        if (node.isLeaf) return node.majority;
        const key = node.axis === 0 ? 'x' : 'y';
        if (point[key] <= node.threshold) return this.predictPoint(node.left, point);
        return this.predictPoint(node.right, point);
    },

    buildAndRender() {
        if (this.points.length >= 2) {
            this.tree = this.buildTree(this.points);
        } else {
            this.tree = null;
        }
        this.render();
    },

    // --- Rendering ---
    drawDecisionRegions() {
        if (!this.tree) return;
        const step = 4;
        const imgData = this.canvas.getImageData();
        const d = imgData.data;
        const colorMap = [[239, 68, 68], [59, 130, 246]];

        for (let py = 0; py < this.canvas.height; py += step) {
            for (let px = 0; px < this.canvas.width; px += step) {
                const pt = this.canvas.toDataCoord(px, py);
                const cls = this.predictPoint(this.tree, pt);
                const c = colorMap[cls];
                for (let dy = 0; dy < step && py + dy < this.canvas.height; dy++) {
                    for (let dx = 0; dx < step && px + dx < this.canvas.width; dx++) {
                        const idx = ((py + dy) * this.canvas.width + (px + dx)) * 4;
                        d[idx] = c[0]; d[idx + 1] = c[1]; d[idx + 2] = c[2]; d[idx + 3] = 30;
                    }
                }
            }
        }
        this.canvas.putImageData(imgData);
    },

    drawSplitLines(node) {
        if (!node || node.isLeaf) return;
        const color = `hsla(${200 + node.depth * 40}, 80%, 70%, ${0.6 - node.depth * 0.08})`;
        const b = node.bounds;

        if (node.axis === 0) {
            this.canvas.drawSplitLine(0, node.threshold, b.yMin, b.yMax, color, 2);
        } else {
            this.canvas.drawSplitLine(1, node.threshold, b.xMin, b.xMax, color, 2);
        }
        this.drawSplitLines(node.left);
        this.drawSplitLines(node.right);
    },

    drawTreeDiagram(node, x, y, spread, depth = 0) {
        if (!node) return;
        const ctx = this.treeCanvas.ctx;
        const r = 14;

        // Node color
        const color = node.isLeaf ? this.classColors[node.majority] : '#6366f1';
        const bgAlpha = node.isLeaf ? 0.9 : 0.5;

        // Draw children connections first
        if (!node.isLeaf) {
            const childY = y - 50;
            const leftX = x - spread;
            const rightX = x + spread;

            ctx.beginPath();
            ctx.moveTo(x * this.treeCanvas.width, (1 - y) * this.treeCanvas.height);
            ctx.lineTo(leftX * this.treeCanvas.width, (1 - childY) * this.treeCanvas.height);
            ctx.strokeStyle = 'rgba(99,102,241,0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(x * this.treeCanvas.width, (1 - y) * this.treeCanvas.height);
            ctx.lineTo(rightX * this.treeCanvas.width, (1 - childY) * this.treeCanvas.height);
            ctx.stroke();

            this.drawTreeDiagram(node.left, leftX, childY, spread * 0.5, depth + 1);
            this.drawTreeDiagram(node.right, rightX, childY, spread * 0.5, depth + 1);
        }

        // Draw node circle
        const cx = x * this.treeCanvas.width;
        const cy = (1 - y) * this.treeCanvas.height;

        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = bgAlpha;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#fff';
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'center';
        if (node.isLeaf) {
            ctx.fillText(`${this.classNames[node.majority][0]}`, cx, cy + 3);
        } else {
            const axisLabel = node.axis === 0 ? 'x' : 'y';
            ctx.fillText(`${axisLabel}`, cx, cy + 3);
        }
    },

    render() {
        this.canvas.clear();
        if (this.treeCanvas) this.treeCanvas.clear();

        if (this.tree) {
            this.drawDecisionRegions();
            this.drawSplitLines(this.tree);
        }

        // Points
        this.points.forEach(p => this.canvas.drawPoint(p.x, p.y, this.classColors[p.cls], 5, true));

        // Tree diagram
        if (this.tree && this.treeCanvas) {
            this.drawTreeDiagram(this.tree, 0.5, 0.9, 0.22);
        }

        this.updateMathPanel();
    },

    updateMathPanel() {
        const content = document.getElementById('tree-math-content');
        if (!content) return;

        const impLabel = this.impurityFn === 'gini' ? 'Gini Impurity' : 'Entropy';
        const rootImpurity = this.tree ? formatNum(this.tree.impurity) : '—';

        const countNodes = (n) => {
            if (!n) return { total: 0, leaves: 0 };
            if (n.isLeaf) return { total: 1, leaves: 1 };
            const l = countNodes(n.left);
            const r = countNodes(n.right);
            return { total: 1 + l.total + r.total, leaves: l.leaves + r.leaves };
        };
        const stats = countNodes(this.tree);

        content.innerHTML = `
            <div class="math-block"><h4>${impLabel}</h4>
            <code class="math-eq">${this.impurityFn === 'gini' ? 'G = 1 - Σ p²ᵢ' : 'H = -Σ pᵢ log₂(pᵢ)'}</code>
            <div class="live-vars">
                <div class="var-row"><span>Root impurity:</span><span class="val">${rootImpurity}</span></div>
                <div class="var-row"><span>Total nodes:</span><span class="val">${stats.total}</span></div>
                <div class="var-row"><span>Leaf nodes:</span><span class="val">${stats.leaves}</span></div>
                <div class="var-row"><span>Max depth:</span><span class="val">${this.maxDepth}</span></div>
                <div class="var-row"><span>Data points:</span><span class="val">${this.points.length}</span></div>
            </div></div>`;
    }
};
