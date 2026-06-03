// =============================================
// Dataset Generators
// =============================================

const DatasetGenerator = {
    // Gaussian blobs for clustering/classification
    blobs(n = 100, centers = 3, spread = 0.06) {
        const points = [];
        const cx = [], cy = [];
        for (let i = 0; i < centers; i++) {
            cx.push(0.2 + Math.random() * 0.6);
            cy.push(0.2 + Math.random() * 0.6);
        }
        for (let i = 0; i < n; i++) {
            const cls = i % centers;
            points.push({
                x: clamp(cx[cls] + (Math.random() - 0.5) * spread * 2, 0.02, 0.98),
                y: clamp(cy[cls] + (Math.random() - 0.5) * spread * 2, 0.02, 0.98),
                cls: cls
            });
        }
        return points;
    },

    // Two interleaving half-moons
    moons(n = 150, noise = 0.04) {
        const points = [];
        const half = Math.floor(n / 2);
        for (let i = 0; i < half; i++) {
            const angle = Math.PI * i / half;
            points.push({
                x: clamp(0.3 + 0.2 * Math.cos(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                y: clamp(0.55 + 0.2 * Math.sin(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                cls: 0
            });
        }
        for (let i = 0; i < half; i++) {
            const angle = Math.PI * i / half;
            points.push({
                x: clamp(0.5 + 0.2 * Math.cos(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                y: clamp(0.45 - 0.2 * Math.sin(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                cls: 1
            });
        }
        return points;
    },

    // Concentric circles
    circles(n = 150, noise = 0.03) {
        const points = [];
        const half = Math.floor(n / 2);
        for (let i = 0; i < half; i++) {
            const angle = 2 * Math.PI * i / half;
            points.push({
                x: clamp(0.5 + 0.12 * Math.cos(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                y: clamp(0.5 + 0.12 * Math.sin(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                cls: 0
            });
        }
        for (let i = 0; i < half; i++) {
            const angle = 2 * Math.PI * i / half;
            points.push({
                x: clamp(0.5 + 0.28 * Math.cos(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                y: clamp(0.5 + 0.28 * Math.sin(angle) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                cls: 1
            });
        }
        return points;
    },

    // Two spirals (harder classification)
    spirals(n = 200, noise = 0.02) {
        const points = [];
        const half = Math.floor(n / 2);
        for (let i = 0; i < half; i++) {
            const t = 1.5 * Math.PI * (1 + 2 * i / half);
            const r = 0.04 + 0.22 * i / half;
            points.push({
                x: clamp(0.5 + r * Math.cos(t) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                y: clamp(0.5 + r * Math.sin(t) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                cls: 0
            });
        }
        for (let i = 0; i < half; i++) {
            const t = 1.5 * Math.PI * (1 + 2 * i / half);
            const r = 0.04 + 0.22 * i / half;
            points.push({
                x: clamp(0.5 - r * Math.cos(t) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                y: clamp(0.5 - r * Math.sin(t) + (Math.random() - 0.5) * noise, 0.02, 0.98),
                cls: 1
            });
        }
        return points;
    },

    // XOR-like pattern (for neural networks)
    xor(n = 120, noise = 0.06) {
        const points = [];
        const quadrants = [
            { cx: 0.3, cy: 0.7, cls: 0 },
            { cx: 0.7, cy: 0.7, cls: 1 },
            { cx: 0.3, cy: 0.3, cls: 1 },
            { cx: 0.7, cy: 0.3, cls: 0 }
        ];
        const perQ = Math.floor(n / 4);
        for (const q of quadrants) {
            for (let i = 0; i < perQ; i++) {
                points.push({
                    x: clamp(q.cx + (Math.random() - 0.5) * noise * 2, 0.02, 0.98),
                    y: clamp(q.cy + (Math.random() - 0.5) * noise * 2, 0.02, 0.98),
                    cls: q.cls
                });
            }
        }
        return points;
    },

    // Simple linear separable
    linear(n = 80, noise = 0.05) {
        const points = [];
        for (let i = 0; i < n; i++) {
            const x = Math.random() * 0.8 + 0.1;
            const y = Math.random() * 0.8 + 0.1;
            const cls = (y > x + (Math.random() - 0.5) * noise) ? 0 : 1;
            points.push({ x, y, cls });
        }
        return points;
    }
};
