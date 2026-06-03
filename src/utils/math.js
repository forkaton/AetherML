// =============================================
// Math Helpers for all ML Modules
// =============================================

// --- Regression ---
function calculateMSE(points, m, c) {
    if (points.length === 0) return 0;
    let sum = 0;
    for (const p of points) {
        const pred = m * p.x + c;
        sum += (p.y - pred) ** 2;
    }
    return sum / points.length;
}

function calculateGradients(points, m, c) {
    let dm = 0, dc = 0;
    const n = points.length;
    for (const p of points) {
        const err = p.y - (m * p.x + c);
        dm += -2 * p.x * err;
        dc += -2 * err;
    }
    return { dm: dm / n, dc: dc / n };
}

// --- Distance Functions (KNN, K-Means) ---
function euclideanDist(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function manhattanDist(a, b) {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function minkowskiDist(a, b, p) {
    return Math.pow(Math.abs(a.x - b.x) ** p + Math.abs(a.y - b.y) ** p, 1 / p);
}

function getDistanceFn(metric) {
    if (metric === 'manhattan') return manhattanDist;
    if (metric === 'minkowski') return (a, b) => minkowskiDist(a, b, 3);
    return euclideanDist; // default
}

// --- Activation Functions (Neural Network) ---
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function sigmoidDeriv(x) { const s = sigmoid(x); return s * (1 - s); }

function relu(x) { return Math.max(0, x); }
function reluDeriv(x) { return x > 0 ? 1 : 0; }

function tanhAct(x) { return Math.tanh(x); }
function tanhDeriv(x) { return 1 - Math.tanh(x) ** 2; }

function getActivation(name) {
    const fns = {
        sigmoid: { fn: sigmoid, deriv: sigmoidDeriv },
        relu:    { fn: relu,    deriv: reluDeriv },
        tanh:    { fn: tanhAct, deriv: tanhDeriv }
    };
    return fns[name] || fns.sigmoid;
}

// --- Tree Impurity Measures ---
function giniImpurity(counts) {
    const total = counts.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    let sum = 0;
    for (const c of counts) {
        const p = c / total;
        sum += p * p;
    }
    return 1 - sum;
}

function entropy(counts) {
    const total = counts.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    let sum = 0;
    for (const c of counts) {
        if (c === 0) continue;
        const p = c / total;
        sum -= p * Math.log2(p);
    }
    return sum;
}

// --- Utility ---
function formatNum(num, digits = 4) {
    return Number(num).toFixed(digits);
}

function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
}

function shuffleArray(arr) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// Random number in range
function randRange(min, max) {
    return Math.random() * (max - min) + min;
}
