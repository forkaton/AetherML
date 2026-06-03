// =============================================
// MiniChart — Lightweight chart renderer for loss curves, elbow curves, etc.
// Renders into a <canvas> element.
// =============================================

class MiniChart {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');

        // Resize to CSS size
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.w = rect.width;
        this.h = rect.height;

        this.padding = options.padding || { top: 25, right: 15, bottom: 30, left: 50 };
        this.lineColor = options.lineColor || '#06b6d4';
        this.fillColor = options.fillColor || 'rgba(6, 182, 212, 0.1)';
        this.gridColor = options.gridColor || 'rgba(255,255,255,0.06)';
        this.textColor = options.textColor || '#94a3b8';
        this.title = options.title || '';
        this.xLabel = options.xLabel || '';
        this.yLabel = options.yLabel || '';
        this.data = [];
    }

    get plotW() { return this.w - this.padding.left - this.padding.right; }
    get plotH() { return this.h - this.padding.top - this.padding.bottom; }

    setData(arr) {
        this.data = arr;
        this.render();
    }

    addPoint(val) {
        this.data.push(val);
        // Keep last 200 points for performance
        if (this.data.length > 200) this.data.shift();
        this.render();
    }

    clear() {
        this.data = [];
        this.render();
    }

    render() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.w, this.h);

        const { top, right, bottom, left } = this.padding;
        const pw = this.plotW;
        const ph = this.plotH;

        if (this.data.length < 2) {
            ctx.fillStyle = this.textColor;
            ctx.font = '12px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data...', this.w / 2, this.h / 2);
            return;
        }

        const minVal = Math.min(...this.data);
        const maxVal = Math.max(...this.data);
        const range = maxVal - minVal || 1;

        // Title
        if (this.title) {
            ctx.fillStyle = '#f8fafc';
            ctx.font = '11px Outfit, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(this.title, left, 15);
        }

        // Grid lines
        ctx.strokeStyle = this.gridColor;
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = top + (ph / 4) * i;
            ctx.beginPath();
            ctx.moveTo(left, y);
            ctx.lineTo(left + pw, y);
            ctx.stroke();

            // Y axis labels
            const val = maxVal - (range / 4) * i;
            ctx.fillStyle = this.textColor;
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.textAlign = 'right';
            ctx.fillText(val.toFixed(3), left - 5, y + 3);
        }

        // Data line
        ctx.beginPath();
        ctx.strokeStyle = this.lineColor;
        ctx.lineWidth = 2;
        const step = pw / (this.data.length - 1);

        for (let i = 0; i < this.data.length; i++) {
            const x = left + i * step;
            const y = top + ph - ((this.data[i] - minVal) / range) * ph;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Fill under curve
        const lastX = left + (this.data.length - 1) * step;
        const lastY = top + ph - ((this.data[this.data.length - 1] - minVal) / range) * ph;
        ctx.lineTo(lastX, top + ph);
        ctx.lineTo(left, top + ph);
        ctx.closePath();
        ctx.fillStyle = this.fillColor;
        ctx.fill();

        // Current value indicator (last point)
        ctx.beginPath();
        ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
        ctx.fillStyle = this.lineColor;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();

        // X-axis label
        if (this.xLabel) {
            ctx.fillStyle = this.textColor;
            ctx.font = '10px Outfit, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(this.xLabel, left + pw / 2, this.h - 3);
        }
    }
}
