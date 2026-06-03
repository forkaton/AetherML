// Canvas Helper Functions — Enhanced
class CanvasHelper {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.resizeToDisplay();
    }

    // Fix: match canvas buffer to CSS display size for correct click mapping
    resizeToDisplay() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.width = rect.width;
        this.height = rect.height;
    }

    clear() {
        this.ctx.clearRect(0, 0, this.width, this.height);
    }

    // Convert data coordinates (0-1) to canvas coordinates
    toCanvasCoord(x, y) {
        return {
            cx: x * this.width,
            cy: (1 - y) * this.height
        };
    }

    // Convert canvas coordinates to data coordinates (0-1)
    toDataCoord(cx, cy) {
        return {
            x: cx / this.width,
            y: 1 - (cy / this.height)
        };
    }

    drawPoint(x, y, color = '#f8fafc', radius = 5, glow = false) {
        const { cx, cy } = this.toCanvasCoord(x, y);
        if (glow) {
            this.ctx.save();
            this.ctx.shadowColor = color;
            this.ctx.shadowBlur = 12;
        }
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = 'rgba(0,0,0,0.5)';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        if (glow) this.ctx.restore();
    }

    drawLine(m, c, color = '#06b6d4', width = 2) {
        this.ctx.beginPath();
        const y0 = m * 0 + c;
        const y1 = m * 1 + c;
        const p0 = this.toCanvasCoord(0, y0);
        const p1 = this.toCanvasCoord(1, y1);
        this.ctx.moveTo(p0.cx, p0.cy);
        this.ctx.lineTo(p1.cx, p1.cy);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = width;
        this.ctx.stroke();
    }

    drawResiduals(points, m, c) {
        this.ctx.save();
        this.ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([4, 4]);
        points.forEach(p => {
            const predY = m * p.x + c;
            const pData = this.toCanvasCoord(p.x, p.y);
            const pLine = this.toCanvasCoord(p.x, predY);
            this.ctx.beginPath();
            this.ctx.moveTo(pData.cx, pData.cy);
            this.ctx.lineTo(pLine.cx, pLine.cy);
            this.ctx.stroke();
        });
        this.ctx.restore();
    }

    // Draw filled rectangle in data coords
    fillRect(x, y, w, h, color) {
        const p = this.toCanvasCoord(x, y + h);
        this.ctx.fillStyle = color;
        this.ctx.fillRect(p.cx, p.cy, w * this.width, h * this.height);
    }

    // Draw text at data coordinates
    drawText(text, x, y, color = '#f8fafc', size = 12, align = 'center') {
        const { cx, cy } = this.toCanvasCoord(x, y);
        this.ctx.fillStyle = color;
        this.ctx.font = `${size}px 'JetBrains Mono', monospace`;
        this.ctx.textAlign = align;
        this.ctx.fillText(text, cx, cy);
    }

    // Draw a connecting line between two data points
    drawConnection(x1, y1, x2, y2, color = '#6366f1', width = 1) {
        const p1 = this.toCanvasCoord(x1, y1);
        const p2 = this.toCanvasCoord(x2, y2);
        this.ctx.beginPath();
        this.ctx.moveTo(p1.cx, p1.cy);
        this.ctx.lineTo(p2.cx, p2.cy);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = width;
        this.ctx.stroke();
    }

    // Draw axis-aligned split line for decision trees
    drawSplitLine(axis, value, min, max, color = 'rgba(255,255,255,0.3)', width = 1) {
        this.ctx.beginPath();
        this.ctx.setLineDash([6, 4]);
        if (axis === 0) { // vertical line at x=value
            const p1 = this.toCanvasCoord(value, min);
            const p2 = this.toCanvasCoord(value, max);
            this.ctx.moveTo(p1.cx, p1.cy);
            this.ctx.lineTo(p2.cx, p2.cy);
        } else { // horizontal line at y=value
            const p1 = this.toCanvasCoord(min, value);
            const p2 = this.toCanvasCoord(max, value);
            this.ctx.moveTo(p1.cx, p1.cy);
            this.ctx.lineTo(p2.cx, p2.cy);
        }
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = width;
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    // Get ImageData for pixel-wise operations (KNN decision boundary)
    getImageData() {
        return this.ctx.createImageData(this.width, this.height);
    }

    putImageData(imgData) {
        this.ctx.putImageData(imgData, 0, 0);
    }
}
