import { app } from "../../../scripts/app.js";

/**
 * Z-FUSE UI Extension
 * Provides visual block-level feedback and stylized sliders for Z-Image tuning.
 */
app.registerExtension({
    name: "Z-FUSE.UI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // We only care about our specific Z-FUSE nodes
        const isTuner = nodeData.name === "ZFuseVisualLayerTuner";
        const isOrchestrator = nodeData.name === "ZFuseOrchestrator";

        if (!isTuner && !isOrchestrator) return;

        // --- EXTEND TUNER (Sliders) ---
        if (isTuner) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Color code the block sliders (0-29)
                // We use a gradient from Blue (Early/Structural) to Purple (Late/Detail)
                this.widgets.forEach(w => {
                    if (w.name.startsWith("block_")) {
                        const blockId = parseInt(w.name.split("_")[1]);
                        const hue = 200 + (blockId * 3); // 200 (Blue) to 290 (Purple)
                        
                        w.options.color = `hsla(${hue}, 50%, 50%, 0.2)`;
                        w.options.precision = 2;
                    }
                });

                return r;
            };
        }

        // --- EXTEND ORCHESTRATOR (Live Feedback) ---
        if (isOrchestrator) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;

                if (message?.ui?.analysis_report) {
                    // Update the visual bars based on the report data
                    // This expects the backend to send a structured string or JSON
                    this.setProperty("last_analysis", message.ui.analysis_report);
                }
                return r;
            };

            // Custom Draw for the Orchestrator to show a 'Heatmap'
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
                if (this.flags.collapsed) return r;

                // If we have analysis data, draw a small sparkline/heatmap at the bottom
                const analysis = this.properties?.last_analysis;
                if (analysis && Array.isArray(analysis)) {
                    const margin = 10;
                    const w = this.size[0] - (margin * 2);
                    const h = 40;
                    const x = margin;
                    const y = this.size[1] - h - margin;

                    ctx.fillStyle = "#1a1a1a";
                    ctx.fillRect(x, y, w, h);

                    const barWidth = w / analysis.length;
                    const maxVal = Math.max(...analysis, 0.0001);

                    analysis.forEach((val, i) => {
                        const barHeight = (val / maxVal) * h;
                        ctx.fillStyle = `hsla(${200 + (i * 3)}, 70%, 60%, 0.8)`;
                        ctx.fillRect(x + (i * barWidth), y + h - barHeight, barWidth - 1, barHeight);
                    });
                }
                return r;
            };
        }
    },

    /**
     * Listen for messages from the backend (the "ui" return key)
     */
    async nodeDataReceived(node, data) {
        if (node.type === "ZFuseOrchestrator" && data.analysis_report) {
            // Store the raw numeric data for the heatmap drawer
            if (!node.properties) node.properties = {};
            node.properties.last_analysis = data.analysis_report;
            node.setDirtyCanvas(true, true);
        }
    }
});