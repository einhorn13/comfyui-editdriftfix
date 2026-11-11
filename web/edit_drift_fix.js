// Edit Drift Fix: preset <-> widgets sync and a one-line result summary on the node.
import { app } from "../../scripts/app.js";

// Mirror of PRESETS in edit_drift_fix/pipeline.py (the Python side is authoritative at run time).
const PRESETS = {
    spot_edit:       { mask_mode: "auto", sensitivity: 1.0, minor_filter: 0.5, mask_expand: 0.5, feather: 0.4, color_fix: "global_plus_local", grain_match: 0.5 },
    object_edit:     { mask_mode: "auto", sensitivity: 1.2, minor_filter: 0.5, mask_expand: 1.2, feather: 1.0, color_fix: "global_plus_local", grain_match: 0.5 },
    large_soft_edit: { mask_mode: "auto", sensitivity: 1.5, minor_filter: 0.5, mask_expand: 2.5, feather: 2.5, color_fix: "global", grain_match: 0.3 },
    full_image:      { mask_mode: "whole_image", sensitivity: 1.0, minor_filter: 0.0, mask_expand: 0.0, feather: 0.0, color_fix: "off", grain_match: 0.0 },
};

const same = (a, b) => (typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-6 : a === b);

app.registerExtension({
    name: "EditDriftFix.Presets",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "EditDriftFix") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;
            const widget = (name) => node.widgets?.find((w) => w.name === name);
            const preset = widget("preset");
            if (!preset) return;
            let applying = false;

            const presetCallback = preset.callback;
            preset.callback = function (value) {
                presetCallback?.apply(this, arguments);
                const values = PRESETS[value];
                if (!values) return;
                applying = true;
                for (const [name, v] of Object.entries(values)) {
                    const w = widget(name);
                    if (w) w.value = v;
                }
                applying = false;
                node.setDirtyCanvas(true, true);
            };

            // editing any controlled widget by hand turns the preset into "custom"
            for (const name of Object.keys(PRESETS.spot_edit)) {
                const w = widget(name);
                if (!w) continue;
                const cb = w.callback;
                w.callback = function (value) {
                    cb?.apply(this, arguments);
                    const values = PRESETS[preset.value];
                    if (!applying && values && !same(values[name], w.value)) {
                        preset.value = "custom";
                        node.setDirtyCanvas(true, true);
                    }
                };
            }
        };

        // show "model · drift · mask · confidence" under the widgets after a run
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text?.[0];
            if (text === undefined) return;
            this.edfSummary = text;
            this.setDirtyCanvas(true, true);
        };
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (!this.edfSummary || this.flags?.collapsed) return;
            ctx.save();
            ctx.font = "11px sans-serif";
            ctx.fillStyle = this.edfSummary.includes("⚠") || this.edfSummary.includes("GLOBAL") ? "#e0a030" : "#8c8";
            ctx.textAlign = "left";
            const line = this.edfSummary.split("\n")[0];
            ctx.fillText(line, 10, this.size[1] - 8, this.size[0] - 20);
            ctx.restore();
        };
        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const size = computeSize?.apply(this, arguments) ?? [240, 200];
            size[1] += 18; // room for the summary line
            return size;
        };
    },
});
