"""Submission figures, regenerated from recorded artifacts (never typed-in numbers).

Writes PNGs plus a JSON data table per figure to v2/submission/figures/. Static
light-mode images for the Devpost gallery. Colours follow the entity in every
figure (V2 = categorical slot 1 blue, V1 = slot 2 orange; pair validated with the
dataviz validator). Rerun after the main run finishes to refresh figure 4.

Run: PYTHONPATH=src python -m scglm_v2.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from .common import ROOT  # noqa: E402

OUT = ROOT / "v2/submission/figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
V2, V1 = "#2a78d6", "#eb6834"
DPI, SCALE = 100, 2  # sizes are in CSS-like px at 100 dpi, saved at 2x
BAR_PX = 22

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 11,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.edgecolor": AXIS, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _px_to_data(ax):
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    return abs(x1 - x0) / bbox.width, abs(y1 - y0) / bbox.height


def _hbar(ax, y, value, color, height_px=BAR_PX, radius_px=4):
    """Horizontal bar from 0: 4px rounded data-end, square at the baseline."""
    dx, dy = _px_to_data(ax)
    h, rx, ry = height_px * dy, radius_px * dx, radius_px * dy
    ax.add_patch(FancyBboxPatch((0, y - h / 2), value, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((0, y - h / 2), min(value, 2 * rx), h, facecolor=color, edgecolor="none", zorder=3))


def _vbar(ax, x, value, color, width_px, radius_px=4):
    """Vertical bar from 0: 4px rounded top, square at the baseline."""
    dx, dy = _px_to_data(ax)
    w, rx, ry = width_px * dx, radius_px * dx, radius_px * dy
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, value, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - w / 2, 0), w, min(value, 2 * ry), facecolor=color, edgecolor="none", zorder=3))


def _save(fig, name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI * SCALE, facecolor=SURFACE)
    plt.close(fig)
    (OUT / f"{name}.data.json").write_text(json.dumps(data, indent=2) + "\n")


def _load(path):
    return json.loads((ROOT / path).read_text())


# ------------------------------------------------------------------ figure 1
def key_numbers():
    status = _load("v2/runs/main/status.json")
    v1 = _load("runs/continuation_12b/status.json")
    manifest = _load("v2/data/rich_v1/manifest.json")
    v1_rate = v1["main_tokens"] / v1["elapsed_seconds"]
    # Median logged rate over the run (status.json omits the rate right after a checkpoint).
    rates = sorted(json.loads(l)["tokens_per_second"] for l in (ROOT / "v2/runs/main/events.jsonl").open()
                   if '"event": "train"' in l and json.loads(l)["step"] > 1)
    v2_rate = rates[len(rates) // 2]
    done = status["status"] == "completed"
    tiles = [
        ("Parameters (cap 50M)", f"{status['parameter_count'] / 1e6:.1f}M", "tied embeddings, 12 layers x 512"),
        ("Pretraining tokens", f"{status['cumulative_tokens'] / 1e9:.1f}B" if done else "45B",
         "V1: 13.0B" if done else f"planned (V1: 13.0B) · {status['cumulative_tokens'] / 1e9:.1f}B done"),
        ("Training throughput", f"{v2_rate / 1e3:.0f}k tok/s", f"{v2_rate / v1_rate:.1f}x V1 ({v1_rate / 1e3:.0f}k) · one H100"),
        ("Unique corpus tokens", f"{manifest['total_train_tokens'] / 1e9:.1f}B", "6 human-written sources"),
    ]
    fig = plt.figure(figsize=(9.6, 2.4), dpi=DPI)
    for i, (label, value, sub) in enumerate(tiles):
        x = 0.02 + i * 0.245
        fig.text(x, 0.72, label, fontsize=11, color=INK2)
        fig.text(x, 0.40, value, fontsize=24, color=INK, fontweight="bold")
        fig.text(x, 0.18, sub, fontsize=9.5, color=MUTED)
    _save(fig, "fig1_key_numbers", {"tiles": [dict(zip(("label", "value", "detail"), t)) for t in tiles],
                                    "sources": ["v2/runs/main/status.json", "runs/continuation_12b/status.json",
                                                "v2/data/rich_v1/manifest.json"]})


# ------------------------------------------------------------------ figure 2
LABELS = {"dclm": "DCLM (web)", "edu": "FineWeb-Edu", "targeted": "Benchmark-targeted", "wiki": "Wikipedia",
          "stackexchange": "StackExchange", "books": "Gutenberg books"}


def corpus():
    manifest = _load("v2/data/rich_v1/manifest.json")
    shares = manifest["planned_training_shares"]
    rows = sorted(((b, s["train"]["tokens"]) for b, s in manifest["sources"].items()), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=DPI)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.84, bottom=0.14)
    ax.set_xlim(0, max(v for _, v in rows) / 1e9 * 1.42)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    _style(ax)
    for i, (bucket, tokens) in enumerate(rows):
        _hbar(ax, i, tokens / 1e9, V2)
        ax.text(tokens / 1e9 + ax.get_xlim()[1] * 0.012, i, f"{tokens / 1e9:.2f}B  ·  {shares[bucket]:.0%} of training",
                va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(len(rows)), [LABELS[b] for b, _ in rows])
    ax.set_xlabel("Unique training tokens (billions)")
    fig.text(0.02, 0.93, f"V2 corpus: {manifest['total_train_tokens'] / 1e9:.2f}B unique tokens, deduplicated and benchmark-filtered",
             fontsize=12.5, color=INK, fontweight="bold")
    _save(fig, "fig2_corpus", {"rows": [{"bucket": b, "label": LABELS[b], "unique_tokens": t, "training_share": shares[b]}
                                         for b, t in reversed(rows)], "source": "v2/data/rich_v1/manifest.json"})


# ------------------------------------------------------------------ figure 3
def pilot():
    gate = _load("v2/phases/pilot_1b/gate.json")
    tasks = [t for t in ("arc_easy", "piqa", "hellaswag", "winogrande", "sciq", "openbookqa", "commonsense_qa", "arc_challenge")
             if t in gate["per_task_acc"]]
    names = {"arc_easy": "ARC-\nEasy", "piqa": "PIQA", "hellaswag": "Hella-\nSwag", "winogrande": "Wino-\nGrande",
             "sciq": "SciQ", "openbookqa": "Open-\nBookQA", "commonsense_qa": "CSQA", "arc_challenge": "ARC-\nChall."}
    srcs = ["edu", "dclm", "targeted", "wiki", "stackexchange", "books"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=DPI, gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.17, wspace=0.18)
    for ax, keys, labels, get, title, ylab in (
            (a1, tasks, [names[t] for t in tasks], lambda m, k: gate["per_task_acc"][k][m] * 100,
             "Zero-shot accuracy, held-out benchmark train items (%)", "Accuracy (%)"),
            (a2, srcs, ["Fine-\nWeb-Edu", "DCLM", "Tar-\ngeted", "Wiki-\npedia", "Stack-\nExchange", "Books"],
             lambda m, k: gate["v1_1b" if m == "v1_1b" else "v2_pilot"]["bits_per_byte"][k],
             "Bits per byte on V2 development panels (lower is better)", "Bits per byte")):
        top = max(max(get("v1_1b", k), get("v2_pilot", k)) for k in keys)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_ylim(0, top * 1.12)
        _style(ax, grid_axis="y")
        for i, k in enumerate(keys):
            _vbar(ax, i - 0.2, get("v1_1b", k), V1, width_px=15)
            _vbar(ax, i + 0.2, get("v2_pilot", k), V2, width_px=15)
        ax.set_xticks(range(len(keys)), labels, rotation=0, fontsize=9.5)
        ax.set_title(title, fontsize=11, color=INK2, loc="left", pad=8)
        ax.set_ylabel(ylab)
    proxy = gate["proxy_mean_acc"]
    fig.text(0.02, 0.93, f"1B-token pilot: V2 data vs V1 data at the same model, seed and schedule "
             f"(4-task proxy {proxy['v2_pilot'] * 100:.1f}% vs {proxy['v1_1b'] * 100:.1f}%)",
             fontsize=12.5, color=INK, fontweight="bold")
    for x, color, label in ((0.02, V1, "V1 1B (FineWeb 80% / Wikipedia 20%)"), (0.32, V2, "V2 pilot (rich corpus)")):
        fig.patches.append(Rectangle((x, 0.855), 0.012, 0.03, transform=fig.transFigure, facecolor=color, edgecolor="none"))
        fig.text(x + 0.018, 0.858, label, fontsize=10, color=INK2)
    _save(fig, "fig3_pilot", {"per_task_accuracy": gate["per_task_acc"],
                              "bits_per_byte": {"v1_1b": gate["v1_1b"]["bits_per_byte"], "v2_pilot": gate["v2_pilot"]["bits_per_byte"]},
                              "proxy_mean_acc": proxy, "source": "v2/phases/pilot_1b/gate.json",
                              "note": "Development data only; official evaluation splits not used."})


# ------------------------------------------------------------------ figure 4
def training_curve():
    train, val = {}, {}
    for line in (ROOT / "v2/runs/main/events.jsonl").open():
        r = json.loads(line)
        if r["event"] == "train":
            train[r["step"]] = r["loss"]  # later lines (after a resume) replace earlier ones
        elif r["event"] == "validation":
            val[r["step"]] = r["metrics"]["fixed_q_nll"]
    steps = sorted(train)
    window = 50
    smooth = [sum(train[s] for s in steps[max(0, i - window + 1):i + 1]) / len(steps[max(0, i - window + 1):i + 1])
              for i in range(len(steps))]
    tok = lambda s: s * 65536 / 1e9  # noqa: E731
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=DPI)
    fig.subplots_adjust(left=0.08, right=0.84, top=0.82, bottom=0.14)
    _style(ax, grid_axis="y")
    ax.plot([tok(s) for s in steps], smooth, color=V2, linewidth=2, zorder=3)
    vs = sorted(val)
    ax.plot([tok(s) for s in vs], [val[s] for s in vs], linestyle="none", marker="o", markersize=8,
            markerfacecolor=V1, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.set_ylim(min(smooth[len(smooth) // 20:] + [val[s] for s in vs]) * 0.97, 4.6)
    ax.set_xlim(0, 45)
    ax.set_xlabel("Training tokens (billions)")
    ax.set_ylabel("Cross-entropy (nats per token)")
    ax.axvline(40.5, color=AXIS, linewidth=1, zorder=1)
    ax.text(40.6, ax.get_ylim()[1] * 0.995, "decay\nstarts", va="top", fontsize=9, color=MUTED)
    # The two measures converge by design, so end labels would collide: the legend carries identity.
    fig.text(0.02, 0.92, f"V2 main run: {tok(steps[-1]):.1f}B of 45B tokens", fontsize=12.5, color=INK, fontweight="bold")
    for x, color, label, marker in ((0.02, V2, f"Training loss ({window}-log moving average)", "line"),
                                    (0.36, V1, "Held-out validation NLL (monitor panels)", "dot")):
        if marker == "line":
            fig.lines.append(plt.Line2D([x, x + 0.022], [0.872, 0.872], transform=fig.transFigure, color=color, linewidth=2))
        else:
            fig.lines.append(plt.Line2D([x + 0.011], [0.872], transform=fig.transFigure, marker="o", markersize=7,
                                        color=color, linestyle="none"))
        fig.text(x + 0.03, 0.86, label, fontsize=10, color=INK2)
    _save(fig, "fig4_training_curve", {"train_loss": {str(s): train[s] for s in steps},
                                       "validation_nll": {str(s): val[s] for s in vs}, "tokens_per_step": 65536,
                                       "source": "v2/runs/main/events.jsonl"})


def main():
    key_numbers(); corpus(); pilot(); training_curve()
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
