"""Part 5: sweep the LR at three widths, find the three minima, predict width 4096.

Standard parameterization, so the optimum is expected to move with width -- that
movement is the measurement. The extrapolation to 4096 is a two-octave reach from
a three-point fit, so the script also reports how much the prediction moves when
each measured minimum is nudged by half a grid step. That sensitivity, not the
fit's R-squared, is the honest confidence statement.

The grid is per width because it has to be: a shared grid left width 256 pinned at
its top edge, and a minimum you have only bounded is not a minimum you can fit.
`--resume` reuses every (width, lr) already in the JSON and trains only what is new.
"""
import json
import math
import os
import sys

import numpy as np

import common

WIDTHS = [256, 512, 1024]
BASE = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3, 6.4e-3]
GRID = {256: BASE + [1.28e-2, 2.56e-2],   # 256's optimum sits above the shared top edge
        512: BASE,
        1024: BASE}
STEPS, WARMUP, TARGET = 300, 20, 4096
JSON = "results/exp4_lr_width.json"


def key(lr):
    return f"{lr:.6g}"


def load_cache():
    """Previously measured (width, lr) -> val loss. Tolerates the pre-resume format."""
    if "--resume" not in sys.argv or not os.path.exists(JSON):
        return {}
    o = json.load(open(JSON))
    loss = o.get("loss", {})
    if loss and isinstance(next(iter(loss.values())), list):   # legacy: list aligned to one grid
        loss = {w: {key(lr): v for lr, v in zip(o["grid"], row)} for w, row in loss.items()}
    return loss


def refine(lrs, losses):
    """Parabola in log-LR through the minimum and its two neighbours.

    The grid is a factor of 2 apart, so the grid argmin is only good to +/-sqrt(2);
    the vertex of the local parabola is the usable estimate.
    """
    i = int(np.argmin(losses))
    if i in (0, len(lrs) - 1):
        return lrs[i], True            # minimum is at the edge -- grid was too narrow
    x = np.log10(lrs[i - 1:i + 2])
    a, b, _ = np.polyfit(x, losses[i - 1:i + 2], 2)
    return float(10 ** (-b / (2 * a))), False


def main():
    dev = common.device()
    data = common.Data(dev)
    cached = load_cache()
    print(f"device={dev}  widths={WIDTHS}  {STEPS} steps"
          + (f"  (resuming: {sum(len(v) for v in cached.values())} points cached)" if cached else ""))

    out = {"widths": WIDTHS, "grid": {str(w): GRID[w] for w in WIDTHS}, "loss": {}, "min": {}}
    for w in WIDTHS:
        print()
        row = {}
        for lr in GRID[w]:
            hit = cached.get(str(w), {}).get(key(lr))
            if hit is not None:
                row[key(lr)] = hit
                print(f"width {w:>5}  lr {lr:>8.1e}  val {hit:>8.4f}   (cached)")
                continue
            model = common.build(data, w)
            _, ev, _, _ = common.train(model, data, STEPS, lr, kind="cosine",
                                       warmup=WARMUP, eval_every=STEPS)
            v = ev[-1][1] if ev and math.isfinite(ev[-1][1]) else float("nan")
            row[key(lr)] = v
            print(f"width {w:>5}  lr {lr:>8.1e}  val {v:>8.4f}"
                  + ("  DIVERGED" if math.isnan(v) else ""))
        out["loss"][str(w)] = row

        lrs = sorted(float(k) for k in row)
        vals = [row[key(lr)] if math.isfinite(row[key(lr)]) else 1e9 for lr in lrs]
        lr_star, edge = refine(lrs, vals)
        out["min"][str(w)] = {"lr_star": lr_star, "grid_argmin": lrs[int(np.argmin(vals))],
                              "best_loss": min(vals), "at_edge": edge}
        print(f"  -> width {w}: grid argmin {lrs[int(np.argmin(vals))]:.2e}, "
              f"parabola vertex {lr_star:.2e}, best val {min(vals):.4f}"
              + ("   [AT GRID EDGE -- widen the grid]" if edge else ""))

    edges = [w for w in WIDTHS if out["min"][str(w)]["at_edge"]]
    if edges:
        print(f"\nWARNING: width(s) {edges} still pinned at a grid edge; the fit below is "
              f"not trustworthy until the grid is widened there.")

    # ---- how does the optimum move with width? ------------------------
    lw = np.log2(WIDTHS)
    ll = np.log2([out["min"][str(w)]["lr_star"] for w in WIDTHS])
    slope, icpt = np.polyfit(lw, ll, 1)
    pred = 2 ** (slope * math.log2(TARGET) + icpt)
    resid = ll - (slope * lw + icpt)
    r2 = 1 - resid.var() / ll.var()

    print("\n" + "=" * 72)
    print(f"optimal LR vs width:  log2(lr*) = {slope:.3f} * log2(width) + {icpt:.3f}")
    print(f"slope {slope:.3f}  ->  lr* scales as width^{slope:.2f}   (R^2 = {r2:.4f}, 3 points, 1 dof)")
    print(f"  theory: standard parameterization predicts ~width^-1; muP would predict ~width^0")
    print(f"\nprediction at width {TARGET}:  lr* = {pred:.2e}")

    # ---- how much do I believe it? ------------------------------------
    half = math.log2(math.sqrt(2))          # half a grid step, in log2 units
    lo = []
    for signs in [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]:
        s2, i2 = np.polyfit(lw, ll + half * np.array(signs), 1)
        lo.append(2 ** (s2 * math.log2(TARGET) + i2))
    print(f"nudging each measured minimum by +/- half a grid step moves the prediction to "
          f"[{min(lo):.2e}, {max(lo):.2e}]  ({max(lo) / min(lo):.1f}x span)")
    print("that span is one two-octave extrapolation from 3 points -- see README for the call.")
    out["fit"] = {"slope": float(slope), "intercept": float(icpt), "r2": float(r2),
                  "pred_4096": float(pred), "pred_range": [float(min(lo)), float(max(lo))]}

    json.dump(out, open(JSON, "w"), indent=1)
    plot(out)
    print("\nwrote figures/part5_lr_vs_width.png, " + JSON)


def plot(out):
    plt = common.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    cols = [common.C_BLUE, common.C_ORANGE, common.C_AQUA]

    # the three minima sit on a diagonal, so alternate the callouts to keep them apart
    callout = {256: (0, 16, "center"), 512: (0, -34, "center"), 1024: (-10, 14, "right")}
    for w, c in zip(WIDTHS, cols):
        row = out["loss"][str(w)]
        lrs = sorted(float(k) for k in row)
        y = [row[key(lr)] if math.isfinite(row[key(lr)]) else None for lr in lrs]
        ax1.plot(lrs, y, color=c, marker="o", ms=5, label=f"width {w}")
        m = out["min"][str(w)]
        ax1.axvline(m["lr_star"], color=c, lw=1, ls=":")
        ax1.plot([m["grid_argmin"]], [m["best_loss"]], marker="o", ms=12, mfc="none", mec=c, mew=2)
        dx, dy, ha = callout[w]
        ax1.annotate(f"width {w}\nlr* {m['lr_star']:.1e}", (m["grid_argmin"], m["best_loss"]),
                     xytext=(dx, dy), textcoords="offset points", ha=ha,
                     color=c, fontsize=8, fontweight="bold")
    ax1.set_xscale("log"); ax1.set_xlabel("learning rate"); ax1.set_ylabel(f"val loss @ {STEPS}")
    ax1.set_title("a. Three sweeps, three minima", loc="left", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)

    f = out["fit"]
    xs = np.array([256, 512, 1024, TARGET])
    ax2.plot(xs, 2 ** (f["slope"] * np.log2(xs) + f["intercept"]),
             color=common.INK_MUTED, lw=1.5, ls="--", label=f"fit: width^{f['slope']:.2f}")
    for w, c in zip(WIDTHS, cols):
        ax2.plot([w], [out["min"][str(w)]["lr_star"]], marker="o", ms=8, color=c)
    ax2.fill_between([TARGET * 0.9, TARGET * 1.1], f["pred_range"][0], f["pred_range"][1],
                     color=common.C_AQUA, alpha=0.25, lw=0)
    ax2.plot([TARGET], [f["pred_4096"]], marker="*", ms=16, color=common.C_AQUA,
             mec=common.INK, mew=0.5, label=f"predicted {f['pred_4096']:.1e}")
    ax2.annotate(f"width {TARGET}: {f['pred_4096']:.1e}\n(half-step range "
                 f"{f['pred_range'][0]:.0e}–{f['pred_range'][1]:.0e})",
                 (TARGET, f["pred_4096"]), xytext=(-10, -4), textcoords="offset points",
                 ha="right", va="top", color=common.INK, fontsize=8, fontweight="bold")
    ax2.axvspan(1024, TARGET * 1.35, color=common.INK_MUTED, alpha=0.06, lw=0)
    ax2.annotate("extrapolated,\nnot measured", (1500, f["pred_4096"] * 5.0),
                 color=common.INK_MUTED, fontsize=8)
    ax2.set_xscale("log", base=2); ax2.set_yscale("log")
    ax2.set_xlabel("width (d_model)"); ax2.set_ylabel("optimal learning rate")
    ax2.set_title("b. Where the optimum goes", loc="left", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/part5_lr_vs_width.png", bbox_inches="tight")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        plot(json.load(open(JSON)))
        print("replotted figures/part5_lr_vs_width.png from " + JSON)
    else:
        main()
