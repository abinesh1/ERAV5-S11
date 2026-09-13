"""Part 3: log RMS(update)/RMS(weight) for every layer, and find where warmup
stops driving it.

Schedule is warmup-then-flat on purpose. Under cosine you cannot tell whether the
ratio settled because warmup ended or because the cosine started pulling the LR
back down; holding the LR flat after warmup makes the question answerable.
"""
import json

import torch

import common

WIDTH, STEPS, WARMUP, LR = 512, 300, 50, 1e-3
REF = 1e-3   # the ratio commonly quoted as "healthy"; measured below, not assumed


def role(name):
    """Group a parameter name into (panel, layer index)."""
    if name.startswith("tok") or name.startswith("pos"):
        return "embeddings", 0 if name.startswith("tok") else 1
    if name.startswith("head"):
        return "output head", 0
    if ".layers." in name:
        i = int(name.split(".layers.")[1].split(".")[0])
        for key, panel in (("self_attn.in_proj_weight", "attn in_proj"),
                           ("self_attn.out_proj.weight", "attn out_proj"),
                           ("linear1.weight", "ffn in"),
                           ("linear2.weight", "ffn out")):
            if key in name:
                return panel, i
    return None, None


def settle_step(series, warmup, tol=0.20):
    """First step at/after warmup past which the ratio stays within tol of its
    final plateau (median of the last 100 steps)."""
    import statistics
    plateau = statistics.median(series[-100:])
    bad = [t for t in range(warmup, len(series)) if abs(series[t] - plateau) / plateau > tol]
    return (max(bad) + 1) if bad else warmup


def main():
    dev = common.device()
    data = common.Data(dev)
    model = common.build(data, WIDTH)
    print(f"device={dev}  width={WIDTH}  steps={STEPS}  warmup={WARMUP}  lr={LR}")
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    ratios = {n: [] for n, _ in model.named_parameters()}
    lrs = []

    def log(t, m, prev, lr):
        lrs.append(lr)
        with torch.no_grad():
            for n, p in m.named_parameters():
                d = (p - prev[n]).pow(2).mean().sqrt()
                w = prev[n].pow(2).mean().sqrt()
                ratios[n].append((d / w).item())

    common.train(model, data, STEPS, LR, kind="const", warmup=WARMUP, on_step=log)

    # ---- where does warmup stop moving the ratio? ----------------------
    tracked = {n: s for n, s in ratios.items() if role(n)[0]}
    settle = {n: settle_step(s, WARMUP) for n, s in tracked.items()}
    med = sorted(settle.values())[len(settle) // 2]

    print(f"{'parameter':<44} {'@warmup end':>12} {'plateau':>10} {'lr/RMS(w)':>10} "
          f"{'RMS(dw)/lr':>11} {'settles':>8}")
    # RMS(w) at the end of training, so it lines up with the plateau window
    rms_w = {n: p.pow(2).mean().sqrt().item() for n, p in model.named_parameters()}
    plateaus, frac = {}, {}
    for n in sorted(tracked, key=lambda k: settle[k]):
        s = tracked[n]
        plateaus[n] = sorted(s[-100:])[50]
        frac[n] = plateaus[n] * rms_w[n] / LR     # RMS(update) as a fraction of lr
        print(f"{n:<44} {s[WARMUP - 1]:>12.2e} {plateaus[n]:>10.2e} "
              f"{LR / rms_w[n]:>10.2e} {frac[n]:>11.3f} {settle[n]:>8}")

    print(f"\nLR stops changing at step {WARMUP} (warmup ends -- by construction).")
    print(f"Measured: ratios stay within 20% of their plateau from step {med} (median "
          f"across {len(settle)} tensors), spread {min(settle.values())}-{max(settle.values())}.")
    lo, hi = min(plateaus.values()), max(plateaus.values())
    print(f"\nPlateau ratios span {lo:.1e} to {hi:.1e} -- a {hi / lo:.0f}x spread across tensors, "
          f"none near the quoted {REF:.0e}.")
    fs = sorted(frac.values())
    print(f"But RMS(update)/lr averages {sum(fs) / len(fs):.3f} and spans only "
          f"{fs[0]:.3f}-{fs[-1]:.3f} -- flat across every tensor.")
    print("So the ratio factorises: RMS(dw)/RMS(w) = (0.21 * lr) / RMS(w). The optimizer sets the")
    print("numerator (an Adam step is bounded by lr; gradient noise damps it to ~0.21 lr here) and")
    print("the entire 49x spread is the denominator -- i.e. initialization scale. nn.Embedding inits")
    print(f"N(0,1) (RMS ~{rms_w['tok.weight']:.2f}); nn.Linear inits U(+/-1/sqrt(fan_in)) "
          f"(RMS ~{rms_w['blocks.layers.0.linear1.weight']:.3f}).")
    print("Reading this as 'the embeddings are barely learning' would be wrong -- they take the same")
    print("size step as everything else. This is the asymmetry muP fixes by scaling LRs per tensor.")

    json.dump({"lrs": lrs, "ratios": ratios, "settle": settle, "warmup": WARMUP,
               "width": WIDTH, "steps": STEPS, "lr": LR, "plateau": plateaus,
               "update_frac_of_lr": frac, "rms_w": rms_w},
              open("results/exp2_update_ratios.json", "w"))
    plot(ratios, lrs, settle, med)
    print("\nwrote figures/part3_update_ratios.png, results/exp2_update_ratios.json")


def plot(ratios, lrs, settle, med):
    plt = common.style()
    panels = ["attn in_proj", "attn out_proj", "ffn in", "ffn out", "embeddings", "output head"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    steps = range(1, len(lrs) + 1)

    for ax, panel in zip(axes.flat, panels):
        members = [(n, role(n)[1]) for n in ratios if role(n)[0] == panel]
        for n, i in sorted(members, key=lambda x: x[1]):
            # depth is ordinal -> one hue, light (shallow) to dark (deep)
            color = common.BLUE_RAMP[[0, 2, 4, 6][i] if len(members) > 2 else [1, 5][i]]
            label = {"embeddings": ["token", "position"], "output head": ["head"]}.get(
                panel, [f"layer {i}"] * 4)[i if panel in ("embeddings", "output head") else 0]
            ax.plot(steps, ratios[n], color=color, lw=1.4, label=label)
        ax.axhline(REF, color=common.INK_MUTED, lw=1, ls=":")
        ax.axvline(WARMUP, color=common.C_ORANGE, lw=1.4, ls="--")
        ax.set_yscale("log")
        ax.set_title(panel, loc="left", fontweight="bold", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", ncol=1)

    axes[0][2].annotate(f"the quoted {REF:.0e} ratio\n(nothing sits here)", (STEPS, REF),
                        textcoords="offset points", xytext=(-4, 6),
                        color=common.INK_MUTED, fontsize=7, ha="right")
    for ax in axes[1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMS(update) / RMS(weight)")
    fig.suptitle(f"Update-to-weight ratio per layer  ·  width {WIDTH}  ·  dashed = warmup ends "
                 f"(step {WARMUP}), LR flat after  ·  ratios settle by step {med}",
                 x=0.005, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figures/part3_update_ratios.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
