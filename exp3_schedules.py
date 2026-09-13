"""Part 4: cosine vs WSD, both tuned, both stopped at step 200.

The instruction that matters here is "tune both sides". A schedule comparison at
one shared LR measures the LR, not the schedule -- so each schedule gets its own
sweep at the planned 300-step budget first, and only the two winners meet.

Three numbers come out, not two. Stopping a cosine run at 200 gives you a model
whose LR never annealed; stopping WSD at 200 gives you one still at full LR. The
third number is WSD's actual claim: take that step-200 checkpoint and decay it to
zero over 40 steps. Cosine at step 240 is the fair-compute control for it.
"""
import copy
import json

import torch

import common

WIDTH, STEPS, STOP, ANNEAL, WARMUP = 512, 300, 200, 40, 20
GRID = [3e-4, 6e-4, 1e-3, 2e-3, 3e-3]
KINDS = ["cosine", "wsd"]


def run(data, lr, kind, steps=STEPS, **kw):
    model = common.build(data, WIDTH)
    return model, common.train(model, data, steps, lr, kind=kind, warmup=WARMUP, **kw)


def main():
    dev = common.device()
    data = common.Data(dev)
    print(f"device={dev}  width={WIDTH}  budget={STEPS} steps  stop at {STOP}  warmup={WARMUP}")

    # ---- tune both sides ----------------------------------------------
    print(f"\n{'=' * 74}\nTUNING -- each schedule swept over the same grid at the 300-step budget")
    print(f"{'=' * 74}\n{'schedule':<9} {'lr':>8} {'val@100':>9} {'val@200':>9} {'val@300':>9}")
    tune = {k: {} for k in KINDS}
    for kind in KINDS:
        for lr in GRID:
            _, (_, ev, _, _) = run(data, lr, kind, eval_every=100)
            tune[kind][lr] = {str(s): v for s, v in ev}
            print(f"{kind:<9} {lr:>8.0e} " + " ".join(f"{v:>9.4f}" for _, v in ev))

    best = {k: min(GRID, key=lambda lr: tune[k][lr]["300"]) for k in KINDS}
    best200 = {k: min(GRID, key=lambda lr: tune[k][lr]["200"]) for k in KINDS}
    print(f"\nbest LR at the planned 300-step budget: " +
          ", ".join(f"{k} {best[k]:.0e}" for k in KINDS))
    print(f"best LR if you had tuned for 200 instead:  " +
          ", ".join(f"{k} {best200[k]:.0e}" for k in KINDS))
    for k in KINDS:
        note = "same" if best[k] == best200[k] else "DIFFERENT -- the stop point changes the tune"
        print(f"  {k}: {note}")

    # ---- the two tuned runs, stopped at 200 ---------------------------
    print(f"\n{'=' * 74}\nHEAD TO HEAD -- each at its own best LR, identical data order\n{'=' * 74}")
    out = {"tune": tune, "best": best, "best200": best200, "curves": {}, "final": {}}
    ckpt = {}
    for kind in KINDS:
        model, (_, ev, opt, gen) = run(data, best[kind], kind, steps=STOP, total=STEPS, eval_every=20)
        ckpt[kind] = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()),
                      gen.get_state(), model, opt, gen)
        out["curves"][kind] = ev
        out["final"][f"{kind}@{STOP}"] = ev[-1][1]
        print(f"{kind:<9} lr={best[kind]:.0e}  val@{STOP} = {ev[-1][1]:.4f}   "
              f"(LR at step {STOP} was {common.lr_at(STOP, STEPS, best[kind], kind, WARMUP):.2e})")

    # ---- let them finish the planned 300, for context -----------------
    for kind in KINDS:
        _, _, model, opt, gen = ckpt[kind][1:]
        _, ev, _, _ = common.train(model, data, STEPS - STOP, best[kind], kind=kind,
                                   warmup=WARMUP, opt=opt, gen=gen, start=STOP,
                                   total=STEPS, eval_every=20)
        out["curves"][kind] += ev
        out["final"][f"{kind}@{STEPS}"] = ev[-1][1]
        print(f"{kind:<9} if allowed to finish:  val@{STEPS} = {ev[-1][1]:.4f}")

    # ---- WSD's actual claim: anneal the step-200 checkpoint -----------
    print(f"\n{'=' * 74}\nANNEAL -- decay the WSD step-{STOP} checkpoint to zero over {ANNEAL} steps")
    print("=" * 74)
    msd, osd, gstate = ckpt["wsd"][:3]
    m2 = common.build(data, WIDTH)
    m2.load_state_dict(msd)
    o2 = torch.optim.AdamW(m2.parameters(), lr=best["wsd"], betas=(0.9, 0.95), weight_decay=0.1)
    o2.load_state_dict(osd)
    g2 = torch.Generator(device=dev)
    g2.set_state(gstate)
    # decay_frac=1.0 + warmup=0 -> linear from the stable LR to zero across ANNEAL steps
    _, ev, _, _ = common.train(m2, data, ANNEAL, best["wsd"], kind="wsd", decay_frac=1.0,
                               warmup=0, opt=o2, gen=g2, total=ANNEAL, eval_every=10)
    out["curves"]["wsd_annealed"] = [(STOP + s, v) for s, v in ev]
    out["final"][f"wsd@{STOP}+{ANNEAL}anneal"] = ev[-1][1]

    cos240 = dict(out["curves"]["cosine"])[STOP + ANNEAL]
    out["final"][f"cosine@{STOP + ANNEAL}"] = cos240
    f = out["final"]
    print(f"\n{'model':<34} {'steps':>6} {'val loss':>9}")
    for k in (f"cosine@{STOP}", f"wsd@{STOP}", f"wsd@{STOP}+{ANNEAL}anneal",
              f"cosine@{STOP + ANNEAL}", f"cosine@{STEPS}", f"wsd@{STEPS}"):
        steps = STOP + ANNEAL if "anneal" in k else int(k.split("@")[1])
        print(f"{k:<34} {steps:>6} {f[k]:>9.4f}")

    print(f"\nat the stop point:     cosine {f[f'cosine@{STOP}']:.4f} vs wsd {f[f'wsd@{STOP}']:.4f}"
          f"  -> cosine ahead by {f[f'wsd@{STOP}'] - f[f'cosine@{STOP}']:.4f}")
    print(f"after annealing WSD:   {f[f'wsd@{STOP}+{ANNEAL}anneal']:.4f} vs cosine at equal compute "
          f"({STOP + ANNEAL}) {cos240:.4f}  -> "
          f"{'WSD' if f[f'wsd@{STOP}+{ANNEAL}anneal'] < cos240 else 'cosine'} ahead by "
          f"{abs(cos240 - f[f'wsd@{STOP}+{ANNEAL}anneal']):.4f}")

    json.dump(out, open("results/exp3_schedules.json", "w"), indent=1)
    plot(out)
    print("\nwrote figures/part4_schedules.png, results/exp3_schedules.json")


def plot(out):
    plt = common.style()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 4),
                                        gridspec_kw={"width_ratios": [1, 1, 1.35]})
    col = {"cosine": common.C_BLUE, "wsd": common.C_ORANGE, "wsd_annealed": common.C_AQUA}

    # (a) both sides tuned
    for kind in KINDS:
        y = [out["tune"][kind][lr]["300"] for lr in GRID]
        ax0.plot(GRID, y, color=col[kind], marker="o", ms=5, label=kind)
        b = out["best"][kind]
        ax0.plot([b], [out["tune"][kind][b]["300"]], marker="o", ms=11, mfc="none",
                 mec=col[kind], mew=2)
        dy = 20 if kind == "cosine" else -32
        ax0.annotate(f"{kind} best\n{b:.0e}", (b, out["tune"][kind][b]["300"]),
                     xytext=(-14, dy), textcoords="offset points", ha="center",
                     color=col[kind], fontsize=8, fontweight="bold")
    ax0.set_xscale("log"); ax0.set_xlabel("learning rate"); ax0.set_ylabel("val loss @ 300")
    ax0.set_title("a. Tune both sides first", loc="left", fontweight="bold")
    ax0.legend(loc="upper center")

    # (b) the schedules themselves
    for kind in KINDS:
        ax1.plot(range(STEPS), [common.lr_at(t, STEPS, out["best"][kind], kind, WARMUP)
                                for t in range(STEPS)], color=col[kind], label=kind)
    ax1.plot(range(STOP, STOP + ANNEAL),
             [common.lr_at(t, ANNEAL, out["best"]["wsd"], "wsd", 0, 1.0) for t in range(ANNEAL)],
             color=col["wsd_annealed"], label="wsd + anneal")
    ax1.axvline(STOP, color=common.INK_MUTED, lw=1, ls="--")
    ax1.annotate("stop", (STOP, 0), xytext=(4, 4), textcoords="offset points",
                 color=common.INK_MUTED, fontsize=8)
    lo = out["best"]["wsd"]
    ax1.annotate("wsd: hold, then decay", (150, lo), xytext=(0, 6), textcoords="offset points",
                 color=col["wsd"], fontsize=8, fontweight="bold", ha="center")
    ax1.annotate("cosine: decaying\nthe whole way", (105, lo * 0.42), xytext=(0, -6),
                 textcoords="offset points", color=col["cosine"], fontsize=8,
                 fontweight="bold", ha="center", va="top")
    ax1.annotate("anneal from\nthe checkpoint", (STOP + ANNEAL, lo * 0.25), xytext=(6, 10),
                 textcoords="offset points", color=col["wsd_annealed"], fontsize=8,
                 fontweight="bold")
    ax1.set_xlabel("step"); ax1.set_ylabel("learning rate")
    ax1.set_title("b. What each schedule does", loc="left", fontweight="bold")

    # (c) the result
    for kind in KINDS:
        xs, ys = zip(*sorted(out["curves"][kind]))
        ax2.plot(xs, ys, color=col[kind], label=kind)
        ax2.annotate(kind, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                     color=col[kind], fontsize=8, fontweight="bold", va="center")
    xs, ys = zip(*out["curves"]["wsd_annealed"])
    ax2.plot(xs, ys, color=col["wsd_annealed"], label="wsd + 40-step anneal")
    ax2.annotate(f"wsd + anneal\n{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(0, -26),
                 textcoords="offset points", ha="center",
                 color=col["wsd_annealed"], fontsize=8, fontweight="bold")
    ax2.axvline(STOP, color=common.INK_MUTED, lw=1, ls="--")
    for kind in KINDS:
        v = out["final"][f"{kind}@{STOP}"]
        ax2.plot([STOP], [v], marker="o", ms=7, color=col[kind], mec=common.SURFACE, mew=1.5)
        ax2.annotate(f"{v:.3f}", (STOP, v), xytext=(-7, 7 if kind == "wsd" else -13),
                     textcoords="offset points",
                     ha="right", color=col[kind], fontsize=8, fontweight="bold")
    ax2.set_xlabel("step"); ax2.set_ylabel("val loss")
    ax2.set_title(f"c. Both stopped at {STOP}", loc="left", fontweight="bold")
    ax2.set_xlim(right=STEPS + 45)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/part4_schedules.png", bbox_inches="tight")


if __name__ == "__main__":
    import sys
    if "--replot" in sys.argv:
        o = json.load(open("results/exp3_schedules.json"))
        o["tune"] = {k: {float(lr): v for lr, v in d.items()} for k, d in o["tune"].items()}
        plot(o)
        print("replotted figures/part4_schedules.png from results/exp3_schedules.json")
    else:
        main()
