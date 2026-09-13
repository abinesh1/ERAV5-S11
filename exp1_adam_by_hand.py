"""Parts 1-2: Adam reproduced by hand, and what bias correction is worth.

Part 1 -- one weight, five gradients. Every intermediate (m, v, m-hat, v-hat,
step) computed from the paper's Algorithm 1 and checked against torch.optim.Adam.
Part 2 -- the same optimizer with bias correction switched off, twenty steps,
both trajectories plotted, and the step count after which the gap stops mattering.

torch.optim.Adam has no flag to disable bias correction, so the uncorrected arm
is the hand implementation. Part 1 is what earns the right to use it: the
corrected arm of the same function matches torch to ~1e-17, so the only
difference between the two arms in part 2 is the correction itself.
"""
import json
import math

import common

LR, B1, B2, EPS, W0 = 0.1, 0.9, 0.999, 1e-8, 0.7
GRADS = [0.5, -0.3, 0.8, 0.1, -0.6]


def adam_by_hand(w, grads, correct=True, lr=LR, b1=B1, b2=B2, eps=EPS):
    """Kingma & Ba Algorithm 1, written out. One dict per step."""
    m = v = 0.0
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g              # 1st moment, EMA of the gradient
        v = b2 * v + (1 - b2) * g * g          # 2nd moment, EMA of its square
        mh = m / (1 - b1 ** t) if correct else m
        vh = v / (1 - b2 ** t) if correct else v
        step = lr * mh / (math.sqrt(vh) + eps)
        w -= step
        yield dict(t=t, g=g, m=m, v=v, mhat=mh, vhat=vh, step=step, w=w)


def adam_torch(w0, grads, lr=LR, b1=B1, b2=B2, eps=EPS):
    """Same thing, PyTorch's own. float64 so the comparison tests the maths."""
    import torch
    p = torch.tensor([w0], dtype=torch.float64, requires_grad=True)
    p.grad = torch.zeros_like(p)
    opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps)
    out = []
    for t, g in enumerate(grads, start=1):
        p.grad.fill_(g)
        opt.step()
        s = opt.state[p]
        m, v = s["exp_avg"].item(), s["exp_avg_sq"].item()
        out.append(dict(t=t, g=g, m=m, v=v, mhat=m / (1 - b1 ** t),
                        vhat=v / (1 - b2 ** t), w=p.item()))
    return out


def ratio(t, b1=B1, b2=B2):
    """Corrected step / uncorrected step at step t.

    m and v cancel, so with eps -> 0 this depends only on t and the betas --
    not on the gradients at all. That is the whole content of part 2.
    """
    return math.sqrt(1 - b2 ** t) / (1 - b1 ** t)


def settles_at(tol, b1=B1, b2=B2, cap=100_000):
    """First step after which |ratio - 1| stays below tol forever."""
    last = max((t for t in range(1, cap) if abs(ratio(t, b1, b2) - 1) >= tol), default=0)
    return last + 1


def main():
    out = {}

    # ---- Part 1 -------------------------------------------------------
    print("=" * 92)
    print("PART 1  Adam by hand vs torch.optim.Adam")
    print(f"w0={W0}  lr={LR}  betas=({B1}, {B2})  eps={EPS}  grads={GRADS}")
    print("=" * 92)
    hand = list(adam_by_hand(W0, GRADS))
    ref = adam_torch(W0, GRADS)

    print(f"{'t':>2} {'g':>6} {'m':>12} {'v':>12} {'m_hat':>12} {'v_hat':>12} "
          f"{'step':>11} {'w':>12} {'|w-torch|':>11}")
    worst = 0.0
    for h, r in zip(hand, ref):
        d = abs(h["w"] - r["w"])
        worst = max(worst, d, abs(h["m"] - r["m"]), abs(h["v"] - r["v"]))
        print(f"{h['t']:>2} {h['g']:>6.2f} {h['m']:>12.8f} {h['v']:>12.8f} "
              f"{h['mhat']:>12.8f} {h['vhat']:>12.8f} {h['step']:>11.8f} "
              f"{h['w']:>12.8f} {d:>11.2e}")
    print(f"\nworst absolute disagreement across m, v and w, all 5 steps: {worst:.3e}")
    assert worst < 1e-12, f"hand Adam disagrees with torch by {worst:.2e}"
    print("agreement to ~16 decimal places -- float64 round-off, not a formula difference.")
    out["part1"] = {"steps": hand, "worst_abs_diff_vs_torch": worst}

    # ---- Part 2 -------------------------------------------------------
    print("\n" + "=" * 92)
    print("PART 2  bias correction on vs off, first 20 steps")
    print("=" * 92)
    g20 = [GRADS[i % len(GRADS)] for i in range(20)]   # same five gradients, cycled
    on = list(adam_by_hand(W0, g20, correct=True))
    off = list(adam_by_hand(W0, g20, correct=False))

    print(f"{'t':>3} {'step_on':>11} {'step_off':>11} {'|on/off|':>9} {'analytic':>9} "
          f"{'w_on':>11} {'w_off':>11} {'|gap|':>9}")
    for a, b in zip(on, off):
        r_meas = abs(a["step"] / b["step"])
        print(f"{a['t']:>3} {a['step']:>11.7f} {b['step']:>11.7f} {r_meas:>9.4f} "
              f"{ratio(a['t']):>9.4f} {a['w']:>11.7f} {b['w']:>11.7f} "
              f"{abs(a['w'] - b['w']):>9.5f}")

    err = max(abs(abs(a["step"] / b["step"]) - ratio(a["t"])) for a, b in zip(on, off))
    print(f"\nmeasured ratio vs sqrt(1-b2^t)/(1-b1^t): worst deviation {err:.2e}")
    print("The ratio is gradient-independent -- m and v cancel. It is a pure LR multiplier.")

    tmin = min(range(1, 21), key=ratio)
    print(f"\nworst mismatch is NOT step 1: ratio bottoms at t={tmin} "
          f"({ratio(tmin):.4f}, i.e. uncorrected steps {1 / ratio(tmin):.2f}x too large)")
    print(f"at t=1 ratio={ratio(1):.4f}   at t=20 ratio={ratio(20):.4f} "
          f"(still {1 / ratio(20):.2f}x too large)")

    print(f"\n{'beta2':>7} {'10% tol':>9} {'5% tol':>9} {'1% tol':>9}   steps until the gap stops mattering")
    tol_tbl = {}
    for b2 in (0.999, 0.99, 0.95):
        row = [settles_at(t, B1, b2) for t in (0.10, 0.05, 0.01)]
        tol_tbl[b2] = row
        print(f"{b2:>7} {row[0]:>9} {row[1]:>9} {row[2]:>9}")
    print("\nThe timescale is set by beta2 alone: t ~ log(1-(1-tol)^2)/log(beta2).")

    gap = [abs(a["w"] - b["w"]) for a, b in zip(on, off)]
    out["part2"] = {"ratio_min_step": tmin, "ratio_min": ratio(tmin), "ratio_t20": ratio(20),
                    "measured_vs_analytic_max_err": err, "weight_gap_at_20": gap[-1],
                    "settles": {str(k): v for k, v in tol_tbl.items()},
                    "w_on": [a["w"] for a in on], "w_off": [b["w"] for b in off]}

    plot(on, off, tmin)
    json.dump(out, open("results/exp1_adam.json", "w"), indent=1)
    print("\nwrote figures/part2_bias_correction.png, results/exp1_adam.json")


def plot(on, off, tmin):
    plt = common.style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    t = [a["t"] for a in on]

    ax1.plot(t, [a["w"] for a in on], color=common.C_BLUE, marker="o", ms=3.5, label="bias correction ON")
    ax1.plot(t, [b["w"] for b in off], color=common.C_ORANGE, marker="o", ms=3.5, label="OFF")
    ax1.annotate("ON", (t[-1], on[-1]["w"]), xytext=(4, 0), textcoords="offset points",
                 color=common.C_BLUE, fontweight="bold", va="center")
    ax1.annotate("OFF", (t[-1], off[-1]["w"]), xytext=(4, 0), textcoords="offset points",
                 color=common.C_ORANGE, fontweight="bold", va="center")
    ax1.set_xlabel("step"); ax1.set_ylabel("weight")
    ax1.set_title("Weight trajectory, 20 steps", loc="left", fontweight="bold")
    ax1.legend(loc="lower left")

    r = [ratio(i) for i in t]
    ax2.axhline(1.0, color=common.INK_MUTED, lw=1, ls=":")
    ax2.annotate("parity (correction stops mattering here)", (20, 1.0), xytext=(0, 5),
                 textcoords="offset points", color=common.INK_MUTED, fontsize=8, ha="right")
    ax2.plot(t, r, color=common.C_BLUE, label=r"analytic  $\sqrt{1-\beta_2^t}\,/\,(1-\beta_1^t)$")
    ax2.plot(t, [abs(a["step"] / b["step"]) for a, b in zip(on, off)],
             ls="none", marker="x", ms=6, mew=1.6, color=common.C_ORANGE,
             label="measured, from the 20 steps")
    ax2.annotate(f"worst at t={tmin}: uncorrected steps\nare {1 / ratio(tmin):.1f}x too large",
                 (tmin, ratio(tmin)), xytext=(0, 48), textcoords="offset points",
                 color=common.INK, fontsize=8, ha="center",
                 arrowprops=dict(arrowstyle="->", color=common.INK_MUTED, lw=1))
    ax2.set_ylim(0, 1.2); ax2.set_xlabel("step")
    ax2.set_ylabel("corrected step / uncorrected step")
    ax2.set_title("The correction is a pure LR multiplier", loc="left", fontweight="bold")
    ax2.legend(loc="center left", fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/part2_bias_correction.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
