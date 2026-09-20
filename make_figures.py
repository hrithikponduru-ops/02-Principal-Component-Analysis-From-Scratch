"""Conceptual diagrams for README.md (data-independent). Writes to figures/.

Run from this folder: python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, BLUE, GREEN, RED, GREY = "#1f2937", "#2563eb", "#059669", "#dc2626", "#9ca3af"

def fig_variance_maximisation():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(300, 2)) @ np.array([[2.2, 1.4], [0.0, 0.6]])
    C = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(C)
    vals, vecs = vals[::-1], vecs[:, ::-1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=GREY, alpha=0.7)
    for i, color in enumerate((BLUE, GREEN)):
        v = vecs[:, i] * 2.2 * np.sqrt(vals[i])
        ax.annotate("", xy=v, xytext=-v, arrowprops=dict(arrowstyle="<->", color=color, lw=2.5))
        ax.text(*(v * 1.15), f"$v_{i + 1}$", color=color, fontsize=13, ha="center", va="center")
    ax.set_title("Centered data with its two principal axes")

    ax = axes[1]
    thetas = np.linspace(0, np.pi, 181)
    variances = [np.var(pts @ np.array([np.cos(t), np.sin(t)]), ddof=1) for t in thetas]
    ax.plot(np.degrees(thetas), variances, color=INK, lw=2)
    t1 = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])) % 180
    ax.axvline(t1, color=BLUE, ls="--", lw=1.2)
    ax.axvline((t1 + 90) % 180, color=GREEN, ls="--", lw=1.2)
    ax.text(t1, max(variances), f" max = $\\lambda_1$ = {vals[0]:.2f}", color=BLUE, va="top", fontsize=9)
    ax.text((t1 + 90) % 180, min(variances), f" min = $\\lambda_2$ = {vals[1]:.2f}",
            color=GREEN, va="bottom", fontsize=9)
    ax.set_xlabel("direction angle $\\theta$ (degrees)")
    ax.set_ylabel("variance of data projected onto $[\\cos\\theta, \\sin\\theta]$")
    ax.set_title("Variance depends on direction")

    ax = axes[2]
    proj = (pts @ vecs[:, 0])[:, None] * vecs[:, 0]
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=GREY, alpha=0.5)
    for p, q in list(zip(pts, proj))[:60]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=RED, lw=0.6, alpha=0.6)
    ax.scatter(proj[:, 0], proj[:, 1], s=8, color=BLUE)
    ax.set_title("Projecting onto $v_1$: red = what is lost")

    for ax in (axes[0], axes[2]):
        ax.set_aspect("equal")
        ax.set_xlim(-7, 7)
        ax.set_ylim(-5, 5)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/variance_maximisation.png", dpi=150)
    plt.close(fig)


def fig_power_iteration_convergence():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    for ratio, color in ((0.5, BLUE), (0.8, GREEN), (0.95, RED)):
        A = np.diag([1.0, ratio, ratio * 0.5, 0.1])
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        errs = []
        for _ in range(60):
            v = A @ v
            v /= np.linalg.norm(v)
            errs.append(1 - abs(v[0]))
        ax.semilogy(errs, color=color, lw=2, label=f"$\\lambda_2/\\lambda_1$ = {ratio}")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("error $1 - |v_t \\cdot v_1|$")
    ax.set_title("Power iteration converges geometrically")
    ax.legend(fontsize=9)

    ax = axes[1]
    A = np.array([[2.0, 0.8], [0.8, 1.0]])
    vals, vecs = np.linalg.eigh(A)
    v1 = vecs[:, 1]
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle), np.sin(circle), color=GREY, lw=0.8)
    v = np.array([-0.3, 0.95])
    v /= np.linalg.norm(v)
    for t in range(6):
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=plt.cm.Blues(0.35 + 0.13 * t), lw=2))
        if t in (0, 1, 2, 5):
            ax.text(*(v * 1.14), f"$v_{t}$", fontsize=9, ha="center", va="center")
        v = A @ v
        v /= np.linalg.norm(v)
    v1 = v1 if v1 @ v > 0 else -v1  # pick the sign that matches the iterates
    ax.annotate("", xy=v1 * 1.3, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5, ls="--"))
    ax.text(v1[0] * 1.3 + 0.05, v1[1] * 1.3 - 0.28, "true\\neigenvector", color=RED, fontsize=9, ha="left")
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title("Each multiply by A rotates v toward the top eigenvector")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/power_iteration.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    fig_variance_maximisation()
    fig_power_iteration_convergence()
    print("wrote figures/")