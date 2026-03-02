"""
Flow Matching GMM Demo — enhanced interactive educational app.

Improvements over v1:
  - Trajectory snapshots at multiple time slices (t=0, 0.25, 0.5, 0.75, 1)
  - Velocity field (quiver) visualization on a 2D grid
  - Smoothed training loss curve
  - Robust metrics: SW mean±std, moment mismatch
  - st.spinner progress feedback during training
  - Preset configs for common experiments
  - JSON config export for reproducibility
  - Gradient clipping + cosine LR schedule options
  - std_min / std_max guard
  - Light code-structure cleanup with clear section functions
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCfg:
    k: int
    n: int
    noise_std: float
    seed: int
    layout: str = "Ring (structured)"
    r: float = 2.3
    comp_std: float = 0.25
    skew: float = 0.8
    mean_span: float = 3.0
    std_min: float = 0.10
    std_max: float = 0.50
    weight_alpha: float = 0.70


@dataclass
class TrainCfg:
    source: str
    hidden: int
    layers: int
    lr: float
    steps: int
    batch: int
    integ: int
    grad_clip: float = 0.0   # 0 means disabled
    cosine_lr: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def make_gmm(c: DataCfg):
    """Create clean / noisy samples from a 2-D GMM (ring or randomised)."""
    rng = np.random.default_rng(c.seed)
    if c.layout == "Randomized components":
        m = rng.uniform(-c.mean_span, c.mean_span, size=(c.k, 2))
        s_lo, s_hi = sorted((c.std_min, c.std_max))
        comp_stds = rng.uniform(s_lo, max(s_hi, s_lo + 1e-6), size=c.k)
        w = rng.dirichlet(np.full(c.k, max(c.weight_alpha, 1e-3)))
    else:
        a = np.linspace(0, 2 * np.pi, c.k, endpoint=False) + rng.uniform(0, 2 * np.pi)
        m = np.c_[c.r * np.cos(a), c.r * np.sin(a)] + rng.normal(0, 0.15, (c.k, 2))
        logits = rng.normal(size=c.k) * c.skew
        w = np.exp(logits - logits.max()); w /= w.sum()
        comp_stds = np.full(c.k, c.comp_std)

    comp = rng.choice(c.k, size=c.n, p=w)
    clean = m[comp] + rng.normal(0, comp_stds[comp, None], (c.n, 2))
    noisy = clean + rng.normal(0, c.noise_std, clean.shape)
    return (
        clean.astype(np.float32),
        noisy.astype(np.float32),
        m.astype(np.float32),
        w.astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class VelNet(nn.Module):
    """Small MLP that predicts 2-D velocity given (x, t)."""

    def __init__(self, h: int, l: int):
        super().__init__()
        seq = [nn.Linear(3, h), nn.SiLU()]
        for _ in range(l - 1):
            seq += [nn.Linear(h, h), nn.SiLU()]
        seq.append(nn.Linear(h, 2))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Training & inference
# ─────────────────────────────────────────────────────────────────────────────

def train_flow(clean_np, noisy_np, c: TrainCfg, seed: int, progress_bar=None):
    """
    Train velocity field with conditional flow matching.

    Linear interpolation path: x_t = (1-t)*x0 + t*x1
    Target velocity: v = x1 - x0 (constant along each straight trajectory).

    Returns
    -------
    losses : list[float]
        Per-step MSE loss.
    snapshots : dict[float, np.ndarray]
        Particle positions at t = 0, 0.25, 0.5, 0.75, 1.0.
    """
    torch.manual_seed(seed)
    clean = torch.tensor(clean_np)
    noisy = torch.tensor(noisy_np)
    net = VelNet(c.hidden, c.layers)
    opt = torch.optim.Adam(net.parameters(), lr=c.lr)

    # Cosine LR schedule
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c.steps)
        if c.cosine_lr
        else None
    )

    losses: list[float] = []
    for step in range(c.steps):
        idx = torch.randint(0, len(clean), (c.batch,))
        x1 = clean[idx]
        x0 = noisy[idx] if c.source == "Noisy observation -> Clean target" else torch.randn_like(x1)
        t = torch.rand(c.batch, 1)
        xt = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        loss = ((net(xt, t) - v_target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if c.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), c.grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach()))

        if progress_bar is not None and step % max(1, c.steps // 100) == 0:
            progress_bar.progress((step + 1) / c.steps)

    # ── Euler integration with trajectory snapshots ──
    snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    with torch.no_grad():
        if c.source == "Noisy observation -> Clean target":
            x = torch.tensor(noisy_np)
        else:
            x = torch.randn(len(clean_np), 2)

        snapshots: dict[float, np.ndarray] = {0.0: x.numpy().copy()}
        dt = 1.0 / c.integ
        for k in range(c.integ):
            t_now = k / c.integ
            t_vec = torch.full((len(x), 1), t_now)
            x = x + net(x, t_vec) * dt
            t_frac = round((k + 1) / c.integ, 6)
            for snap_t in snap_times[1:]:
                if abs(t_frac - snap_t) < dt / 2:
                    snapshots[snap_t] = x.numpy().copy()

        # Guarantee t=1 snap
        if 1.0 not in snapshots:
            snapshots[1.0] = x.numpy().copy()

    return losses, snapshots, net


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def sw2d(x, y, nproj: int = 64, seed: int = 0) -> float:
    """Sliced Wasserstein distance (single run)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(nproj, 2))
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return float(np.mean([np.mean(np.abs(np.sort(x @ u) - np.sort(y @ u))) for u in d]))


def sw2d_stats(x, y, nproj: int = 64, n_trials: int = 8, seed: int = 0):
    """Return mean ± std of SW over several random projection seeds."""
    vals = [sw2d(x, y, nproj, seed + i) for i in range(n_trials)]
    return float(np.mean(vals)), float(np.std(vals))


def moment_mismatch(gen, clean):
    """
    Compare first two moments of generated vs clean samples.

    Returns dict with mean error (L2) and max abs covariance difference.
    """
    mean_err = float(np.linalg.norm(gen.mean(0) - clean.mean(0)))
    cov_diff = float(np.abs(np.cov(gen.T) - np.cov(clean.T)).max())
    return {"mean_err": mean_err, "cov_diff": cov_diff}


def smooth(losses, w: int = 20):
    """Simple moving-average smoothing of a 1-D array."""
    if len(losses) < w:
        return losses
    kernel = np.ones(w) / w
    return np.convolve(losses, kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "clean": "#4CAF50",
    "noisy": "#FF6B6B",
    "gen":   "#6A5ACD",
    "src":   "#5B9BD5",
}


def fig_data(clean, noisy, means):
    """Side-by-side: clean target vs noisy observation."""
    f = make_subplots(rows=1, cols=2,
                      subplot_titles=("Clean target (x₁)", "Noisy observation (x₀)"))
    f.add_trace(go.Scattergl(x=clean[:, 0], y=clean[:, 1], mode="markers",
                             name="Clean", marker=dict(size=4, opacity=0.5,
                                                       color=_COLORS["clean"])), row=1, col=1)
    f.add_trace(go.Scatter(x=means[:, 0], y=means[:, 1], mode="markers",
                           name="Component mean",
                           marker=dict(size=10, symbol="x", color="black",
                                       line=dict(width=2, color="black"))), row=1, col=1)
    f.add_trace(go.Scattergl(x=noisy[:, 0], y=noisy[:, 1], mode="markers",
                             name="Noisy", marker=dict(size=4, opacity=0.5,
                                                       color=_COLORS["noisy"])), row=1, col=2)
    f.update_layout(height=370, showlegend=True, legend=dict(orientation="h", y=-0.12))
    f.update_xaxes(scaleanchor="y", scaleratio=1)
    return f


def fig_trajectories(snapshots: dict[float, np.ndarray]):
    """Show particle positions at 5 time slices in a 1×5 grid."""
    times = sorted(snapshots)
    f = make_subplots(rows=1, cols=len(times),
                      subplot_titles=[f"t = {t:.2f}" for t in times])
    for col, t in enumerate(times, 1):
        pts = snapshots[t]
        alpha = 0.15 + 0.7 * t
        f.add_trace(go.Scattergl(
            x=pts[:, 0], y=pts[:, 1], mode="markers",
            marker=dict(size=3, opacity=alpha, color=_COLORS["gen"]),
            showlegend=False,
        ), row=1, col=col)
    f.update_layout(height=290, margin=dict(t=50))
    f.update_xaxes(scaleanchor="y", scaleratio=1)
    return f


def fig_velocity_field(net: VelNet, xlim: tuple, ylim: tuple, t_val: float = 0.5,
                       grid_n: int = 20):
    """Visualise the learned velocity field on a 2-D grid at a given t."""
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        x_t = torch.tensor(pts)
        t_t = torch.full((len(pts), 1), t_val)
        vxy = net(x_t, t_t).numpy()
    mag = np.linalg.norm(vxy, axis=1, keepdims=True) + 1e-6
    vnorm = vxy / mag          # normalise for display; colour encodes magnitude
    scale = (xlim[1] - xlim[0]) / grid_n * 0.8

    fig = go.Figure()
    for i in range(len(pts)):
        x0, y0 = pts[i]
        dx, dy = vnorm[i] * scale
        fig.add_annotation(
            x=x0 + dx, y=y0 + dy,
            ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor=f"rgba(90,90,200,{min(1.0, float(mag[i, 0]) / (mag.mean() * 2)):.2f})",
        )
    fig.update_layout(
        height=420, title=f"Velocity field at t = {t_val:.2f}",
        xaxis=dict(range=[xlim[0], xlim[1]], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[ylim[0], ylim[1]]),
    )
    return fig


def fig_res(start, gen, clean):
    """Source, generated, overlay — final result view."""
    f = make_subplots(rows=1, cols=3,
                      subplot_titles=("Source  t=0", "Generated  t=1", "Overlay"))
    f.add_trace(go.Scattergl(x=start[:, 0], y=start[:, 1], mode="markers",
                             name="Source",
                             marker=dict(size=4, opacity=0.45, color=_COLORS["src"])), row=1, col=1)
    f.add_trace(go.Scattergl(x=gen[:, 0], y=gen[:, 1], mode="markers",
                             name="Generated",
                             marker=dict(size=4, opacity=0.45, color=_COLORS["gen"])), row=1, col=2)
    f.add_trace(go.Scattergl(x=clean[:, 0], y=clean[:, 1], mode="markers",
                             name="Target",
                             marker=dict(size=3, opacity=0.20, color=_COLORS["clean"])), row=1, col=3)
    f.add_trace(go.Scattergl(x=gen[:, 0], y=gen[:, 1], mode="markers",
                             name="Generated (overlay)",
                             marker=dict(size=4, opacity=0.45, color=_COLORS["gen"])), row=1, col=3)
    f.update_layout(height=390, showlegend=True, legend=dict(orientation="h", y=-0.12))
    f.update_xaxes(scaleanchor="y", scaleratio=1)
    return f


def fig_loss(losses):
    """Raw + smoothed training loss curve."""
    raw_y = np.array(losses, dtype=float)
    sm_y = smooth(losses, w=max(2, len(losses) // 40))
    f = go.Figure()
    f.add_trace(go.Scatter(y=raw_y, mode="lines", name="Raw",
                           line=dict(color="lightgrey", width=1)))
    f.add_trace(go.Scatter(
        x=np.arange(len(raw_y) - len(sm_y), len(raw_y)),
        y=sm_y, mode="lines", name="Smoothed",
        line=dict(color="#E74C3C", width=2),
    ))
    f.update_layout(height=240, xaxis_title="Training step",
                    yaxis_title="Velocity MSE",
                    legend=dict(orientation="h", y=1.1))
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "Easy denoise": dict(
        k=4, n=2000, noise_std=0.3, layout="Ring (structured)",
        r=2.3, comp_std=0.25, skew=0.8,
        src="Noisy observation -> Clean target",
        hidden=64, layers=2, lr=1e-3, steps=800, batch=256, integ=50,
    ),
    "Hard denoise": dict(
        k=6, n=3000, noise_std=1.2, layout="Ring (structured)",
        r=2.3, comp_std=0.30, skew=1.5,
        src="Noisy observation -> Clean target",
        hidden=128, layers=3, lr=5e-4, steps=2000, batch=512, integ=100,
    ),
    "Generation (Gaussian source)": dict(
        k=5, n=2000, noise_std=0.5, layout="Randomized components",
        r=2.3, comp_std=0.25, skew=0.8,
        src="Standard Gaussian -> Clean target",
        hidden=96, layers=2, lr=1e-3, steps=1500, batch=256, integ=80,
    ),
    "Extreme imbalance": dict(
        k=4, n=2000, noise_std=0.6, layout="Ring (structured)",
        r=2.5, comp_std=0.20, skew=2.5,
        src="Noisy observation -> Clean target",
        hidden=96, layers=2, lr=1e-3, steps=1200, batch=256, integ=80,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar():
    """Build sidebar controls and return (DataCfg, TrainCfg)."""
    st.header("⚙️ Configuration")

    # Presets
    preset_name = st.selectbox("Quick preset (optional)", ["— custom —"] + list(PRESETS))
    preset = PRESETS.get(preset_name, {})

    def pv(key, default):
        return preset.get(key, default)

    st.divider()
    st.subheader("Data")
    seed = st.number_input("Seed", 0, 999_999, 42, 1)
    k    = st.slider("# Components", 2, 10, pv("k", 4))
    n    = st.slider("# Samples", 300, 6000, pv("n", 2000), 100)
    layout = st.radio("GMM layout", ["Ring (structured)", "Randomized components"],
                      index=0 if pv("layout", "Ring (structured)") == "Ring (structured)" else 1)

    if layout == "Randomized components":
        mean_span    = st.slider("Mean sampling span", 0.5, 6.0, 3.0)
        std_min      = st.slider("Component std (min)", 0.02, 1.0, 0.10)
        std_max_raw  = st.slider("Component std (max)", 0.02, 1.5, 0.50)
        std_max      = max(std_max_raw, std_min + 1e-3)   # guard
        weight_alpha = st.slider("Weight concentration (Dirichlet α)", 0.05, 5.0, 0.70)
        r = comp_std = skew = None
    else:
        r         = st.slider("Component separation", 0.5, 5.0, pv("r", 2.3))
        comp_std  = st.slider("Target std", 0.05, 1.0, pv("comp_std", 0.25))
        skew      = st.slider("Mixture imbalance", 0.0, 2.5, pv("skew", 0.8))
        mean_span = std_min = std_max = weight_alpha = None

    noise_std = st.slider("Observation noise std", 0.0, 2.0, pv("noise_std", 0.7))

    st.divider()
    st.subheader("Model & Training")
    src    = st.radio("Source", ["Noisy observation -> Clean target",
                                  "Standard Gaussian -> Clean target"],
                      index=0 if pv("src", "Noisy observation -> Clean target")
                                 == "Noisy observation -> Clean target" else 1)
    hidden = st.select_slider("Hidden width", [32, 64, 96, 128, 192], pv("hidden", 96))
    layers = st.slider("Layers", 1, 5, pv("layers", 2))
    lr     = st.select_slider("Learning rate", [1e-4, 2e-4, 5e-4, 1e-3, 2e-3], pv("lr", 1e-3))
    steps  = st.slider("Train steps", 100, 4000, pv("steps", 1200), 100)
    batch  = st.select_slider("Batch size", [64, 128, 256, 512, 1024], pv("batch", 256))
    integ  = st.slider("Euler integration steps", 10, 200, pv("integ", 80))

    st.divider()
    st.subheader("Robustness options")
    grad_clip = st.slider("Gradient clip (0 = off)", 0.0, 5.0, 0.0, 0.5)
    cosine_lr = st.toggle("Cosine LR decay", value=False)

    run = st.button("▶  Train / Retrain", type="primary", width="stretch")

    data_cfg = DataCfg(
        k=k, n=n, noise_std=noise_std, seed=int(seed),
        layout=layout,
        r=r or 2.3,
        comp_std=comp_std or 0.25,
        skew=skew or 0.8,
        mean_span=mean_span or 3.0,
        std_min=std_min or 0.10,
        std_max=std_max or 0.50,
        weight_alpha=weight_alpha or 0.70,
    )
    train_cfg = TrainCfg(
        source=src, hidden=hidden, layers=layers, lr=lr, steps=steps,
        batch=batch, integ=integ, grad_clip=grad_clip, cosine_lr=cosine_lr,
    )
    return data_cfg, train_cfg, run


def _config_export(data_cfg: DataCfg, train_cfg: TrainCfg):
    """Show JSON config snippet for reproducibility."""
    with st.expander("📋 Export config (for reproducibility)", expanded=False):
        cfg_dict = {"data": asdict(data_cfg), "train": asdict(train_cfg)}
        st.code(json.dumps(cfg_dict, indent=2), language="json")


def main():
    st.set_page_config(page_title="Flow Matching GMM Demo", layout="wide",
                       initial_sidebar_state="expanded")
    st.title("🌊 Flow Matching Demo: Noisy Gaussian Mixture")
    st.caption(
        "Train a small velocity-field MLP to transport samples from a source distribution "
        "to a 2-D Gaussian Mixture target using straight-path conditional flow matching."
    )

    with st.sidebar:
        data_cfg, train_cfg, run = _sidebar()

    # ── Data visualisation ──────────────────────────────────────────────────
    clean, noisy, means, w = make_gmm(data_cfg)

    st.subheader("📊 Dataset")
    st.plotly_chart(fig_data(clean, noisy, means), width="stretch")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Observation noise σ", f"{data_cfg.noise_std:.2f}")
    col_b.metric("Largest component weight", f"{w.max():.2f}")
    col_c.metric(
        "Target std",
        f"{data_cfg.comp_std:.2f}" if data_cfg.layout == "Ring (structured)"
        else f"{data_cfg.std_min:.2f} – {data_cfg.std_max:.2f}",
    )

    _config_export(data_cfg, train_cfg)

    # ── Training ────────────────────────────────────────────────────────────
    if run:
        with st.spinner("Training velocity field…"):
            progress_placeholder = st.progress(0)
            t0 = time.perf_counter()
            losses, snapshots, net = train_flow(
                clean, noisy, train_cfg, data_cfg.seed, progress_placeholder
            )
            elapsed = time.perf_counter() - t0
        progress_placeholder.empty()
        st.success(f"Training finished in {elapsed:.1f} s")

        gen = snapshots[1.0]
        sw_mean, sw_std = sw2d_stats(gen, clean, n_trials=8, seed=data_cfg.seed + 7)
        mm = moment_mismatch(gen, clean)
        st.session_state["res"] = (losses, snapshots, net, clean, sw_mean, sw_std, mm,
                                   data_cfg, train_cfg)

    # ── Results ─────────────────────────────────────────────────────────────
    if "res" in st.session_state:
        (losses, snapshots, net, clean_ref,
         sw_mean, sw_std, mm, dcfg, tcfg) = st.session_state["res"]

        gen = snapshots[1.0]
        start = snapshots[0.0]

        st.divider()
        st.subheader("📈 Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Sliced Wasserstein (↓)", f"{sw_mean:.4f}", f"±{sw_std:.4f}")
        mc2.metric("Mean error (L2, ↓)", f"{mm['mean_err']:.4f}")
        mc3.metric("Max covariance diff (↓)", f"{mm['cov_diff']:.4f}")
        mc4.metric("Final loss", f"{losses[-1]:.5f}")

        st.divider()
        st.subheader("🔄 Trajectory snapshots (particle positions over time)")
        st.plotly_chart(fig_trajectories(snapshots), width="stretch")

        st.divider()
        st.subheader("🏁 Final result")
        st.plotly_chart(fig_res(start, gen, clean_ref), width="stretch")

        st.divider()
        st.subheader("🧭 Learned velocity field")
        # Compute axis limits from data range
        all_pts = np.vstack([clean_ref, noisy])
        pad = 0.5
        xlim = (float(all_pts[:, 0].min()) - pad, float(all_pts[:, 0].max()) + pad)
        ylim = (float(all_pts[:, 1].min()) - pad, float(all_pts[:, 1].max()) + pad)
        t_vis = st.slider("Show velocity field at t =", 0.0, 1.0, 0.5, 0.05)
        st.plotly_chart(fig_velocity_field(net, xlim, ylim, t_vis), width="stretch")

        st.divider()
        st.subheader("📉 Training loss")
        st.plotly_chart(fig_loss(losses), width="stretch")

    # ── Tips ────────────────────────────────────────────────────────────────
    with st.expander("💡 Exploration tips", expanded=False):
        st.markdown(
            """
| Experiment | What to change |
|---|---|
| Harder denoising | ↑ Observation noise std |
| Underfitting | ↓ Width / Layers |
| Slow convergence | ↓ Train steps |
| Pure generation | Source = Gaussian |
| Mode collapse risk | ↑ Imbalance, ↑ k |
| Trajectory instability | ↓ Integration steps |
| Slow but stable training | Enable cosine LR |
| Gradient explosion | Enable grad clip |
            """
        )


if __name__ == "__main__":
    main()
