# Flow Matching Demo on Noisy Gaussian Mixture

Interactive educational demo of **conditional flow matching** where the target is a synthetic 2-D Gaussian mixture and the source can be either noisy observations or a standard Gaussian.

A small MLP is trained to predict the velocity of straight-line "flow" trajectories connecting source → target, then integrated with Euler's method to generate samples.

## Features

| Feature | Description |
|---|---|
| **Trajectory snapshots** | See particle positions at t = 0, 0.25, 0.5, 0.75, 1 after integration |
| **Velocity field viewer** | Interactive quiver plot of the learned vector field at any chosen t |
| **Robust metrics** | Sliced Wasserstein mean ± std (8 trials), mean error (L2), max covariance diff |
| **Smoothed loss curve** | Raw + moving-average training loss plotted together |
| **Quick presets** | Easy denoise / Hard denoise / Generation (Gaussian) / Extreme imbalance |
| **Config export** | JSON snapshot of all hyperparams + seed for reproducibility |
| **Robustness options** | Optional gradient clipping and cosine LR schedule |
| **GMM layouts** | Ring (structured) or Randomized components (random means, stds, weights) |
| **Training progress** | Live progress bar + elapsed-time display during training |

## Run

```bash
cd /path/to/flow_matching_gmm_demo
pip install -r requirements.txt
streamlit run app.py
```

## Key parameters

### Data
| Parameter | Effect |
|---|---|
| Observation noise std | Higher → harder denoising task |
| # Components / separation | Shape of the target GMM |
| Target std | Spread within each component |
| Mixture imbalance (skew) | Unequal component weights |
| GMM layout | Ring vs random |

### Model & training
| Parameter | Effect |
|---|---|
| Source mode | Noisy→Clean (denoising) or Gaussian→Clean (generation) |
| Hidden width / Layers | Model capacity |
| Learning rate | Optimiser step size |
| Train steps | More → better convergence |
| Batch size | Larger → lower variance gradient |
| Euler integration steps | More → smoother trajectories |
| Gradient clip | Prevents exploding gradients |
| Cosine LR | Slow decay for stable final convergence |

## Exploration experiments

| Goal | Setting |
|---|---|
| Harder denoising | ↑ Observation noise std |
| Underfitting | ↓ Width / Layers |
| Slow convergence | ↓ Train steps |
| Pure generation | Source = Standard Gaussian |
| Mode collapse risk | ↑ Imbalance, ↑ k |
| Trajectory instability | ↓ Integration steps |
| Stable final convergence | Enable cosine LR |
| Gradient explosion test | Enable grad clip |

## Architecture

```
Input: (x ∈ ℝ², t ∈ [0,1])  →  [3]
MLP: Linear → SiLU (×layers)  →  [hidden]
Output: v̂(x,t) ∈ ℝ²

Loss: E[ ‖v̂(xₜ, t) − (x₁ − x₀)‖² ]
      where xₜ = (1−t)·x₀ + t·x₁  (linear interpolant)
```
