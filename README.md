# Flow Matching Demo on Noisy Gaussian Mixture

Interactive educational demo of flow matching where the target is a synthetic 2D Gaussian mixture and the input can be noisy observations.

## Run
cd /Users/yinglongxia/Code/flow_matching_gmm_demo
python3 -m pip install -r requirements.txt
streamlit run app.py

## Explore key parameters
- Observation noise std (harder denoising as it increases)
- Number of components / separation / target std
- Model width, layers, training steps, learning rate
- Source mode: noisy->clean or Gaussian->clean
- Euler integration steps
