# EVT-MixtureQuantile-ML
Research code for EVT + Mixture Quantile Modeling
## Overview

This project aims to replicate and extend the methodology from the paper:
Sarmadi, H., & Yuen, K.-V. (2022). Structural health monitoring by a novel probabilistic machine learning method based on extreme value theory and mixture quantile modeling. Mechanical Systems and Signal Processing, 173, 109049.
https://doi.org/10.1016/j.ymssp.2022.109049

The goal is to apply a principled probabilistic approach based on **Extreme Value Theory (EVT)** and **Mixture Quantile Modeling** to detect **anomalies or extreme events** across multiple domains:

* 🏗️ Structural Health Monitoring (SHM)
* 💰 Financial Risk Analysis
* 🌍 Natural Disaster Prediction
* 🏥 Health Monitoring (e.g. ICU signals, pandemics)

## Method Summary

The approach combines:

* **Unsupervised anomaly detection**
* **Extreme Value Theory (EVT)** for tail modeling
* **Mixture Quantile Estimation**: blending non-parametric and GEVD-based parametric quantiles

### Key Steps:

1. Calculate pairwise distance between feature samples
2. Sort distances and reverse sign to focus on tail
3. Fit GEVD to largest negative distances via KL-based selection
4. Compute **mixture quantile** as a **novelty score**
5. Use EVT to set threshold and trigger anomaly alarms

## Dataset

This repo currently uses synthetic or public data (e.g., financial returns, simulated sensor data) for demonstration. You can adapt to:

* Z24 Bridge / Yonghe Bridge (original paper)
* Any spatio-temporal signal data
* Stock market returns
* Health sensor outputs

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py  # Run full training and evaluation pipeline
```

## File Structure

```
📁 root
├── main.py                # Entry point
├── utils/                 # Helper functions
├── data/                  # Sample or real datasets
├── models/                # EVT, quantile models, novelty score
├── results/               # Output logs, plots
├── requirements.txt
└── README.md
```

## License

MIT License. You are free to use, modify, and distribute.

## Citation

If you use this codebase, please cite:

```
@article{sarmadi2022structural,
  title={Structural health monitoring by a novel probabilistic machine learning method based on extreme value theory and mixture quantile modeling},
  author={Sarmadi, Hassan and Yuen, Ka-Veng},
  journal={Mechanical Systems and Signal Processing},
  volume={173},
  pages={109049},
  year={2022},
  publisher={Elsevier}
}
```

---

Feel free to reach out or contribute by pull requests!
