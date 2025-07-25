# Soft Actor-Critic (SAC) Implementation

## 0. Project Introduction

This project implements the **Soft Actor-Critic (SAC)** algorithm for continuous control tasks using PyTorch and Gymnasium. SAC is an advanced off-policy deep reinforcement learning method based on maximum entropy reinforcement learning, balancing exploration and exploitation with entropy regularization.

### Algorithm Structure & Components

* **Replay Buffer:** Stores past experiences for training.
* **Actor Network (Policy):** Outputs action distributions.
* **Critic Network (Value):** Estimates Q-values (twin Q-networks).
* **Alpha (Entropy Regularization):** Automatically adjusted to maximize entropy.
* **SAC Agent:** Integrates the above modules for training and inference.

### Project Directory Structure

```
project_root/
│
├── main.py           # Main script, SAC implementation
├── models/           # Directory to save/load model weights
├── figures/          # All result figures and visualization images
├── README.md         # Project documentation
└── requirements.txt  # Dependencies
```

---

## 1. Algorithmic Theory

### 1.1. Mathematical Foundations

The SAC algorithm optimizes a stochastic policy \$\pi\_\theta(a|s)\$ by maximizing the expected return plus an entropy term:

$
J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t,a_t)\sim\rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]
$

where \$\mathcal{H}(\pi(\cdot|s\_t)) = -\mathbb{E}\_{a\_t\sim\pi}\[ \log \pi(a\_t|s\_t)]\$ is the policy entropy, and \$\alpha\$ is the temperature parameter (entropy coefficient).

#### Critic (Q-function) update:

$
J_Q(\theta) = \mathbb{E}_{(s_t,a_t)\sim D} \left[ \frac{1}{2} \left( Q_\theta(s_t,a_t) - \left( r_t + \gamma (1-d_t) V_{\bar{\theta}}(s_{t+1}) \right) \right)^2 \right]
$

#### Value function:

$
V_{\bar{\theta}}(s_t) = \mathbb{E}_{a_t \sim \pi_\phi} \left[ Q_{\bar{\theta}}(s_t, a_t) - \alpha \log \pi_\phi(a_t|s_t) \right]
$

#### Policy update:

$
J_\pi(\phi) = \mathbb{E}_{s_t \sim D, a_t \sim \pi_\phi} \left[ \alpha \log \pi_\phi(a_t|s_t) - Q_\theta(s_t, a_t) \right]
$

#### Temperature (Alpha) update:

$
J(\alpha) = \mathbb{E}_{a_t \sim \pi_\phi} \left[ -\alpha \log \pi_\phi(a_t|s_t) - \alpha \mathcal{H}_{target} \right]
$

### 1.2. Reward Function

The reward function \$r(s, a)\$ depends on the specific Gymnasium environment chosen. For example, in `HalfCheetah-v5`, the reward typically measures running velocity minus energy costs, incentivizing both fast and efficient movement.

### 1.3. Algorithm Principles

* **Off-policy:** Uses experience replay for efficient sample use.
* **Entropy regularization:** Encourages exploration by maximizing entropy.
* **Automatic temperature tuning:** Balances reward maximization and entropy.
* **Twin Critic Networks:** Reduces Q-value overestimation bias.

---

## 2. Installation and Dependencies

### Requirements

* Python >= 3.11
* torch
* numpy
* tqdm
* gymnasium[mujoco]

### Quick Install

```bash
pip install torch numpy tqdm gymnasium[mujoco]
```

---

## 3. Training Method

To train the SAC agent:

```bash
python main.py
```

**Key training parameters:**

* `--max-steps`: Total training steps (default: 1,000,000)
* `--start-steps`: Steps of initial random exploration (default: 10,000)
* `--batch-size`: Mini-batch size for updates (default: 256)
* `--save-interval`: How often to save models (default: 50,000 steps)

You can adjust hyperparameters in the command line or directly in the code.

---

## 4. Inference Method

To run inference (evaluation) with a trained model:

```bash
python main.py --eval --load-model-step 50000
```

* `--eval`: Run in evaluation mode.
* `--load-model-step`: Load model at a specific training step.
* `--eval-episodes`: Number of evaluation episodes (default: 1).

During evaluation, the environment will render the agent's behavior for visualization.

---

## 5. Results and Analysis

### 5.1. Learning Curves

**Training for 1,000 steps (unstable movement):**

![Placeholder: Curve 1](./figures/eval_10000.gif)

*The agent has not yet learned a stable movement policy. Rewards are low and performance is volatile.*

**Training for 50,000 steps (stable movement):**

![Placeholder: Curve 1](./figures/eval_50000.gif)

*After sufficient training, the agent achieves stable locomotion. Reward curves rise and stabilize, showing effective learning.*

### 5.2. Discussion

* **Short Training (1,000 steps):**
  The agent explores randomly, with little to no meaningful progress. Most trajectories fail to achieve significant rewards or stable movement.

* **Longer Training (50,000 steps):**
  The agent effectively learns optimal movement strategies. Rewards increase and stabilize, reflecting successful policy and value learning.

---

## Acknowledgement

This implementation is based on the original SAC paper:
*Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic Algorithms and Applications.*
[arXiv:1812.05905](https://arxiv.org/abs/1812.05905)

---
