---
title: RL-Recommendation-System
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<h1>🎯 RL Ad Recommendation System</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=7c3aed&center=true&vCenter=true&width=700&lines=Deep+Q-Learning+for+Ad+Recommendation;GRU+State+Encoder+·+LinUCB+Baseline;CTR+·+CVR+·+eCPM+·+NDCG@K" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/RL-Recommendation-System)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/RL-Recommendation-System)

<br/>

**🎯 RL Ad Recommendation System** — A reinforcement-learning ad recommender that learns which ads to show each user. A **DQN agent** trained on synthetic real-time-bidding impression logs balances clicks, conversions, ad fatigue and revenue, with a **LinUCB contextual bandit** as an interpretable baseline.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🧠 <b>DQN Agent</b></td>
    <td>Q-network with a target network and Huber loss picks the next ad from a 52-dim user state; rewards combine clicks, conversions, bid revenue and an ad-fatigue penalty</td>
  </tr>
  <tr>
    <td>🔁 <b>GRU State Encoder</b></td>
    <td>Encodes each user's recent ad-interaction sequence with a GRU, plus user features and sinusoidal hour / day-of-week context</td>
  </tr>
  <tr>
    <td>🎰 <b>LinUCB Bandit Baseline</b></td>
    <td>A simple, interpretable contextual bandit on the same state vector for comparison</td>
  </tr>
  <tr>
    <td>📈 <b>CTR / CVR Model</b></td>
    <td>Annotates every recommended ad with an estimated click and conversion probability</td>
  </tr>
  <tr>
    <td>🏭 <b>Synthetic RTB Data</b></td>
    <td>Users, advertisers, ad inventory and impression logs generated from a latent-factor interest model</td>
  </tr>
  <tr>
    <td>⚡ <b>Background Training</b></td>
    <td>Offline training runs in a background thread at startup; retrain on demand from the Training tab</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────┐   history   ┌────────────────┐  52-dim state  ┌──────────────┐
│  Synthetic   │────────────▶│ AdStateEncoder │───────────────▶│  DQN Agent   │
│  RTB data    │  user feats │  GRU + Linear  │                │ Q(s, ad)     │
└──────────────┘             └────────────────┘                └──────┬───────┘
                                                                      │ top-K ads
                     ┌────────────────┐                        ┌──────▼───────┐
                     │   CTR / CVR    │◀───────────────────────│  Flask API   │
                     │   estimates    │                        │  + Web UI    │
                     └────────────────┘                        └──────────────┘
```

**RL formulation**

| | |
|---|---|
| **State** | GRU(ad history, 32) ⊕ user features (16) ⊕ time context (4) |
| **Action** | Which ad to show (ad_id) |
| **Reward** | click + conversion + bid revenue − fatigue penalty |
| **Exploration** | ε-greedy with decay |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/RL-Recommendation-System.git
cd RL-Recommendation-System

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
docker build -t rl-recommendation-system .
docker run -p 7860:7860 rl-recommendation-system
```

---

## 📊 Dashboard Modules

| Tab | Description |
|-----|-------------|
| 🧭 Overview | What the system does, live dataset KPIs, architecture and the RL formulation |
| 🎬 Live Demo | Pick a user, see their profile and impression history, and get a ranked ad feed |
| 📈 Analytics | Category, format and time-of-day performance across the impression log |
| 🏢 Advertisers | Per-advertiser spend, CTR, CVR and revenue |
| 🏋️ Training | Start training and watch loss, ε and ranking metrics update |

---

## 📡 API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | Training status (epoch, loss, ε, progress) |
| `GET` | `/api/users` | Sample of user profiles |
| `GET` | `/api/recommend/<user_id>` | Top-K recommended ads with CTR / CVR estimates |
| `GET` | `/api/analytics` | Aggregate impression analytics |
| `GET` | `/api/advertisers` | Advertiser-level KPIs |
| `POST` | `/api/train` | Start a training run |
| `GET` | `/api/training_history` | Loss and metric history |
| `GET` | `/api/metrics` | Precision@K, Recall@K, NDCG@K, Hit-Rate@K, CTR, CVR, eCPM, coverage |

---

## 📁 Project Structure

```
RL-Recommendation-System/
│
├── 📂 src/
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   └── training/
├── 📂 static/                    # CSS, JavaScript & assets
│   ├── css/
│   └── js/
├── 📂 templates/                 # HTML templates
│   └── index.html
├── 📄 Dockerfile                 # Container definition (port 7860)
├── 📄 app.py                     # Application entry point
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available. This software is provided "as is" without warranty of any kind; use at your own risk.

---

## 📜 License

Distributed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7c3aed,100:06b6d4&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/RL-Recommendation-System?style=social)](https://github.com/mnoorchenar/RL-Recommendation-System/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/RL-Recommendation-System?style=social)](https://github.com/mnoorchenar/RL-Recommendation-System/fork)

<sub>The name "RL Ad Recommendation System" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>
