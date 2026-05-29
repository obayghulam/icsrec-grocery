# ICSRec on Amazon Grocery

> Reproducing and extending *Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation* (WSDM 2024) on the Amazon Reviews 2023 Grocery and Gourmet Food dataset.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Course](https://img.shields.io/badge/course-CCAI%20422-blue)

This repository contains the full supplementary materials for our final project in **CCAI 422 — Recommender Systems** at the University of Jeddah, Spring 2025/2026.

We reproduce ICSRec on a domain it was not originally tested on (Amazon grocery), tune the number of latent intents, and present two contributions that go beyond the original paper.

---

## Highlights

- **Best K-means configuration:** $K = 512$ intents reaches **HIT@20 = 0.0842** and **NDCG@20 = 0.0357** — a ~3% relative gain over the paper's default $K = 256$ on a dataset with 99.7% sparsity.
- **Contribution 1 (GMM soft clustering):** ties K-means on NDCG@20 (both 0.0352), loses 5.3% on HIT@5. An honest negative result — hard K-means is well-calibrated for grocery.
- **Contribution 2 (cold-start analysis):** warm users (≥20 interactions) beat cold users by **+12.8% NDCG@5** and **+11.7% HIT@5** — a real limitation of ICSRec that the original paper does not report.

---

## Authors

- Obay R. Ghulam  2240497@uj.edu.sa 
- Eyad A. Alghamdi  2240059@uj.edu.sa 
- Abdulaziz F. Etaiwi  2140040@uj.edu.sa


**University of Jeddah
College of Computer Science and Engineering
Department of Computer Science and Artificial Intelligence** 

---

## Repository layout

```
ICSRec_project/
├── README.md                              ← you are here
├── LICENSE                                ← MIT
├── .gitignore
├── notebooks/                             ← run in numerical order
│   ├── 1_preprocess_grocery_for_icsrec.ipynb
│   ├── 2_train_icsrec_grocery.ipynb           (baseline + hyperparameter sweep)
│   ├── 3_contribution_gmm.ipynb               (Contribution 1)
│   └── 4_coldstart_analysis.ipynb             (Contribution 2)
├── code_patches/                          ← our modifications to the official repo
│   ├── README.md                              (how to apply)
│   ├── models_faiss_cpu.patch
│   ├── models_gmm_addition.py
│   ├── trainers_cluster_switch.patch
│   └── main_cluster_method_arg.patch
├── data/
│   ├── README.md                              (dataset source + link)
│   └── Grocery_and_Gourmet_Food.txt           (preprocessed, 1.68 MB)
└── results/
    ├── training_logs/                         ← raw ICSRec output, 5 files
    │   ├── baseline_K256.txt
    │   ├── tuning_K64.txt
    │   ├── tuning_K128.txt
    │   ├── tuning_K512.txt
    │   └── contribution1_gmm_K128.txt
    └── summaries/                             ← clean summary tables
        ├── tuning_comparison.txt
        ├── coldstart_analysis.txt
        └── contribution1_gmm_comparison.txt
```

---

## How to reproduce

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is strongly recommended. We used an **NVIDIA A100** via Google Colab Pro. Each training run with `intent_num=256` takes about 30 minutes.

### Required packages

```
torch >= 2.0
faiss-cpu
scikit-learn
gensim
pandas, numpy, matplotlib
datasets, huggingface_hub
```

The notebooks install everything they need at the top of each one, so you can just open them in Colab and click Run.

### Steps

1. **Clone the official ICSRec repository.** We do **not** vendor it here — only our patches.
   ```bash
   git clone https://github.com/QinHsiu/ICSRec.git
   ```
2. **Apply our patches** to the cloned repository — see [`code_patches/README.md`](code_patches/README.md). The notebooks apply them automatically when run; you only need to do this manually if you want a clean inspectable copy.
3. **Copy the preprocessed data** from [`data/Grocery_and_Gourmet_Food.txt`](data/) into `ICSRec/data/`. This skips the preprocessing step entirely. To regenerate from the raw Amazon CSV, run notebook 1.
4. **Run the notebooks in order:**
   - `2_train_icsrec_grocery.ipynb` — baseline training and full hyperparameter sweep (four runs).
   - `3_contribution_gmm.ipynb` — GMM soft-clustering experiment.
   - `4_coldstart_analysis.ipynb` — cold-start dissection (no retraining required).

Each notebook prints its final test metrics in the last cell. The complete raw output logs are also in [`results/training_logs/`](results/training_logs/) for direct inspection.

### Training configuration

SASRec backbone, hidden size 64, 2 attention layers, 2 heads, dropout 0.5, batch size 256, max sequence length 50. Adam optimiser at lr = 0.001, $\lambda_0 = 0.3$, $\beta_0 = 0.1$, false-negative mitigation enabled. Up to 200 epochs with early stopping (patience 40 on validation NDCG@20). Leave-one-out evaluation: the last item per user is the test target, the second-to-last is the validation target.

---

## What we contributed beyond the paper

We deliberately use the word **"contribution"** rather than "improvement," because one of our results was neutral/negative — and we think honest reporting matters more than overselling.

### Contribution 1: GMM soft intent clustering

ICSRec uses **hard** K-means clustering: every subsequence is assigned to exactly one of $K$ intent prototypes. We argued that grocery baskets are intrinsically multi-intent (a single basket can mix breakfast, snacks, cleaning supplies, pet food, etc.) and replaced K-means with a **Gaussian Mixture Model** (diagonal covariance, scikit-learn backend):

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \Sigma_k)$$

We implemented `GMM` as a drop-in replacement for the original `KMeans` class (same `train(x)` / `query(x)` API) and added a `--cluster_method {kmeans,gmm}` command-line switch so the rest of the codebase is untouched.

**Result:** at $K = 128$, GMM **ties K-means on NDCG@20** (both 0.0352), loses 5.3% on HIT@5, and gains 0.5% on HIT@20. Soft assignment broadens the recommendation pool slightly (helping recall at depth) but introduces a little noise in the most confident predictions (hurting top-of-list precision). For grocery, the trade does not pay off — hard K-means is doing fine.

We use $K = 128$ rather than the $K = 512$ winner from our sweep because scikit-learn's GMM on CPU is prohibitively slow at high $K$ (each clustering step takes over a minute), and the $K = 128$ comparison isolates the *soft-vs-hard* assignment question rather than mixing it with a capacity change.

### Contribution 2: Cold-start analysis

We hypothesised that ICSRec underperforms on users with short interaction histories — they produce fewer subsequences, so the intent clustering has less signal to work with. We tested this by re-evaluating our trained baseline on three test subsets without retraining: *All* users, *Cold* users (< 20 interactions), and *Warm* users (≥ 20 interactions).

**Result:** the cold-start gap is **real and largest at the top of the ranking**, where it matters most for users. The model does not collapse on cold users (HIT@20 drops only 2.7%, helped by grocery's repetitive purchase patterns), but the experience for new users is measurably worse — a clear avenue for future work.

---

## Results

### Hyperparameter sweep over the number of intents $K$ (K-means)

| $K$ | HIT@5 | NDCG@5 | HIT@10 | NDCG@10 | HIT@20 | NDCG@20 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64           | 0.0328 | 0.0216 | 0.0506 | 0.0274 | 0.0827 | 0.0354 |
| 128          | 0.0337 | 0.0215 | 0.0531 | 0.0278 | 0.0830 | 0.0352 |
| 256 (paper)  | **0.0336** | **0.0217** | 0.0514 | 0.0275 | 0.0816 | 0.0350 |
| **512**      | 0.0334 | 0.0215 | **0.0540** | **0.0281** | **0.0842** | **0.0357** |

$K = 512$ wins on four of six metrics, including both ranking metrics at depth 20.

### Contribution 1 — GMM vs K-means at $K = 128$

| Method | HIT@5 | NDCG@5 | HIT@10 | NDCG@10 | HIT@20 | NDCG@20 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| K-means | **0.0337** | **0.0215** | **0.0531** | **0.0278** | 0.0830 | 0.0352 |
| GMM     | 0.0319 | 0.0209 | 0.0523 | 0.0274 | **0.0834** | **0.0352** |
| $\Delta$ relative | $-5.3\%$ | $-2.8\%$ | $-1.5\%$ | $-1.4\%$ | $+0.5\%$ | $\pm 0.0\%$ |

### Contribution 2 — Cold-start analysis (K-means, $K = 256$, threshold = 20)

| Group | # users | HIT@5 | NDCG@5 | HIT@10 | NDCG@10 | HIT@20 | NDCG@20 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| All  | 12,713 | 0.0336 | 0.0217 | 0.0514 | 0.0275 | 0.0816 | 0.0350 |
| Cold ($<20$) | 5,941 | 0.0316 | 0.0203 | 0.0502 | 0.0263 | 0.0805 | 0.0339 |
| Warm ($\geq 20$) | 6,772 | **0.0353** | **0.0229** | **0.0526** | **0.0285** | **0.0827** | **0.0360** |
| **Warm advantage** | | $+11.7\%$ | $+12.8\%$ | $+4.8\%$ | $+8.4\%$ | $+2.7\%$ | $+6.2\%$ |

---

## Dataset

| Property | Value |
|---|---|
| Source | Amazon Reviews 2023, *Grocery and Gourmet Food* (Hou et al., 2024) |
| Hugging Face | [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| Project page | <https://amazon-reviews-2023.github.io/> |
| Original size | 3.95M interactions (5-core filtered) |
| After our 15-core filter | **329,411 interactions** |
| # users / # items | 12,713 / 8,766 |
| Avg. / median sequence length | 25.91 / 20 |
| Sparsity | 99.7044% |

See [`data/README.md`](data/README.md) for full preprocessing details.

---

## Citations

If you build on this work, please cite the original ICSRec paper and the Amazon Reviews 2023 dataset:

```bibtex
@inproceedings{qin2024icsrec,
  title     = {Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation},
  author    = {Qin, Xiuyuan and Yuan, Huanhuan and Zhao, Pengpeng and Fang, Junhua and
               Zhuang, Fuzhen and Liu, Guanfeng and Liu, Yanchi and Sheng, Victor S.},
  booktitle = {Proceedings of the 17th ACM International Conference on Web Search and Data Mining (WSDM)},
  year      = {2024},
  pages     = {548--556},
  doi       = {10.1145/3616855.3635773}
}

@article{hou2024bridging,
  title   = {Bridging Language and Items for Retrieval and Recommendation},
  author  = {Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and
             Chen, Xiusi and McAuley, Julian},
  journal = {arXiv preprint arXiv:2403.03952},
  year    = {2024}
}
```

---

## Acknowledgments

- The authors of the original ICSRec paper for releasing their code openly.
- The McAuley Lab at UC San Diego for the Amazon Reviews 2023 dataset.
- Google Colab Pro for the A100 GPU compute used in our experiments.
- Our instructor, Dr. Mohamed Hamed Mousa, for guidance throughout the project.

## License

This work is released under the [MIT License](LICENSE). The original ICSRec code referenced from <https://github.com/QinHsiu/ICSRec> is subject to its own license terms.

Trained model checkpoints (`.pt` files) are available on request — they are not included to keep the repository lightweight.
