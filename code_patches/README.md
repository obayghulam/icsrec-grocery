# Code patches for ICSRec

This folder contains four small modifications we applied to the official ICSRec source code (<https://github.com/QinHsiu/ICSRec>) to (a) make it run on Google Colab without `faiss-gpu`, and (b) add our **Contribution 1: GMM soft intent clustering** as an optional clustering backend.

We deliberately **do not vendor** the original ICSRec codebase here. To reproduce our experiments, clone the official repo and apply the patches below.

---

## Files in this folder

| File | What it does | Where it applies |
|---|---|---|
| `models_faiss_cpu.patch` | Replace 6 lines of GPU-only `faiss` setup with a single CPU `faiss.IndexFlatL2` line | `src/models.py` |
| `models_gmm_addition.py` | A new `GMM` class with the same `train(x)` / `query(x)` interface as ICSRec's `KMeans`, implemented on top of `sklearn.mixture.GaussianMixture` | append to `src/models.py` |
| `trainers_cluster_switch.patch` | Add a runtime switch between K-means and GMM based on `args.cluster_method` | `src/trainers.py` |
| `main_cluster_method_arg.patch` | Add the `--cluster_method {kmeans,gmm}` command-line argument | `src/main.py` |

---

## How to apply (the easy way)

Our four Jupyter notebooks apply these patches **automatically** when they clone the ICSRec repo. If you run the notebooks in order, you do not need to touch the patches by hand.

If you want to inspect what they do, the patch files use a simplified unified-diff format with explanatory comments — they are easier to read than to apply mechanically.

---

## How to apply (manual, if you want a clean ICSRec copy)

```bash
# 1. Clone the official ICSRec repo
git clone https://github.com/QinHsiu/ICSRec.git
cd ICSRec

# 2. Open src/models.py and apply models_faiss_cpu.patch by hand.
#    (Find the 6-line GPU faiss block in the KMeans class and replace it
#    with the single CPU line shown in the patch.)

# 3. Append the GMM class to src/models.py:
cat path/to/code_patches/models_gmm_addition.py >> src/models.py

# 4. Open src/trainers.py and apply trainers_cluster_switch.patch by hand.
#    (Update the `from models import KMeans` line, and replace the
#    `cluster = KMeans(...)` instantiation block.)

# 5. Open src/main.py and apply main_cluster_method_arg.patch by hand.
#    (Insert one new parser.add_argument line for --cluster_method.)
```

After that, you can run training with either clustering method:

```bash
# Original ICSRec (K-means clustering)
python src/main.py --data_name Grocery_and_Gourmet_Food --intent_num 512 \
    --cluster_method kmeans --model_idx 2 --epochs 200

# Our GMM variant (Contribution 1)
python src/main.py --data_name Grocery_and_Gourmet_Food --intent_num 128 \
    --cluster_method gmm --model_idx 11 --epochs 200
```

---

## Why these patches?

- **`models_faiss_cpu.patch`** — ICSRec hardcodes `faiss-gpu`, which is not installable on modern Colab (CUDA 12). We fall back to `faiss-cpu`. Per-epoch K-means on CPU is slightly slower but acceptable for our dataset size (~12.7K subsequences × 64 dims).
- **`models_gmm_addition.py`** — adds the new `GMM` class. Same interface as `KMeans`, so it drops in without touching the rest of the codebase. Uses `sklearn.mixture.GaussianMixture` with diagonal covariance and `max_iter=20` to match K-means's iteration budget for a fair compute comparison.
- **`trainers_cluster_switch.patch`** — chooses between `KMeans` and `GMM` at runtime instead of hard-editing the source for every experiment.
- **`main_cluster_method_arg.patch`** — exposes the choice on the command line.

All four patches together preserve the original behaviour by default (`--cluster_method kmeans` is the default and reproduces vanilla ICSRec), so existing reproductions of the paper continue to work unchanged.
