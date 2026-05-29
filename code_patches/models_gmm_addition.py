# ============================================================
# Contribution 1: GMM soft intent clustering
# A methodological alternative to ICSRec's hard K-means clustering.
# Drop-in replacement with the same train(x) / query(x) interface,
# so the rest of the ICSRec codebase needs no changes.
#
# Added for CCAI 422 course project, University of Jeddah, Spring 2025/2026.
# Reference: see notebooks/3_contribution_gmm.ipynb for the experiment.
# ============================================================

from sklearn.mixture import GaussianMixture
import numpy as _np


class GMM(object):
    """Gaussian Mixture Model soft-clustering replacement for KMeans.

    Same interface as ICSRec's original KMeans class so it can be selected
    at runtime via --cluster_method gmm.

    Parameters
    ----------
    num_cluster : int
        Number of mixture components (analogous to K in K-means).
    seed : int
        Random seed for reproducibility.
    hidden_size : int
        Dimensionality of the input embeddings.
    gpu_id : int
        Kept for API compatibility; GMM runs on CPU via scikit-learn.
    device : str
        Torch device on which to place the resulting centroids tensor.
    """

    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        self.seed = seed
        self.num_cluster = num_cluster
        self.hidden_size = hidden_size
        self.device = device
        self.gmm = None
        self.centroids = []  # holds the L2-normalised Gaussian means
        print("Using GMM soft clustering with", num_cluster, "components")

    def train(self, x):
        """Fit a GMM on the user-intent embeddings.

        x: numpy array of shape [N, hidden_size]
        """
        if x.shape[0] <= self.num_cluster:
            # Not enough points to fit all components; reduce defensively.
            n_comp = max(2, x.shape[0] // 2)
        else:
            n_comp = self.num_cluster

        self.gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type='diag',   # fast & stable for many components
            max_iter=20,              # match K-means niter=20 for a fair compute budget
            n_init=1,
            random_state=self.seed,
            reg_covar=1e-4,           # numerical stability
        )
        self.gmm.fit(x)
        means = torch.Tensor(self.gmm.means_).to(self.device)
        # L2-normalise the means, exactly like the original KMeans did.
        self.centroids = nn.functional.normalize(means, p=2, dim=1)

    def query(self, x):
        """Assign each embedding to its most-likely component.

        x: numpy array of shape [B, hidden_size]
        Returns (seq2cluster, centroid_vectors) — same shape as KMeans.query().
        """
        labels = self.gmm.predict(x)
        seq2cluster = torch.LongTensor(labels).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]
