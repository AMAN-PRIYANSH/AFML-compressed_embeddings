# ---- mini-vec2vec (Dar 2025, arXiv 2510.02348) ----
# Follows the official notebook github.com/guy-dar/mini-vec2vec (commit 948de01, 11 Oct 2025).
# One change: the paper only aligns spaces of the SAME size (768 -> 768). BGE-large is 1024-d and
# GTR is 768-d, so Procrustes is solved with an SVD that allows a rectangular map
# (W has orthonormal columns). For equal sizes this is exactly the paper's Procrustes.
import numpy as np
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans

def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def procrustes(A, B):
    """W minimising ||A W - B|| with orthonormal columns/rows (rectangular-safe)."""
    U, _, Vt = np.linalg.svd(A.T.astype(np.float64) @ B.astype(np.float64), full_matrices=False)
    return (U @ Vt).astype(np.float32)

def _double_centered_gram(C):
    # same as the official sim(): H C C^T H
    G = C @ C.T
    n = len(C)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ G @ H

def aligned_centroids(X, Y, rng, n_clusters=20, n_qap=30, subsample=10_000):
    xs = X[rng.permutation(len(X))[:subsample]]
    ys = Y[rng.permutation(len(Y))[:subsample]]
    ca = KMeans(n_clusters=n_clusters, n_init=1, random_state=int(rng.integers(1 << 30))).fit(xs).cluster_centers_
    cb = KMeans(n_clusters=n_clusters, n_init=1, random_state=int(rng.integers(1 << 30))).fit(ys).cluster_centers_
    ka, kb = _double_centered_gram(ca), _double_centered_gram(cb)
    best = None
    for _ in range(n_qap):   # 2-opt is a local search, so restart it and keep the best
        r = quadratic_assignment(ka, kb, method="2opt",
                                 options={"P0": "randomized", "maximize": True})
        if best is None or r.fun > best.fun:
            best = r
    return ca.astype(np.float32), cb[best.col_ind].astype(np.float32)

def _topk_mean(S_rows_fn, n_rows, Y, k, chunk=2048):
    """For each row i, average the k rows of Y with highest score S[i, :]."""
    out = np.empty((n_rows, Y.shape[1]), dtype=np.float32)
    for s in range(0, n_rows, chunk):
        S = S_rows_fn(s, min(s + chunk, n_rows))
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        out[s:s + len(S)] = Y[idx].mean(axis=1)
    return out

def mini_vec2vec(XA, XB, seed=0, n_runs=30, n_clusters=20, n_qap=30, k=50,
                 refine1_iters=100, refine1_sample=1000, refine1_k=50,
                 refine2_clusters=500, alpha=0.5, log=print, eval_fn=None):
    """XA: victim vectors (centered + normalised). XB: attacker vectors (centered + normalised).
    No pairs between them. Returns W (dA x dB) so that XA @ W lives in B's space."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)   # scipy 2-opt draws its random start from here (works on old and new scipy)
    hist = {}
    # Step 1: approximate matching via cluster anchors matched by QAP, ensembled over n_runs
    CA, CB = [], []
    for r in range(n_runs):
        ca, cb = aligned_centroids(XA, XB, rng, n_clusters, n_qap)
        CA.append(ca); CB.append(cb)
    CA, CB = l2n(np.concatenate(CA)), l2n(np.concatenate(CB))
    RA, RB = l2n(l2n(XA) @ CA.T), l2n(l2n(XB) @ CB.T)          # relative representations
    YM = _topk_mean(lambda a, b: RA[a:b] @ RB.T, len(XA), XB, k)
    # Step 2: initial transformation
    W = procrustes(XA, YM)
    hist["initial"] = eval_fn(W) if eval_fn else None
    log("initial", hist["initial"])
    # Step 3a: Refine-1 (iterative closest point, averaged neighbours)
    for t in range(refine1_iters):
        xs = XA[rng.permutation(len(XA))[:refine1_sample]]
        S = l2n(xs @ W) @ XB.T
        idx = np.argpartition(-S, refine1_k, axis=1)[:, :refine1_k]
        W = (1 - alpha) * W + alpha * procrustes(xs, XB[idx].mean(axis=1))
    hist["refine1"] = eval_fn(W) if eval_fn else None
    log("refine1", hist["refine1"])
    # Step 3b: Refine-2 (one round of cluster-based correction)
    km1 = KMeans(n_clusters=refine2_clusters, n_init=1, random_state=seed).fit(XA)
    c1 = km1.cluster_centers_.astype(np.float32)
    km2 = KMeans(n_clusters=refine2_clusters, init=c1 @ W, n_init=1).fit(XB)
    W = (1 - alpha) * W + alpha * procrustes(c1, km2.cluster_centers_.astype(np.float32))
    hist["refine2"] = eval_fn(W) if eval_fn else None
    log("refine2", hist["refine2"])
    return W, hist

def prep(X, mean):
    return l2n(X - mean).astype(np.float32)

def align_eval(XA_eval, YB_eval, W):
    """Top-1 accuracy, average rank (1 = perfect) and mean cosine on paired eval vectors."""
    S = l2n(XA_eval @ W) @ l2n(YB_eval).T
    d = np.diag(S)
    ranks = (S > d[:, None]).sum(1) + 1
    return {"top1": float((ranks == 1).mean()), "avg_rank": float(ranks.mean()), "mean_cos": float(d.mean())}
