# ---- Compression schemes (codecs) ----
# Each codec has three jobs:
#   fit(X)      learn what it needs from the victim database (e.g. PQ codebooks, SVD basis)
#   encode(X)   turn float vectors into what is actually stored (the compressed codes)
#   decode(C)   turn stored codes back into float vectors -- this is what an attacker
#               gets after stealing the index (the codebooks are stored inside the index)
#   search(Q,k) how a real vector DB with this compression answers a query
import numpy as np
import faiss

class Codec:
    name = "base"
    def fit(self, X): return self
    def encode(self, X): raise NotImplementedError
    def decode(self, C): raise NotImplementedError
    def bytes_per_vector(self): raise NotImplementedError
    def build(self, X):
        """Fit on the victim DB, store its compressed codes."""
        self.fit(X)
        self.codes = self.encode(X)
        return self
    def search(self, Q, k):
        raise NotImplementedError


class FP32(Codec):
    """Control: no compression."""
    name = "fp32"
    def fit(self, X):
        self.d = X.shape[1]; return self
    def encode(self, X): return np.ascontiguousarray(X, dtype=np.float32)
    def decode(self, C): return np.asarray(C, dtype=np.float32)
    def bytes_per_vector(self): return 4 * self.d
    def search(self, Q, k):
        index = faiss.IndexFlatIP(self.d)
        index.add(self.codes)
        return index.search(np.ascontiguousarray(Q, dtype=np.float32), k)[1]


class FaissCodec(Codec):
    """int8 / int4 scalar quantisation and product quantisation, all through FAISS.
    Encoding and decoding use the SAME trained index (sa_encode / sa_decode),
    so decode always matches how the DB stored the vector."""
    def __init__(self, name, factory):
        self.name, self.factory = name, factory
    def fit(self, X):
        self.d = X.shape[1]
        self.index = faiss.index_factory(self.d, self.factory, faiss.METRIC_INNER_PRODUCT)
        self.index.train(np.ascontiguousarray(X, dtype=np.float32))
        return self
    def build(self, X):
        self.fit(X)
        self.index.add(np.ascontiguousarray(X, dtype=np.float32))
        self.codes = self.encode(X)
        return self
    def encode(self, X): return self.index.sa_encode(np.ascontiguousarray(X, dtype=np.float32))
    def decode(self, C): return self.index.sa_decode(C)
    def bytes_per_vector(self): return self.index.sa_code_size()
    def search(self, Q, k):
        # Asymmetric search: the query stays float, the DB is compressed.
        # This is what a real FAISS SQ/PQ index does.
        return self.index.search(np.ascontiguousarray(Q, dtype=np.float32), k)[1]


class Binary(Codec):
    """1 bit per dimension: keep only the sign."""
    name = "binary"
    def fit(self, X):
        self.d = X.shape[1]; return self
    def encode(self, X): return np.packbits((X > 0).astype(np.uint8), axis=1)
    def decode(self, C):
        bits = np.unpackbits(C, axis=1)[:, :self.d].astype(np.float32)
        return (2.0 * bits - 1.0) / np.sqrt(self.d)   # +-1, scaled to unit length
    def bytes_per_vector(self): return self.d // 8
    def search(self, Q, k):
        # Hamming search: queries are binarised the same way as the DB.
        index = faiss.IndexBinaryFlat(self.d)
        index.add(self.codes)
        return index.search(self.encode(Q), k)[1]


class SVD(Codec):
    """Keep only the top-r directions of the victim DB (dimensionality reduction)."""
    def __init__(self, r=256):
        self.r = r; self.name = f"svd{r}"
    def fit(self, X):
        self.d = X.shape[1]
        _, _, Vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
        self.V = Vt[:self.r].T.astype(np.float32)     # d x r
        return self
    def encode(self, X): return np.ascontiguousarray(X @ self.V, dtype=np.float32)
    def decode(self, C): return np.asarray(C, dtype=np.float32)   # attacker sees the r-dim vectors as they are
    def bytes_per_vector(self): return 4 * self.r
    def search(self, Q, k):
        # Queries must go through the same projection, or the dimensions do not match.
        index = faiss.IndexFlatIP(self.r)
        index.add(self.codes)
        return index.search(self.encode(Q), k)[1]


def make_codecs():
    return [
        FP32(),
        FaissCodec("int8", "SQ8"),
        FaissCodec("int4", "SQ4"),
        FaissCodec("pq128x8", "PQ128x8"),   # 128 sub-vectors of 8 dims, 8 bits each
        Binary(),
        SVD(256),
    ]


def roundtrip_check(codec, X):
    """Correctness tests for one codec. Raises if something is wrong."""
    C = codec.encode(X)
    Xr = codec.decode(C)
    out = {"scheme": codec.name, "bytes_per_vector": int(codec.bytes_per_vector())}
    if Xr.shape[1] == X.shape[1]:
        out["rel_error"] = float(np.linalg.norm(X - Xr) / np.linalg.norm(X))
        out["mean_cos_to_original"] = float(np.mean(
            np.sum(X * Xr, 1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Xr, axis=1) + 1e-12)))
    else:
        out["rel_error"] = float("nan"); out["mean_cos_to_original"] = float("nan")
    # 1) stored code size is what we claim
    assert C.nbytes == len(X) * codec.bytes_per_vector(), (codec.name, C.nbytes)
    # 2) re-encoding a decoded vector gives the same code (decode really inverts encode)
    if codec.name not in ("fp32", "svd256"):
        C2 = codec.encode(Xr)
        out["reencode_match"] = float(np.mean(np.all(C2 == C, axis=1)))
        assert out["reencode_match"] > 0.99, (codec.name, out["reencode_match"])
    # 3) the index's own search agrees with brute force on decoded vectors
    q = X[:200]
    top_index = codec.search(q, 1)[:, 0]
    if codec.name == "binary":
        top_brute = np.argmax(codec.decode(codec.encode(q)) @ codec.decode(codec.codes).T, axis=1)
    elif codec.name.startswith("svd"):
        top_brute = np.argmax(codec.encode(q) @ codec.decode(codec.codes).T, axis=1)
    else:
        top_brute = np.argmax(q @ codec.decode(codec.codes).T, axis=1)
    out["search_agrees"] = float(np.mean(top_index == top_brute))
    assert out["search_agrees"] > 0.95, (codec.name, out["search_agrees"])
    return out


def topk_recall(codec, Q, gold, k=10):
    I = codec.search(Q, k)
    return float(np.mean([gold[i] in I[i] for i in range(len(Q))]))


# ---- Supervised linear oracle (upper bound for any linear alignment) ----
def fit_ridge(X, Y, lam):
    """Closed-form ridge regression with bias: Y ~ X W + b."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ Yc)
    b = my - mx @ W
    return W.astype(np.float32), b.astype(np.float32)


def cos_rows(A, B):
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def fit_oracle(X_att, Y_att, lams=(0.1, 1.0, 10.0, 100.0), n_val=1000):
    """Pick the ridge strength on a held-out slice of the attacker's OWN data."""
    Xtr, Ytr, Xva, Yva = X_att[:-n_val], Y_att[:-n_val], X_att[-n_val:], Y_att[-n_val:]
    best = None
    for lam in lams:
        W, b = fit_ridge(Xtr, Ytr, lam)
        c = float(np.mean(cos_rows(Xva @ W + b, Yva)))
        if best is None or c > best[0]:
            best = (c, lam)
    W, b = fit_ridge(X_att, Y_att, best[1])
    return W, b, best[1]


def alignment_scores(Y_hat, Y_true):
    """top-1: is the translated vector closest to its own true GTR vector
    among all candidates in the pool? (pool = all rows of Y_true)"""
    A = Y_hat / np.linalg.norm(Y_hat, axis=1, keepdims=True)
    B = Y_true / np.linalg.norm(Y_true, axis=1, keepdims=True)
    S = A @ B.T
    top1 = float(np.mean(np.argmax(S, 1) == np.arange(len(A))))
    ranks = (S > S[np.arange(len(A)), np.arange(len(A))][:, None]).sum(1)
    return {"align_top1": top1, "align_mean_rank": float(ranks.mean() + 1),
            "align_mean_cos": float(np.mean(np.diag(S)))}
