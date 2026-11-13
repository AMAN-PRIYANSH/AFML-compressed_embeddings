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
