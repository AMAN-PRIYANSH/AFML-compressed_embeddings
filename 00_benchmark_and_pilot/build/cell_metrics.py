# ---- Text metrics ----
import re
import sacrebleu

def _tok(s):
    return re.findall(r"\w+", s.lower())

def token_f1(pred, ref):
    p, r = _tok(pred), _tok(ref)
    if not p or not r:
        return 0.0
    rc = {}
    for w in r:
        rc[w] = rc.get(w, 0) + 1
    overlap = 0
    for w in p:
        if rc.get(w, 0) > 0:
            overlap += 1; rc[w] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)

def text_scores(preds, refs):
    return {
        "bleu": float(sacrebleu.corpus_bleu(preds, [refs]).score),   # 0-100
        "token_f1": float(np.mean([token_f1(p, r) for p, r in zip(preds, refs)])) * 100,
        "exact_match": float(np.mean([p.strip() == r.strip() for p, r in zip(preds, refs)])) * 100,
    }
