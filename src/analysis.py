"""Evaluation and statistics: metrics, McNemar (H1), sequence bootstrap (H2), figures."""
import json, math
import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest
from sklearn.metrics import f1_score, confusion_matrix

LOOKALIKE = [["US", "CA"], ["AR", "CL"], ["NO", "SE", "FI"], ["ZA", "BW", "ZM"], ["TH", "ID"], ["AU", "NZ"]]


def mcnemar(correct_a, correct_b):
    """Continuity-corrected McNemar test on paired predictions (Dietterich, 1998)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / max(1, n01 + n10)
    return {"a_only": n01, "b_only": n10, "chi2": stat, "p": float(chi2.sf(stat, 1)),
            "p_exact": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 > 0 else 1.0}


def basic_metrics(pred, y, countries):
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, average="macro"))
    rec = {c: float((pred[y == k] == k).mean()) for k, c in enumerate(countries)}
    return {"top1": acc, "macro_f1": f1, "recall": rec}


def scene_table(test, preds, models, seeds, mask=None):
    """Urban and natural accuracy per model, seed and country."""
    rows = []
    t = test if mask is None else test[mask]
    for m in models:
        for s in seeds:
            ok = preds[(m, s)][t.index.values] == t.label.values
            for c, g in t.assign(ok=ok).groupby("country"):
                for sc in ["urban", "natural"]:
                    gg = g[g.scene == sc]
                    rows.append((m, s, c, sc, len(gg), gg.ok.mean() if len(gg) else np.nan))
    return pd.DataFrame(rows, columns=["model", "seed", "country", "scene", "n", "acc"])


def eligible_countries(test, mask=None, k=50):
    t = test if mask is None else test[mask]
    n = t.groupby(["country", "scene"]).size().unstack(fill_value=0)
    for col in ["urban", "natural"]:
        if col not in n:
            n[col] = 0
    return sorted(n[(n.urban >= k) & (n.natural >= k)].index.tolist())


def _drop_from_weights(ok, w, country_idx, urban, natural, keep, pooled, n_c):
    if pooled:
        u = (ok * w * urban).sum() / max(1e-9, (w * urban).sum())
        n = (ok * w * natural).sum() / max(1e-9, (w * natural).sum())
        return 100 * (u - n)
    cu = np.bincount(country_idx, ok * w * urban, n_c) / np.maximum(1e-9, np.bincount(country_idx, w * urban, n_c))
    cn = np.bincount(country_idx, ok * w * natural, n_c) / np.maximum(1e-9, np.bincount(country_idx, w * natural, n_c))
    return 100 * float(np.mean((cu - cn)[keep]))


def drop_estimate(test, preds, model, seeds, countries, pooled, weights=None):
    """Urban minus natural accuracy (percentage points): per eligible country, averaged over
    countries (or on the pooled set), then averaged over seeds. Returns (mean, per-seed list)."""
    t = test.reset_index(drop=True)
    names = sorted(t.country.unique())
    ci = t.country.map({c: k for k, c in enumerate(names)}).values
    keep = np.array([c in countries for c in names])
    urban = (t.scene.values == "urban").astype(float); natural = (t.scene.values == "natural").astype(float)
    w = np.ones(len(t)) if weights is None else weights
    per = [_drop_from_weights((preds[(model, s)] == t.label.values).astype(float), w, ci, urban, natural, keep,
                              pooled, len(names)) for s in seeds]
    return float(np.mean(per)), per


def h2_bootstrap(test, preds, seeds, countries, pooled, B=2000, seed=0, a="clip", b="vit"):
    """95% interval for drop(a) - drop(b), resampling whole test sequences within each country.
    The same resamples are applied to every seed and both models."""
    rng = np.random.RandomState(seed)
    t = test.reset_index(drop=True)
    names = sorted(t.country.unique())
    ci = t.country.map({c: k for k, c in enumerate(names)}).values
    keep = np.array([c in countries for c in names])
    urban = (t.scene.values == "urban").astype(float); natural = (t.scene.values == "natural").astype(float)
    oks = {(m, s): (preds[(m, s)] == t.label.values).astype(float) for m in (a, b) for s in seeds}
    # per country: local sequence index of every row
    blocks = []
    for c in names:
        rows = np.where(t.country.values == c)[0]
        codes, local = np.unique(t.sequence.values[rows], return_inverse=True)
        blocks.append((rows, local, len(codes)))
    diffs = np.empty(B)
    for i in range(B):
        w = np.zeros(len(t))
        for rows, local, n in blocks:
            cnt = np.bincount(rng.randint(0, n, n), minlength=n)
            w[rows] = cnt[local]
        da = np.mean([_drop_from_weights(oks[(a, s)], w, ci, urban, natural, keep, pooled, len(names)) for s in seeds])
        db = np.mean([_drop_from_weights(oks[(b, s)], w, ci, urban, natural, keep, pooled, len(names)) for s in seeds])
        diffs[i] = da - db
    return {"lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)), "samples": diffs}


def top_confusions(pred, y, countries, k=10):
    cm = confusion_matrix(y, pred, labels=range(len(countries)))
    off = [(cm[i, j], countries[i], countries[j]) for i in range(len(countries)) for j in range(len(countries)) if i != j]
    return sorted(off, reverse=True)[:k], cm


def lookalike_share(pred, y, countries):
    """Share of errors that stay inside a look-alike group."""
    gid = {}
    for g, members in enumerate(LOOKALIKE):
        for c in members:
            gid[c] = g
    err = pred != y
    inside = [gid.get(countries[a], -1) == gid.get(countries[b], -2) for a, b in zip(y[err], pred[err])]
    return float(np.mean(inside)) if len(inside) else float("nan")
