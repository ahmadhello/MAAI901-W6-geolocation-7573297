"""Data preparation for the 25-country study (OSV-5M).

Builds the balanced splits from the metadata, maps every selected image to its
archive on Hugging Face, and caches each image once at 256 x 256 (whole frame,
bicubic) as JPEG shards.
"""
import io, os, re, json, time, struct, zipfile, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from PIL import Image

HF = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main"
COUNTRIES = sorted("US CA MX BR AR CL PE NO ES FI SE FR GB ZA ZM BW TZ RU IN TR TH ID JP AU NZ".split())
LABEL = {c: i for i, c in enumerate(COUNTRIES)}
N_TRAIN, N_VAL, N_TEST = 1500, 200, 1000
CAP_TRAIN, CAP_TEST = 10, 5
N_BACKUP = 40          # spare images per country and split, used only if a file fails to open
SEED = 42
COLS = ["id", "country", "sequence", "creator_id"]


def load_meta(train_csv, test_csv):
    dt = {c: str for c in COLS}
    tr = pd.read_csv(train_csv, usecols=COLS, dtype=dt)
    te = pd.read_csv(test_csv, usecols=COLS, dtype=dt)
    tr = tr[tr.country.isin(COUNTRIES)].reset_index(drop=True)
    te = te[te.country.isin(COUNTRIES)].reset_index(drop=True)
    return tr, te


def _draw(groups, order, cap, n_needed, n_backup, rng, taken):
    """Walk sequences in `order`, taking random images from each until n_needed picks
    (then n_backup spares) are collected. `taken` counts images already drawn from each
    sequence across all countries, so a sequence that crosses a border still respects the cap."""
    picks, spares = [], []
    for s in order:
        if len(picks) >= n_needed and len(spares) >= n_backup:
            break
        room = cap - taken.get(s, 0)
        if room <= 0:
            continue
        ids = groups[s]
        chosen = list(rng.choice(ids, size=min(room, len(ids)), replace=False))
        for i in chosen:
            if len(picks) < n_needed:
                picks.append(i)
            elif len(spares) < n_backup:
                spares.append(i)
            else:
                break
            taken[s] = taken.get(s, 0) + 1
    return picks, spares


def build_splits(tr, te, seed=SEED):
    rngs = {c: np.random.RandomState(seed + LABEL[c]) for c in COUNTRIES}
    tg = {c: {s: sorted(g.id.tolist()) for s, g in tr[tr.country == c].groupby("sequence", sort=True)} for c in COUNTRIES}
    eg = {c: {s: sorted(g.id.tolist()) for s, g in te[te.country == c].groupby("sequence", sort=True)} for c in COUNTRIES}
    out, val_seqs = {}, set()
    # 1) validation: whole sequences, at most 10 images each; Botswana uses short sequences only.
    #    Every sequence touched here is removed from training (in all countries).
    for c in COUNTRIES:
        rng, groups = rngs[c], tg[c]
        cand = [s for s in sorted(groups) if c != "BW" or len(groups[s]) <= CAP_TRAIN]
        val, val_sp = [], []
        for s in rng.permutation(cand):
            if len(val) >= N_VAL and len(val_sp) >= 10:
                break
            ids = groups[s]
            val_seqs.add(s)
            for i in rng.choice(ids, size=min(CAP_TRAIN, len(ids)), replace=False):
                if len(val) < N_VAL:
                    val.append(i)
                elif len(val_sp) < 10:
                    val_sp.append(i)
        out[c] = {"val": val, "val_spare": val_sp}
    # 2) training: remaining sequences, at most 10 images per sequence (cap lifted for Botswana)
    taken = {}
    for c in COUNTRIES:
        rng = rngs[c]
        pool = {s: ids for s, ids in tg[c].items() if s not in val_seqs}
        if c == "BW":
            perm = list(rng.permutation(sorted(i for ids in pool.values() for i in ids)))
            trn, trn_sp = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_BACKUP]
        else:
            trn, trn_sp = _draw(pool, list(rng.permutation(sorted(pool))), CAP_TRAIN, N_TRAIN, N_BACKUP, rng, taken)
        out[c].update(train=trn, train_spare=trn_sp)
    # 3) test: official test set, at most 5 images per sequence
    taken_t = {}
    for c in COUNTRIES:
        rng, groups = rngs[c], eg[c]
        tst, tst_sp = _draw(groups, list(rng.permutation(sorted(groups))), CAP_TEST, N_TEST, N_BACKUP, rng, taken_t)
        out[c].update(test=tst, test_spare=tst_sp)
    rows = []
    for c in COUNTRIES:
        assert len(out[c]["val"]) == N_VAL and len(out[c]["train"]) == N_TRAIN and len(out[c]["test"]) == N_TEST, c
        for split in ["train", "train_spare", "val", "val_spare", "test", "test_spare"]:
            for rank, i in enumerate(out[c][split]):
                rows.append((i, split, c, LABEL[c], rank))
    df = pd.DataFrame(rows, columns=["id", "split", "country", "label", "rank"])
    meta = pd.concat([tr.assign(src="train"), te.assign(src="test")])
    df = df.merge(meta[["id", "sequence", "creator_id", "src"]], on="id", how="left")
    assert df.sequence.notna().all() and not df.id.duplicated().any()
    return df


def split_checksum(df):
    core = df[df.split.isin(["train", "val", "test"])].sort_values(["split", "id"])
    return hashlib.md5(("|".join(core.split + ":" + core.id)).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- archive index
def _final_url(sess, url):
    r = sess.head(url, allow_redirects=True, timeout=60)
    return r.url if r.status_code == 200 else url


def list_archive(url):
    from remotezip import RemoteZip
    with RemoteZip(url) as z:
        out = []
        for i in z.infolist():
            if i.filename.endswith(".jpg"):
                out.append((os.path.basename(i.filename)[:-4], i.header_offset, i.compress_size, i.compress_type))
    return out


def build_index(needed_ids, which, n_archives, workers=8, log=print):
    """Map each needed id to (archive url, offset, size, compression)."""
    needed = set(needed_ids)
    urls = [f"{HF}/images/{which}/{k:02d}.zip" for k in range(n_archives)]
    found = {}
    def job(k):
        return k, list_archive(urls[k])
    with ThreadPoolExecutor(workers) as ex:
        for k, entries in ex.map(job, range(n_archives)):
            for i, off, size, comp in entries:
                if i in needed:
                    found[i] = (k, off, size, comp)
    log(f"{which}: indexed {len(found)}/{len(needed)} images from {n_archives} archives")
    return found, urls


# ---------------------------------------------------------------- fetch + cache
_local = threading.local()

def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fetch_bytes(url, off, size, comp, tries=4):
    for t in range(tries):
        try:
            r = _session().get(url, headers={"Range": f"bytes={off}-{off + 30 + 512 + size}"}, timeout=60)
            b = r.content
            sig, _, _, cm, _, _, _, _, _, nl, el = struct.unpack("<IHHHHHIIIHH", b[:30])
            if sig != 0x04034B50:
                raise ValueError("bad header")
            data = b[30 + nl + el: 30 + nl + el + size]
            if cm == 8:
                import zlib
                data = zlib.decompress(data, -15)
            return data
        except Exception:
            time.sleep(1 + 2 * t)
    return None


def to_cache_jpeg(data, side=256, quality=90):
    im = Image.open(io.BytesIO(data))
    im.draft("RGB", (side, side))                      # fast JPEG decode at reduced scale (still >= side)
    im = im.convert("RGB").resize((side, side), Image.BICUBIC)   # whole frame, no crop
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def cache_split(ids, index, urls, out_dir, prefix, shard_size=2500, workers=48, log=print):
    """Download, resize and store images in shard zips; returns ids that failed.
    Finished shards are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    final = {}
    sess = requests.Session()
    failed = []
    shards = [ids[i:i + shard_size] for i in range(0, len(ids), shard_size)]
    for si, shard in enumerate(shards):
        path = os.path.join(out_dir, f"{prefix}_{si:03d}.zip")
        if os.path.exists(path):
            continue
        t0 = time.time()
        tmp = f"/tmp/{prefix}_{si:03d}.zip"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf, ThreadPoolExecutor(workers) as ex:
            def job(i):
                k, off, size, comp = index[i]
                if k not in final:
                    final[k] = _final_url(sess, urls[k])
                data = fetch_bytes(final[k], off, size, comp)
                if data is None:
                    return i, None
                try:
                    return i, to_cache_jpeg(data)
                except Exception:
                    return i, None
            for fut in as_completed([ex.submit(job, i) for i in shard if i in index]):
                i, jpg = fut.result()
                if jpg is None:
                    failed.append(i)
                else:
                    zf.writestr(f"{i}.jpg", jpg)
        missing = [i for i in shard if i not in index]
        failed += missing
        import shutil
        shutil.move(tmp, path)
        log(f"{prefix} shard {si + 1}/{len(shards)}: {len(shard)} images in {time.time() - t0:.0f}s")
    return failed


def cached_ids(out_dir, prefix):
    have = set()
    for f in sorted(os.listdir(out_dir)):
        if f.startswith(prefix + "_") and f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(out_dir, f)) as z:
                have.update(n[:-4] for n in z.namelist())
    return have
