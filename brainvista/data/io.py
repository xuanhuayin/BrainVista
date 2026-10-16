# brainvista/data/io.py
"""Run lists and on-disk lookup of fMRI, stimulus-feature and token files.

Expected layouts (see README, "Data preparation"):

    FMRI_ROOT/<network>/<subj>/<rid>.npy          (T, P_k)  model-space fMRI
    VIDEO_ROOT/<subj>/<rid>.npy                   (T_stim, D_v)   or VIDEO_ROOT/<rid>.npy
    AUDIO_ROOT/<subj>/<rid>.npy                   (T_stim, D_a)
    TEXT_ROOT/<subj>/<rid>.npy                    (T_stim, D_w)
    Z_ROOT/z_<network>/<subj>/<rid>_z.npy         (T, d_k)  exported brain region tokens

Files may also sit in nested sub-directories below the subject directory.
"""
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RunList = List[Tuple[str, str]]


def parse_run_list(path: str, subjects: Optional[Sequence[str]] = None) -> RunList:
    """Read a run list with one `subj rid` (or bare `rid`) per line.

    Bare run ids are expanded to every subject in `subjects`.
    """
    out: RunList = []
    with open(path) as f:
        for ln in f:
            s = ln.split("#", 1)[0].strip()
            if not s:
                continue
            toks = [t for t in s.replace(",", " ").split() if t]
            if len(toks) == 1:
                if not subjects:
                    raise ValueError(f"{path}: bare run id {toks[0]!r} needs --subjects")
                out.extend((subj, toks[0]) for subj in subjects)
            else:
                out.append((toks[0], toks[1]))
    if subjects:
        keep = set(subjects)
        out = [(s, r) for s, r in out if s in keep]
    return out


def runs_by_subject(run_list: RunList) -> Dict[str, List[str]]:
    by: Dict[str, List[str]] = {}
    for subj, rid in run_list:
        by.setdefault(subj, []).append(rid)
    return by


def subject_variants(subj: str) -> List[str]:
    """`sub-01` <-> `sub01` <-> `sub1` spellings used by different releases."""
    variants = [subj]
    m = re.match(r"sub-?0*(\d+)$", subj)
    if m:
        n = int(m.group(1))
        variants += [f"sub-{n:02d}", f"sub{n:02d}", f"sub{n}", f"sub-{n}"]
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v); out.append(v)
    return out


@lru_cache(maxsize=None)
def _index_dir(root: str) -> Dict[str, str]:
    """Map file stem -> path for every .npy below `root` (first hit wins)."""
    idx: Dict[str, str] = {}
    if not os.path.isdir(root):
        return idx
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith(".npy"):
                idx.setdefault(fn[:-4], os.path.join(dirpath, fn))
    return idx


def find_run_file(root: Optional[str], subj: str, rid: str,
                  suffixes: Sequence[str] = ("",), allow_shared: bool = False) -> Optional[str]:
    """Locate `<rid><suffix>.npy` for `subj` below `root/<subj>/` (any subject spelling).

    With `allow_shared`, subject-independent files directly below `root/` are
    accepted as a fallback (stimulus features shared by all subjects).
    """
    if not root:
        return None
    subj_dirs = [os.path.join(root, v) for v in subject_variants(subj)]
    for d in subj_dirs:
        for suf in suffixes:
            p = os.path.join(d, rid + suf + ".npy")
            if os.path.exists(p):
                return p
    for d in subj_dirs:
        idx = _index_dir(d)
        for suf in suffixes:
            if rid + suf in idx:
                return idx[rid + suf]
    if allow_shared:
        for suf in suffixes:
            p = os.path.join(root, rid + suf + ".npy")
            if os.path.exists(p):
                return p
    return None


def find_fmri_file(fmri_root: str, network: str, subj: str, rid: str) -> Optional[str]:
    """fMRI of one network: FMRI_ROOT/<network>/<subj>/<rid>.npy (or FMRI_ROOT/<subj>/<network>/)."""
    p = find_run_file(os.path.join(fmri_root, network), subj, rid)
    if p is not None:
        return p
    for v in subject_variants(subj):
        q = os.path.join(fmri_root, v, network, rid + ".npy")
        if os.path.exists(q):
            return q
    return None


def find_token_file(z_root: str, network: str, subj: str, rid: str) -> Optional[str]:
    return find_run_file(os.path.join(z_root, f"z_{network}"), subj, rid, suffixes=("_z", ""))


def find_feature_file(feat_root: Optional[str], subj: str, rid: str) -> Optional[str]:
    return find_run_file(feat_root, subj, rid, allow_shared=True)


def load_npy(fp: str) -> np.ndarray:
    arr = np.load(fp, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        arr = arr[list(arr.keys())[0]]
    return np.asarray(arr, dtype=np.float32)
