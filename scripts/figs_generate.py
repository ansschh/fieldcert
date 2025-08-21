# scripts/figs_generate.py
from __future__ import annotations
import os, sys, argparse, glob, json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- style ----------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

METHOD_STYLE = {
    "fieldcert_crc": {"label": "FieldCert (CRC)", "color": "k",    "ls": "-",  "lw": 1.8, "marker": "o"},
    "global_bump":   {"label": "Global bump",     "color": "C0",   "ls": "--", "lw": 1.5, "marker": "s"},
    "morph_cp":      {"label": "Morph CP",        "color": "C1",   "ls": "-.", "lw": 1.5, "marker": "D"},
    "pixel_cp":      {"label": "Pixel CP",        "color": "C2",   "ls": ":",  "lw": 1.5, "marker": "^"},
    "prob_isotonic": {"label": "Prob (isotonic)", "color": "C3",   "ls": "-",  "lw": 1.5, "marker": "v"},
}

# ---------- helpers ----------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_summaries(paths_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(paths_glob))
    if not paths:
        raise FileNotFoundError(f"No summary.csv files match: {paths_glob}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        # infer alpha from parent dir name like alpha_0.10
        alpha = None
        for part in os.path.dirname(p).split(os.sep):
            if part.startswith("alpha_"):
                alpha = float(part.split("_", 1)[1])
                break
        if alpha is None and "alpha" in df.columns:
            alpha = float(df["alpha"].iloc[0])
        df["alpha_from_dir"] = alpha
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    # normalize method names (some runs may have 'fieldcert' vs 'fieldcert_crc')
    out["method"] = out["method"].replace({"fieldcert": "fieldcert_crc"})
    return out

def fig_save_with_json(fig, data_obj: Dict[str, Any], out_pdf: str, out_json: str):
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    with open(out_json, "w") as f:
        json.dump(data_obj, f, indent=2)
    print(f"[OK] wrote {out_pdf} & {out_json}")

# ---------- F2: Validity curves ----------
def plot_validity_curves(df: pd.DataFrame, provider: str, variables: List[str], leads: List[int], outdir: str):
    """
    Plots empirical FPA vs target alpha for each method, separately for each (variable, lead).
    Input df must contain columns: provider, variable, lead_hours, alpha_from_dir/alpha, method, fpa
    """
    cols = set(df.columns)
    sub = df[df["provider"] == provider].copy() if ("provider" in cols) else df.copy()
    if ("provider" in cols) and sub.empty:
        print(f"[WARN] no rows for provider={provider}")
        return
    for var in variables:
        for L in leads:
            mask = np.ones(len(sub), dtype=bool)
            if "variable" in sub.columns:
                mask &= (sub["variable"] == var)
            if "lead_hours" in sub.columns:
                mask &= (sub["lead_hours"] == L)
            ss = sub[mask]
            if ss.empty:
                print(f"[WARN] no rows for {provider}/{var}/{L}h")
                continue
            # aggregate across thresholds: mean FPA per (alpha, method)
            g = (ss.groupby(["alpha_from_dir","method"], as_index=False)["fpa"].mean()
                    .sort_values(["method","alpha_from_dir"]))
            # build figure
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
            alphas = sorted(g["alpha_from_dir"].unique())
            ax.plot(alphas, alphas, color="0.6", ls="--", lw=1.0, label="y=x (target)")
            data_json = {"provider": provider, "variable": var, "lead_hours": int(L), "series": {}}
            for m in METHOD_STYLE:
                gg = g[g["method"] == m]
                if gg.empty: 
                    continue
                xs = gg["alpha_from_dir"].to_numpy()
                ys = gg["fpa"].to_numpy()
                s = METHOD_STYLE[m]
                ax.plot(xs, ys, label=s["label"], color=s["color"], ls=s["ls"], lw=s["lw"], marker=s["marker"], ms=4)
                data_json["series"][m] = {"alpha": xs.tolist(), "fpa": ys.tolist()}
            ax.set_xlabel("Target risk α")
            ax.set_ylabel("Empirical FPA")
            ax.set_title(f"{provider} · {var} · {L}h")
            ax.set_xlim(min(alphas)-0.01, max(alphas)+0.01)
            ax.set_ylim(0, max(0.25, g["fpa"].max()*1.05))
            ax.legend(frameon=False, ncol=1)
            ensure_dir(outdir)
            base = f"F2_validity_{provider}_{var}_L{L}h"
            fig_save_with_json(fig, data_json,
                               os.path.join(outdir, base + ".pdf"),
                               os.path.join(outdir, base + ".json"))
            plt.close(fig)

# ---------- F3: Sharpness frontier (FNA vs IoU at fixed α) ----------
def plot_sharpness_frontier(df: pd.DataFrame, provider: str, alpha: float, variables: List[str], leads: List[int], outdir: str):
    cols = set(df.columns)
    mask = np.ones(len(df), dtype=bool)
    if "provider" in cols:
        mask &= (df["provider"] == provider)
    if "alpha_from_dir" in cols:
        mask &= np.isclose(df["alpha_from_dir"], alpha)
    sub = df[mask].copy()
    if sub.empty:
        print(f"[WARN] no rows for provider={provider} at alpha={alpha}")
        return
    for var in variables:
        for L in leads:
            mask2 = np.ones(len(sub), dtype=bool)
            if "variable" in sub.columns:
                mask2 &= (sub["variable"] == var)
            if "lead_hours" in sub.columns:
                mask2 &= (sub["lead_hours"] == L)
            ss = sub[mask2]
            if ss.empty:
                print(f"[WARN] no rows for {provider}/{var}/{L}h @alpha={alpha}")
                continue
            # average across thresholds for compactness
            g = (ss.groupby(["method"], as_index=False)[["fna","iou"]].mean())
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
            data_json = {"provider": provider, "variable": var, "lead_hours": int(L), "alpha": alpha, "points": {}}
            for m in METHOD_STYLE:
                row = g[g["method"] == m]
                if row.empty: 
                    continue
                x = float(row["fna"].iloc[0]); y = float(row["iou"].iloc[0])
                s = METHOD_STYLE[m]
                ax.scatter(x, y, label=s["label"], color=s["color"], marker=s["marker"], s=50)
                data_json["points"][m] = {"fna": x, "iou": y}
            ax.set_xlabel("Miss (FNA) ↓")
            ax.set_ylabel("Sharpness (IoU) ↑")
            ax.set_title(f"{provider} · {var} · {L}h @ α={alpha}")
            ax.grid(True, ls=":", lw=0.6, color="0.85")
            ax.legend(frameon=False, loc="lower left")
            ensure_dir(outdir)
            base = f"F3_frontier_{provider}_{var}_L{L}h_alpha{alpha:.2f}"
            fig_save_with_json(fig, data_json,
                               os.path.join(outdir, base + ".pdf"),
                               os.path.join(outdir, base + ".json"))
            plt.close(fig)

# ---------- F6: Case-study maps ----------
def plot_case_from_npz(npz_path: str, out_pdf: str, out_json: str, title: str = ""):
    """Render truth mask (fill) + method contours from an example NPZ dumped by fc_run_local_benchmark.py."""
    Z = np.load(npz_path, allow_pickle=True)
    truth_mask = Z["truth_mask"].astype(bool)
    H, W = truth_mask.shape
    # Collect available preds
    panels = [
        ("FieldCert", "pred_crc", "k"),
        ("Global",    "pred_global", "C0"),
        ("Morph-CP",  "pred_morph", "C1"),
        ("Prob-Iso",  "pred_prob", "C3"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.imshow(truth_mask, origin="lower", cmap="Greys", alpha=0.35, interpolation="nearest")
    data_json = {"npz": os.path.basename(npz_path), "contours": []}
    for label, key, color in panels:
        if key not in Z:
            continue
        mask = Z[key].astype(bool)
        # contour via marching squares approximation: draw edges where mask changes
        # we simply plot contour as image edge (mild but clear)
        # For a nicer look, use matplotlib.contour on 0/1 array:
        cs = ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.0, origin="lower")
        data_json["contours"].append({"label": label, "color": color, "level": 0.5})
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title if title else os.path.basename(npz_path))
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    with open(out_json, "w") as f:
        json.dump(data_json, f, indent=2)
    plt.close(fig)
    print(f"[OK] wrote {out_pdf} & {out_json}")

def main():
    ap = argparse.ArgumentParser(description="Generate figures (F2,F3,F6) + JSON data from results.")
    # Where to read summaries (from fc_full_benchmark.py). Use a glob like '/.../full_benchmark/alpha_*/summary.csv'
    ap.add_argument("--summaries_glob", default="/workspace/results/full_benchmark/alpha_*/summary.csv")
    ap.add_argument("--providers", default="ifs_mean,graphcast,neuralgcm,persistence")
    ap.add_argument("--variables", default="10m_wind_speed,total_precipitation_24hr")
    ap.add_argument("--leads", default="24,48,72")
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--outdir", default="/workspace/results/figs")
    # Case-study NPZs (from fc_run_local_benchmark --examples_dir)
    ap.add_argument("--cases_dir", default=None, help="Directory with example_t*.npz files to render F6")
    ap.add_argument("--max_cases", type=int, default=4)
    args = ap.parse_args()

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    leads = [int(x) for x in args.leads.split(",") if x.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    # -------- F2 & F3 from summaries --------
    df = load_summaries(args.summaries_glob)
    figdir = args.outdir; ensure_dir(figdir)

    # F2: validity curves for each provider/var/lead
    for prov in providers:
        plot_validity_curves(df, prov, variables, leads, outdir=figdir)

    # F3: sharpness frontier at each alpha (common main value is 0.10)
    for prov in providers:
        for a in alphas:
            plot_sharpness_frontier(df, prov, a, variables, leads, outdir=figdir)

    # -------- F6 from NPZ examples --------
    if args.cases_dir:
        cand = sorted(glob.glob(os.path.join(args.cases_dir, "example_t*.npz")))
        for i, npz in enumerate(cand[: args.max_cases]):
            base = f"F6_case_{i:02d}"
            plot_case_from_npz(npz,
                               out_pdf=os.path.join(figdir, base + ".pdf"),
                               out_json=os.path.join(figdir, base + ".json"))
    print("[OK] all requested figures generated.")

if __name__ == "__main__":
    main()
