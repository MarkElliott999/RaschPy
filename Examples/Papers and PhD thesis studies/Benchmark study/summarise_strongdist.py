"""Stronger person-distribution departures: all packages, 10 reps per condition."""
import pandas as pd, numpy as np, os
pd.set_option("display.width", 250)
for m, thr in (("slm", None), ("pcm", "thr_RMSE"), ("mfrm", "tau_RMSE"), ("rsm", "tau_RMSE")):
    f = f"results_strongdist_{m}.csv"
    if not os.path.exists(f): continue
    d = pd.read_csv(f); d["failed"] = ~d.converged.astype(bool) | d.RMSE.isna()
    print(f"\n===== {m.upper()} =====")
    for cond, g in d.groupby("condition"):
        rp = g[g.algorithm == "raschpy"]
        rows = []
        for alg, h in g.groupby("algorithm"):
            ok = h[~h.failed]
            r = dict(algorithm=alg, fail=f"{h.failed.sum()}/{len(h)}", item_med=ok.RMSE.median(), item_p90=ok.RMSE.quantile(.9), item_slope=ok.OLS_slope.median())
            if thr:
                r.update(thr_med=ok[thr].median(), thr_p90=ok[thr].quantile(.9), thr_slope=ok[thr.replace("RMSE", "OLS_slope")].median())
            if alg != "raschpy":
                j = ok.merge(rp, on=["condition", "rep"], suffixes=("_o", "_rp"))
                r["rp/alg item"] = np.nanmedian(j.RMSE_rp / j.RMSE_o)
                if thr: r["rp/alg thr"] = np.nanmedian((j[thr + "_rp"] / j[thr + "_o"]).replace([np.inf, -np.inf], np.nan))
            r["fit_s"] = h.fit_seconds.mean()
            rows.append(r)
        print(f"-- {cond}"); print(pd.DataFrame(rows).round(3).to_string(index=False))
