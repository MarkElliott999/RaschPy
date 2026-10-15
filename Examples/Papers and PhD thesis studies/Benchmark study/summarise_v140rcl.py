"""RaschPy v1.4.0 (RCL-CPAT default; results_v1.4.0rcl_*) vs RaschPy 3 Oct (results_thresholds_*) vs external packages (3 Oct rows, reuse validated by spotcheck_v1.4.0rcl_*)."""
import pandas as pd, numpy as np
pd.set_option("display.width", 250)
rows = []
for m in ("slm", "rsm", "pcm", "mfrm"):
    old = pd.read_csv(f"results_thresholds_{m}.csv"); new = pd.read_csv(f"results_v1.4.0rcl_{m}.csv")
    new = new[new.algorithm == "raschpy"].assign(algorithm="raschpy v1.4.0")
    old.loc[old.algorithm == "raschpy", "algorithm"] = "raschpy 3 Oct"
    d = pd.concat([old, new]); d["failed"] = ~d.converged.astype(bool) | d.RMSE.isna()
    thr = "thr_RMSE" if m == "pcm" else ("tau_RMSE" if m in ("rsm", "mfrm") else None)
    for alg, g in d.groupby("algorithm"):
        ok = g[~g.failed]
        r = dict(model=m, algorithm=alg, fail_pct=100 * g.failed.mean(), item_RMSE_med=ok.RMSE.median(), item_RMSE_p90=ok.RMSE.quantile(.9),
                 item_slope_med=ok.OLS_slope.median(), fit_s_mean=g.fit_seconds.mean() if "fit_seconds" in g else np.nan)
        if thr:
            r.update(thr_RMSE_med=ok[thr].median(), thr_RMSE_p90=ok[thr].quantile(.9),
                     thr_slope_med=ok[thr.replace("RMSE", "OLS_slope")].median())
        rows.append(r)
t = pd.DataFrame(rows)
print(t.round(3).to_string(index=False))
# paired RaschPy v1.4.0 vs 3 Oct, same datasets
print("\nPaired, same datasets: v1.4.0 vs 3 Oct RaschPy (share of datasets where v1.4.0 is better by > 0.01 / worse by > 0.01)")
for m in ("slm", "rsm", "pcm", "mfrm"):
    old = pd.read_csv(f"results_thresholds_{m}.csv"); new = pd.read_csv(f"results_v1.4.0rcl_{m}.csv")
    j = new[new.algorithm == "raschpy"].merge(old[old.algorithm == "raschpy"], on=["condition", "rep"], suffixes=("_new", "_old"))
    out = []
    for c in ("RMSE", "thr_RMSE" if m == "pcm" else "tau_RMSE"):
        if c + "_new" in j and j[c + "_new"].notna().any():
            dlt = j[c + "_new"] - j[c + "_old"]
            out.append(f"{c}: better {100*(dlt < -0.01).mean():.0f}% / worse {100*(dlt > 0.01).mean():.0f}% (median ratio {np.nanmedian(j[c+'_new']/j[c+'_old']):.3f})")
    print(f"  {m}: n={len(j)}; " + "; ".join(out))
