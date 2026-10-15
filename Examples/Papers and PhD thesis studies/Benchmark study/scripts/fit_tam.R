#!/usr/bin/env Rscript
# Fits TAM's tam.mml (MML) or tam.jml (JML) to a response CSV, and
# tam.mml.mfr (MML, many-facet) to a (rater,item) long-format rating
# frame for MFRM global. Writes item-location estimates to CSV.
#
# Usage:
#   Rscript fit_tam.R <estimator:mml|jml> <model:slm|pcm|rsm> <responses_csv> <out_prefix>
#   Rscript fit_tam.R mfr global <long_format_csv> <out_prefix>
#
# TAM's fit$item$xsi.item is already difficulty-signed (matches RaschPy's
# own convention) for both tam.mml and tam.jml -- no sign flip needed.

suppressMessages(library(TAM))

args <- commandArgs(trailingOnly = TRUE)
estimator <- args[1]
model <- args[2]
responses_csv <- args[3]
out_prefix <- args[4]

if (estimator %in% c("mml", "jml")) {
  data <- as.matrix(read.csv(responses_csv, row.names = 1))
  # RSM data get a genuine rating scale fit (shared thresholds):
  # irtmodel = "RSM" for MML, the RSM design matrix for JML. SLM/PCM use
  # TAM's default partial credit parameterisation.
  # In-R fitting time: the model fit only (no R startup, package
  # loading or file I/O), for a like-for-like comparison with RaschPy.
  t_fit <- proc.time()[["elapsed"]]
  fit <- if (estimator == "mml") {
    if (model == "rsm") tam.mml(data, irtmodel = "RSM", verbose = FALSE)
    else tam.mml(data, verbose = FALSE)
  } else {
    if (model == "rsm") tam.jml(data, A = designMatrices(modeltype = "RSM", resp = data)$A, verbose = FALSE)
    else tam.jml(data, verbose = FALSE)
  }
  writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
             paste0(out_prefix, "_fit_seconds.txt"))
  if (model == "slm") {
    out <- fit$item[, c("item", "xsi.item")]
    colnames(out) <- c("item", "location")
    write.csv(out, paste0(out_prefix, "_items.csv"), row.names = FALSE)
  } else {
    # AXsi_.Cat k is the cumulative item-category parameter; its first
    # differences are the absolute Rasch-Andrich thresholds (checked
    # against known generating values, r > 0.999, slope ~1). The item
    # location is their mean, as in RaschPy.
    ax <- as.matrix(fit$item[, grep("^AXsi_.Cat", colnames(fit$item)), drop = FALSE])
    th <- ax - cbind(0, ax[, -ncol(ax), drop = FALSE])
    out <- data.frame(item = fit$item$item, location = rowMeans(th, na.rm = TRUE))
    write.csv(out, paste0(out_prefix, "_items.csv"), row.names = FALSE)
    thr <- as.data.frame(th)
    colnames(thr) <- paste0("t", seq_len(ncol(th)))
    thr$item <- fit$item$item
    write.csv(thr, paste0(out_prefix, "_thresholds_abs.csv"), row.names = FALSE)
  }
} else if (estimator == "mfr") {
  # long_format_csv: columns pid (person), rater, item1..itemK scores
  # (wide per item, long per rater) -- see the Python-side fixture
  # generator for the exact layout tam.mml.mfr expects.
  #
  # fit$item$xsi.item is NOT what we want here -- with facets = ~ item +
  # rater + step, TAM internally represents every observed (item, rater)
  # combination as its own "virtual item" for computation, so fit$item
  # has one row per item*rater crossed pair, not per item. The genuine
  # additive decomposition (pure item difficulty, separate from rater
  # severity and the shared step/threshold structure) lives in fit$xsi
  # instead -- confirmed empirically against a toy MFRM_Sim dataset
  # before relying on it here, not assumed from documentation alone.
  long <- read.csv(responses_csv)
  item_cols <- setdiff(colnames(long), c("pid", "rater"))
  resp <- long[, item_cols, drop = FALSE]
  facets <- data.frame(rater = long$rater)
  formulaA <- ~ item + rater + step
  # In-R fitting time (excludes R startup, package loading and file I/O),
  # so the timing comparison with RaschPy is like for like.
  t_fit <- proc.time()[["elapsed"]]
  fit <- tam.mml.mfr(
    resp = resp, facets = facets, pid = long$pid,
    formulaA = formulaA, verbose = FALSE,
  )
  writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
             paste0(out_prefix, "_fit_seconds.txt"))
  xsi <- fit$xsi
  is_item_row <- rownames(xsi) %in% item_cols
  out <- data.frame(item = rownames(xsi)[is_item_row], location = xsi$xsi[is_item_row])
  write.csv(out, paste0(out_prefix, "_items.csv"), row.names = FALSE)
  # Shared (centred) Rasch-Andrich thresholds: TAM's step parameters are
  # tau_1..tau_{K-1}, with tau_K = -sum (checked against known generating
  # values). A category TAM drops as unused gives fewer steps, which the
  # Python side treats as not comparable.
  steps <- xsi[grepl("^step[0-9]+$", rownames(xsi)), "xsi"]
  tau <- c(steps, -sum(steps))
  write.csv(data.frame(k = seq_along(tau), tau = tau),
            paste0(out_prefix, "_tau.csv"), row.names = FALSE)
} else {
  stop(paste0("Unknown estimator: ", estimator))
}

cat("done\n")
