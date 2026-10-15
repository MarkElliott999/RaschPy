#!/usr/bin/env Rscript
# Fits mirt::mirt(itemtype="Rasch") (MML, Bock-Aitkin EM + Gauss-Hermite
# quadrature -- a different numerical implementation of MML than TAM's)
# to a response CSV. Writes item-location estimates to CSV.
#
# Usage: Rscript fit_mirt.R <model:slm|pcm|rsm> <responses_csv> <out_prefix>
#
# Dichotomous: coef(fit, simplify=TRUE)$items$d is the negative of item
# difficulty (mirt's 2PL-style intercept convention, a1 fixed at 1 for
# Rasch) -- confirmed against eRm's own raw (pre-flip) easiness beta on
# toy data, same sign/magnitude pattern. Polytomous: coef(...,
# IRTpars=TRUE)$items gives b1..bK step locations (Masters' PCM
# convention); item location is their mean, confirmed against RaschPy's
# own item locations on toy data (same sign pattern).

suppressMessages(library(mirt))

args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
responses_csv <- args[2]
out_prefix <- args[3]

data <- as.matrix(read.csv(responses_csv, row.names = 1))

# RSM data get mirt's rating scale model (itemtype = "rsm": shared b_k,
# item shift c_i, absolute threshold b_k - c_i -- sign checked against
# known generating values, r = 1.000, slope ~1).
# In-R fitting time: the model fit only (no R startup, package
# loading or file I/O), for a like-for-like comparison with RaschPy.
t_fit <- proc.time()[["elapsed"]]
fit <- mirt(data, 1, itemtype = if (model == "rsm") "rsm" else "Rasch", verbose = FALSE)
writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
           paste0(out_prefix, "_fit_seconds.txt"))

if (model == "slm") {
  items_tab <- coef(fit, simplify = TRUE)$items
  locations <- -items_tab[, "d"]
} else {
  if (model == "rsm") {
    items_tab <- coef(fit, simplify = TRUE)$items
    b_cols <- grep("^b[0-9]+$", colnames(items_tab))
    th <- as.matrix(items_tab[, b_cols, drop = FALSE]) - items_tab[, "c"]
  } else {
    items_tab <- coef(fit, simplify = TRUE, IRTpars = TRUE)$items
    b_cols <- grep("^b[0-9]+$", colnames(items_tab))
    th <- as.matrix(items_tab[, b_cols, drop = FALSE])
  }
  locations <- rowMeans(th, na.rm = TRUE)
  thr <- as.data.frame(th)
  colnames(thr) <- paste0("t", seq_len(ncol(th)))
  thr$item <- rownames(items_tab)
  write.csv(thr, paste0(out_prefix, "_thresholds_abs.csv"), row.names = FALSE)
}

write.csv(
  data.frame(item = rownames(items_tab), location = as.numeric(locations)),
  paste0(out_prefix, "_items.csv"), row.names = FALSE,
)

cat("done\n")
