#!/usr/bin/env Rscript
# Fits psychotools::raschmodel/pcmodel/rsmodel (CML) to a response CSV.
# Writes item-location estimates to CSV.
#
# Usage: Rscript fit_psychotools.R <model:slm|pcm|rsm> <responses_csv> <out_prefix>
#
# itempar() is already difficulty-signed and matches RaschPy's own
# convention directly (confirmed against a toy SLM dataset -- values
# essentially identical to eRm's own CML fit on the same data, as
# expected since both implement the same estimator).

suppressMessages(library(psychotools))

args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
responses_csv <- args[2]
out_prefix <- args[3]

data <- as.matrix(read.csv(responses_csv, row.names = 1))

# In-R fitting time: the model fit only (no R startup, package
# loading or file I/O), for a like-for-like comparison with RaschPy.
t_fit <- proc.time()[["elapsed"]]
fit <- if (model == "slm") {
  raschmodel(data)
} else if (model == "pcm") {
  pcmodel(data)
} else if (model == "rsm") {
  rsmodel(data)
} else {
  stop(paste0("Unknown model: ", model))
}
writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
           paste0(out_prefix, "_fit_seconds.txt"))

items <- itempar(fit)
write.csv(
  data.frame(item = names(items), location = as.numeric(items)),
  paste0(out_prefix, "_items.csv"), row.names = FALSE,
)

# Absolute Rasch-Andrich thresholds per item (threshpar type "mode"),
# difficulty-signed -- checked against known generating values on large
# simulated PCM/RSM data (r > 0.999, slope ~1).
if (model %in% c("pcm", "rsm")) {
  tp <- threshpar(fit, type = "mode")
  k <- max(sapply(tp, length))
  th <- t(sapply(tp, function(x) c(as.numeric(x), rep(NA, k - length(x)))))
  out <- as.data.frame(th)
  colnames(out) <- paste0("t", seq_len(k))
  out$item <- names(tp)
  write.csv(out, paste0(out_prefix, "_thresholds_abs.csv"), row.names = FALSE)
}

cat("done\n")
