#!/usr/bin/env Rscript
# Fits the pairwise package (Choppin 1968/1985 conditional pairwise
# algorithm) to a response CSV. Writes item-location (and, for PCM/RSM,
# threshold) estimates to CSV.
#
# Usage: Rscript fit_pairwise.R <model:slm|pcm|rsm> <responses_csv> <max_score_csv_list> <out_prefix>
#
# max_score_csv_list is a comma-separated per-item max_score list (PCM
# allows a different max_score per item -- the mixed-K condition needs
# this to be a real per-item vector, not one scalar repeated across
# every item; RSM/SLM just get the same value repeated by the caller).
#
# pairwise's fit$sigma / fit$threshold are already difficulty-signed and
# zero-sum by default (zerosum=TRUE), matching RaschPy's own convention
# directly -- no sign flip needed. `m` (categories, not max_score) is
# max_score + 1 per item, matching PCM/RSM's own polytomous handling (a
# per-item widened-Choppin fit, not a rating-scale-constrained one --
# see the benchmark scope notes on this distinction).

suppressMessages(library(pairwise))

args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
responses_csv <- args[2]
max_score_list <- as.integer(strsplit(args[3], ",")[[1]])
out_prefix <- args[4]

data <- as.matrix(read.csv(responses_csv, row.names = 1))

if (model == "slm") {
  # In-R fitting time: the model fit only (no R startup, package
  # loading or file I/O), for a like-for-like comparison with RaschPy.
  t_fit <- proc.time()[["elapsed"]]
  fit <- pair(data)
  writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
             paste0(out_prefix, "_fit_seconds.txt"))
  items <- fit$sigma
  write.csv(
    data.frame(item = names(items), location = as.numeric(items)),
    paste0(out_prefix, "_items.csv"), row.names = FALSE,
  )
} else if (model %in% c("pcm", "rsm")) {
  # In-R fitting time: the model fit only (no R startup, package
  # loading or file I/O), for a like-for-like comparison with RaschPy.
  t_fit <- proc.time()[["elapsed"]]
  fit <- pair(data, m = max_score_list + 1)
  writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
             paste0(out_prefix, "_fit_seconds.txt"))
  items <- fit$sigma
  out <- data.frame(item = names(items), location = as.numeric(items))
  write.csv(out, paste0(out_prefix, "_items.csv"), row.names = FALSE)

  th <- as.data.frame(fit$threshold)
  th$item <- rownames(fit$threshold)
  write.csv(th, paste0(out_prefix, "_thresholds.csv"), row.names = FALSE)
} else {
  stop(paste0("Unknown model: ", model))
}

cat("done\n")
