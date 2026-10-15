#!/usr/bin/env Rscript
# Fits eRm::RM/PCM/RSM (CML) to a response CSV and writes item-location
# (and, for PCM/RSM, threshold) estimates to CSV.
#
# Usage: Rscript fit_erm.R <model:slm|pcm|rsm> <responses_csv> <out_prefix>
#
# eRm's coef()/thresholds() report item EASINESS (beta); RaschPy's own
# convention is item DIFFICULTY -- sign-flipped throughout (-coef, or
# thresholds()'s own "Location"/"Threshold N" columns, which are already
# difficulty-signed).

suppressMessages(library(eRm))

args <- commandArgs(trailingOnly = TRUE)
model <- args[1]
responses_csv <- args[2]
out_prefix <- args[3]

data <- as.matrix(read.csv(responses_csv, row.names = 1))

if (model == "slm") {
  # In-R fitting time: the model fit only (no R startup, package
  # loading or file I/O), for a like-for-like comparison with RaschPy.
  t_fit <- proc.time()[["elapsed"]]
  fit <- RM(data)
  writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
             paste0(out_prefix, "_fit_seconds.txt"))
  items <- -coef(fit)
  names(items) <- sub("^beta ", "", names(items))
  write.csv(
    data.frame(item = names(items), location = as.numeric(items)),
    paste0(out_prefix, "_items.csv"), row.names = FALSE,
  )
} else if (model %in% c("pcm", "rsm")) {
  # se=FALSE -- only item locations are needed here, and computing SEs
  # (a Hessian inversion) can fail outright on some datasets even when
  # the point estimates themselves are fine; se=FALSE sidesteps that
  # class of crash entirely (confirmed thresholds()'s own output is
  # identical either way). Genuinely non-convergent point estimates are
  # still caught downstream by run_benchmark.py's implausible-magnitude
  # sanity check, not silently hidden by skipping SEs.
  # In-R fitting time: the model fit only (no R startup, package
  # loading or file I/O), for a like-for-like comparison with RaschPy.
  t_fit <- proc.time()[["elapsed"]]
  fit <- if (model == "pcm") PCM(data, se = FALSE) else RSM(data, se = FALSE)
  writeLines(format(proc.time()[["elapsed"]] - t_fit, digits = 10),
             paste0(out_prefix, "_fit_seconds.txt"))
  th <- thresholds(fit)$threshtable[[1]]
  # th columns: Location, Threshold 1..K -- Location is item difficulty,
  # Threshold N columns are Rasch-Andrich offsets relative to it (already
  # in RaschPy's own centred convention).
  out <- as.data.frame(th)
  out$item <- rownames(th)
  write.csv(out, paste0(out_prefix, "_items_thresholds.csv"), row.names = FALSE)
} else {
  stop(paste0("Unknown model: ", model))
}

cat("done\n")
