suppressMessages(library(psychotools))
for (nm in commandArgs(TRUE)) {
  X <- as.matrix(read.csv(paste0(nm, ".csv")))
  p <- pcmodel(X, hessian = TRUE)
  tp <- threshpar(p, type = "mode", ref = NULL)        # uncentred item-category thresholds, origin as fitted
  th <- unlist(coef(tp)); V <- vcov(tp)
  A <- diag(length(th)) - 1 / length(th)               # centre all thresholds to mean 0
  th <- as.vector(A %*% th); se <- sqrt(diag(A %*% V %*% t(A)))
  write.csv(data.frame(label = names(unlist(coef(tp))), psy = th, psy_se = se, psy_ll = as.numeric(logLik(p))),
            paste0(nm, "_psychotools.csv"), row.names = FALSE)
}
cat("done\n")
