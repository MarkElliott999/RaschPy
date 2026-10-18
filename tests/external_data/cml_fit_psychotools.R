suppressMessages({library(psychotools); library(eRm)})
out <- list()
for (nm in commandArgs(TRUE)) {
  X <- as.matrix(read.csv(paste0(nm, ".csv")))
  p <- raschmodel(X, hessian = TRUE)
  d <- c(0, coef(p)); d <- d - mean(d)               # difficulties, centred
  V <- vcov(p); I <- ncol(X)
  A <- rbind(0, diag(I - 1)); A <- A - matrix(colMeans(A), I, I - 1, byrow = TRUE)
  se <- sqrt(diag(A %*% V %*% t(A)))
  if (!anyNA(X)) { e <- RM(X); de <- -e$betapar; de <- de - mean(de); ell <- e$loglik } else { de <- NA; ell <- NA }
  write.csv(data.frame(item = colnames(X), psy = d, psy_se = se, erm = de,
                       psy_ll = as.numeric(logLik(p)), erm_ll = ell), paste0(nm, "_r.csv"), row.names = FALSE)
}
cat("done\n")
