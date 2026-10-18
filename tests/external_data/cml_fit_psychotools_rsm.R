suppressMessages(library(psychotools))
for (nm in commandArgs(TRUE)) {
  X <- as.matrix(read.csv(paste0(nm, ".csv")))
  p <- rsmodel(X, hessian = TRUE)
  ip <- itempar(p)                                        # item locations, sum zero
  tp <- threshpar(p, type = "mode", relative = TRUE)      # shared thresholds relative to item location
  K <- length(coef(tp)) / ncol(X); tau <- coef(tp)[1:K]; Vt <- vcov(tp)[1:K, 1:K]
  write.csv(data.frame(kind = c(rep("item", length(coef(ip))), rep("threshold", length(tau))),
                       psy = c(coef(ip), tau), psy_se = c(sqrt(diag(vcov(ip))), sqrt(diag(Vt))),
                       psy_ll = as.numeric(logLik(p)), code = p$code),
            paste0(nm, "_psychotools.csv"), row.names = FALSE)
}
cat("done\n")
