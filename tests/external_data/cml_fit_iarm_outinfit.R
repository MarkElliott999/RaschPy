suppressMessages({library(eRm); library(iarm)})
for (nm in c("slm_complete", "pcm_complete")) {
  X <- as.matrix(read.csv(paste0(nm, ".csv")))
  fit <- if (nm == "slm_complete") RM(X) else PCM(X)
  o <- out_infit(fit)
  write.csv(data.frame(item = colnames(X), outfit = o$Outfit, infit = o$Infit, outfit_se = o$Outfit.se, infit_se = o$Infit.se,
                       outfit_p = o$out.pvalue, infit_p = o$in.pvalue), paste0(nm, "_iarm_fit.csv"), row.names = FALSE)
}
cat("done\n")
