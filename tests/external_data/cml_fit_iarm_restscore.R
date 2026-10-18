suppressMessages({library(eRm); library(iarm)})
for (nm in c("slm_complete", "pcm_complete")) {
  X <- as.matrix(read.csv(paste0(nm, ".csv")))
  fit <- if (nm == "slm_complete") RM(X) else PCM(X)
  o <- item_restscore(fit, p.adj = "BH")
  v <- function(j) as.numeric(as.character(o[, j]))
  write.csv(data.frame(item = colnames(X), observed = v(1), expected = v(2), se = v(3), p = v(4), padj = v(5)),
            paste0(nm, "_iarm_rest.csv"), row.names = FALSE)
}
cat("done\n")
