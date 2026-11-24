# Manual changes (not yet in Overleaf)

Each entry: the section, then the new or changed text. Where a line replaces an earlier one, the old text follows under *Was*.


## slm.tex

### calibrate_slm (TO DRAFT, 2026-10-28)
Default estimation is now `estimation=None`, resolving to `'cp-full-rcl'` (or `'pair'` when `method` is `'cos'`, `'ls'` or `'evm'`). Estimates unchanged (the Bradley-Terry likelihood, which PAIR with method='log-lik' maximises); estimation reads 'cp-full-rcl'. Exact .tex text to be drafted from the current calibrate section.


### std_errors_slm (line 493)

```latex
With \textcolor{gray}{\texttt{se\_\allowbreak method=None}} (default), standard errors are analytic (sandwich) for estimates from \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} and bootstrapped for \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}; an explicit \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }} with those methods raises a \textcolor{gray}{\texttt{ValueError}}.
```

### std_errors_slm (line 497)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None,}}\\
```
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~seed=None, se\_\allowbreak method=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001,}}\\
```
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~seed=None, se\_\allowbreak method=\textquotesingle sandwich\textquotesingle )}}\\
```

### std_errors_slm (line 518)

```latex
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of estimates from pairwise reciprocal matrix \citep{Elliott2022a}. Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=None}}, the value of the object's last \textcolor{gray}{\texttt{calibrate}} call (or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} if the object has not yet been calibrated) \\
```
*Was:*
```latex
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of estimates from pairwise reciprocal matrix \citep{Elliott2022a}. Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} \\
```

### std_errors_slm (line 526)

```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }} or \textcolor{gray}{\texttt{None}} (see Description). Default is \textcolor{gray}{\texttt{se\_\allowbreak method=None}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}. Default is \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }}.\\
```

### calibrate_anchor_slm (line 662)

```latex
To anchor without outlier trimming, taking a plain mean shift over all shared items, with no plot:
```
*Was:*
```latex
To anchor without outlier trimming, taking a plain mean shift over all common items, with no plot:
```

### global_fit_slm (line 1212)

```latex
When the two groups respond to different items, each group is calibrated on the items its persons responded to, and parameters are matched by name: those of items only one group responded to are not compared, and are listed in \textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak excluded}}. A \textcolor{gray}{\texttt{ValueError}} is raised if no parameter is estimable in both groups.\\
```
```latex
Each group is calibrated with the object's last \textcolor{gray}{\texttt{calibrate}} settings. If those use \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, the Wald test uses bootstrap covariances (\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} resamples per group) instead of the sandwich, and \textcolor{gray}{\texttt{se\_\allowbreak correction}} does not apply. \textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak summary}} records which covariance was used.
```

### global_fit_slm (line 1218)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle ,}}\\
```
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle )}}\\
```

### global_fit_slm (line 1246)

```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
```

### dif_test_slm (line 1525)

```latex
Each group is calibrated on the items its persons responded to, with the object's last \textcolor{gray}{\texttt{calibrate}} settings unless others are passed, and only items shared by both groups are compared. A focal group that shares no items with the reference group (possible with more than two groups) is left out with a \textcolor{gray}{\texttt{UserWarning}} and recorded in \textcolor{gray}{\texttt{self.dif\_\allowbreak excluded\_\allowbreak groups}}.
```

### dif_test_slm (line 1534)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None, robust=False, method=None,}}\\
```
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None, robust=False, method=\textquotesingle log-lik\textquotesingle ,}}\\
```
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### dif_test_slm (line 1594)

```latex
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration arguments passed to the group models -- see Section \ref{calibrate_slm}. Default is \textcolor{gray}{\texttt{None}} for each: the value of the object's last \textcolor{gray}{\texttt{calibrate}} call (or \textcolor{gray}{\texttt{calibrate}}'s default if the object has not yet been calibrated).\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration arguments passed to the group models -- see Section \ref{calibrate_slm}.\\
```

## pcm.tex

### calibrate_pcm (TO DRAFT, 2026-10-28)
Default estimation is now `estimation=None`, resolving to `'cp-full-rcl'` (or `'pair'` when `method` is `'cos'`, `'ls'` or `'evm'`). Estimates change from PAIR to the full-pattern pairwise conditional likelihood with its ridge; unused_categories defaults to 'ridge'; estimation='pair' gives the previous estimates. Exact .tex text to be drafted from the current calibrate section.


### std_errors_pcm (line 634)

```latex
With \textcolor{gray}{\texttt{se\_\allowbreak method=None}} (default), standard errors are analytic (sandwich) for estimates from \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} and bootstrapped for \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}; an explicit \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }} with those methods raises a \textcolor{gray}{\texttt{ValueError}}.
```

### std_errors_pcm (line 638)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None,}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001,}}\\
```

### std_errors_pcm (line 659)

```latex
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of central item location estimates from pairwise reciprocal matrix \citep{Elliott2022a}. Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=None}}, the value of the object's last \textcolor{gray}{\texttt{calibrate}} call (or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} if the object has not yet been calibrated) \\
```
*Was:*
```latex
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of central item location estimates from pairwise reciprocal matrix \citep{Elliott2022a}. Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} \\
```

### global_fit_pcm (line 1644)

```latex
When the two groups respond to different items, each group is calibrated on the items its persons responded to, and parameters are matched by name: those of items only one group responded to are not compared, and are listed in \textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak excluded}}. A \textcolor{gray}{\texttt{ValueError}} is raised if no parameter is estimable in both groups.\\
```
```latex
Each group is calibrated with the object's last \textcolor{gray}{\texttt{calibrate}} settings. If those use \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, the Wald test uses bootstrap covariances (\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} resamples per group) instead of the sandwich, and \textcolor{gray}{\texttt{se\_\allowbreak correction}} does not apply. \textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak summary}} records which covariance was used.
```
```latex
Under \textcolor{gray}{\texttt{method=\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} the test is liberal. For a test with a nominal false-positive rate, re-run with \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }}: \textcolor{gray}{\texttt{pcm.calibrate(method=\textquotesingle log-lik\textquotesingle )}}, then \textcolor{gray}{\texttt{pcm.global\_\allowbreak fit(...)}}, or use the per-item tests of \textcolor{gray}{\texttt{pcm.dif\_\allowbreak test()}}.
```

### global_fit_pcm (line 1652)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle ,}}\\
```
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle )}}\\
```

### global_fit_pcm (line 1680)

```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
```

### dif_test_pcm (line 1962)

```latex
The omnibus test can take one of two scopes. With \textcolor{gray}{\texttt{omnibus\_\allowbreak scope=\textquotesingle item\textquotesingle }}, it tests uniform DIF only: each item's category structure is held at the pooled model's values and only a single location shift per item is estimated for each group, with $ df = k - 1 $, where $ k $ is the number of shared items. With \textcolor{gray}{\texttt{omnibus\_\allowbreak scope=\textquotesingle full\textquotesingle }}, it tests uniform DIF and differential category functioning together, with each item's thresholds also free in each group, and $ df = (k - 1) + \sum_i (m_i - 1) $, where $ m_i $ is the maximum score of item $ i $. The result is a single joint test; \textcolor{gray}{\texttt{self.dif\_\allowbreak table}} and \textcolor{gray}{\texttt{self.threshold\_\allowbreak dif\_\allowbreak table}} give the item-by-item results.
```
```latex
Each group is calibrated on the items its persons responded to, with the object's last \textcolor{gray}{\texttt{calibrate}} settings unless others are passed, and only items shared by both groups are compared. A focal group that shares no items with the reference group (possible with more than two groups) is left out with a \textcolor{gray}{\texttt{UserWarning}} and recorded in \textcolor{gray}{\texttt{self.dif\_\allowbreak excluded\_\allowbreak groups}}.
```
*Was:*
```latex
The omnibus test can take one of two scopes. With \textcolor{gray}{\texttt{omnibus\_\allowbreak scope=\textquotesingle item\textquotesingle }}, it tests uniform DIF only: each item's category structure is held at the pooled model's values and only a single location shift per item is estimated for each group, with $ df = k - 1 $, where $ k $ is the number of common items. With \textcolor{gray}{\texttt{omnibus\_\allowbreak scope=\textquotesingle full\textquotesingle }}, it tests uniform DIF and differential category functioning together, with each item's thresholds also free in each group, and $ df = (k - 1) + \sum_i (m_i - 1) $, where $ m_i $ is the maximum score of item $ i $. The result is a single joint test; \textcolor{gray}{\texttt{self.dif\_\allowbreak table}} and \textcolor{gray}{\texttt{self.threshold\_\allowbreak dif\_\allowbreak table}} give the item-by-item results.
```

### dif_test_pcm (line 1974)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### dif_test_pcm (line 2034)

```latex
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration arguments passed to the group models -- see Section \ref{calibrate_pcm}. Default is \textcolor{gray}{\texttt{None}} for each: the value of the object's last \textcolor{gray}{\texttt{calibrate}} call (or \textcolor{gray}{\texttt{calibrate}}'s default if the object has not yet been calibrated).\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration arguments passed to the group models -- see Section \ref{calibrate_pcm}.\\
```

## rsm.tex


### std_errors_rsm (line 528)

```latex
With \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} and \textcolor{gray}{\texttt{method}} set to \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, \textcolor{gray}{\texttt{se\_\allowbreak method=None}} resolves to the bootstrap, and an explicit \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }} raises a \textcolor{gray}{\texttt{ValueError}}.
```

### std_errors_rsm (line 532)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None,}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None,}}\\
```

### std_errors_rsm (line 553)

```latex
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of central item location estimates from pairwise reciprocal matrix \citep{Elliott2022a}. Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=None}}, the value of the object's last \textcolor{gray}{\texttt{calibrate}} call (or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} if the object has not yet been calibrated) \\
```
*Was:*
```latex
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of central item location estimates from pairwise reciprocal matrix \citep{Elliott2022a}. Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} \\
```

### global_fit_rsm (line 1530)

```latex
When the two groups respond to different items, each group is calibrated on the items its persons responded to, and parameters are matched by name: those of items only one group responded to are not compared, and are listed in \textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak excluded}}. A \textcolor{gray}{\texttt{ValueError}} is raised if no parameter is estimable in both groups.\\
```
```latex
Each group is calibrated with the object's last \textcolor{gray}{\texttt{calibrate}} settings. If those use \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, the Wald test uses bootstrap covariances (\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} resamples per group) instead of the sandwich, and \textcolor{gray}{\texttt{se\_\allowbreak correction}} does not apply. \textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak summary}} records which covariance was used.
```

### global_fit_rsm (line 1536)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle ,}}\\
```
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle )}}\\
```

### global_fit_rsm (line 1564)

```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
```

### dif_test_rsm (line 1847)

```latex
Each group is calibrated on the items its persons responded to, with the object's last \textcolor{gray}{\texttt{calibrate}} settings unless others are passed, and only items shared by both groups are compared. A focal group that shares no items with the reference group (possible with more than two groups) is left out with a \textcolor{gray}{\texttt{UserWarning}} and recorded in \textcolor{gray}{\texttt{self.dif\_\allowbreak excluded\_\allowbreak groups}}. The omnibus test is mildly liberal when the groups share few items; for a test with a nominal false-positive rate, use the per-item Wald tests in \textcolor{gray}{\texttt{self.dif\_\allowbreak table}} (computed by the same call).
```

### dif_test_rsm (line 1857)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### dif_test_rsm (line 1915)

```latex
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration arguments passed to the group models -- see Section \ref{calibrate_rsm}. Default is \textcolor{gray}{\texttt{None}} for each: the value of the object's last \textcolor{gray}{\texttt{calibrate}} call (or \textcolor{gray}{\texttt{calibrate}}'s default if the object has not yet been calibrated).\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration arguments passed to the group models -- see Section \ref{calibrate_rsm}.\\
```

## mfrm.tex


### calibrate_anchor_mfrm (line 1518)

```latex
Anchors a calibration to a defined set of `gold standard' anchor facet elements (e.g.\ raters), who provide a frame of reference and for whom mean facet effect is defined as zero (and, for stretch parameters with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, mean stretch as 1); other facet elements' facet effects shifted accordingly. For extended facet element representations, this process has a knock-on effect on the central item location and/or Rasch-Andrich threshold estimates. Full details of the anchoring procedure can be found in \citet{Elliott2022b}. For \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}} calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, item locations and thresholds are rescaled accordingly. For \textcolor{gray}{\texttt{matrix}}, only the item and threshold parts of the anchors' mean facet effect are moved into the item locations and thresholds; the item-by-threshold remainder stays in the facet effects.
```
*Was:*
```latex
Anchors a calibration to a defined set of `gold standard' anchor facet elements (e.g.\ raters), who provide a frame of reference and for whom mean facet effect is defined as zero (and, for stretch parameters with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, mean stretch as 1); other facet elements' facet effects shifted accordingly. For extended facet element representations, this process has a knock-on effect on the central item location and/or Rasch-Andrich threshold estimates. Full details of the anchoring procedure can be found in \citet{Elliott2022b}. For \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}} calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, item locations and thresholds are rescaled accordingly.
```

### dif_test_mfrm (line 3924)

```latex
When the two groups are scored by different facet elements, each group is anchored on the facet elements shared by both groups (\textcolor{gray}{\texttt{calibrate\_\allowbreak anchor}}) before item locations and thresholds are compared. If the groups share no facet elements, only \textcolor{gray}{\texttt{model=\textquotesingle global\textquotesingle }} can be tested.
```
```latex
When the groups are scored by different facet elements, a \textcolor{gray}{\texttt{UserWarning}} is raised for the cases where null simulations found the per-item tests liberal, naming the alternative: under \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}, use \textcolor{gray}{\texttt{model=\textquotesingle bivector\textquotesingle }} if it fits the data; with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} under \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, re-run with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}.
```
```latex
Each group is calibrated on the items its persons responded to, and only items shared by both groups are compared (for \textcolor{gray}{\texttt{matrix}}, the thresholds are averaged over the shared items). A focal group that shares no items, or (except under \textcolor{gray}{\texttt{model=\textquotesingle global\textquotesingle }}) no facet elements, with the reference group (possible with more than two groups) is left out with a \textcolor{gray}{\texttt{UserWarning}} and recorded in \textcolor{gray}{\texttt{self.dif\_\allowbreak excluded\_\allowbreak groups}}.
```

### dif_test_mfrm (line 3954)

```latex
The omnibus test, \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}}, tests items only: facet effects are held at their full-data values in both groups and items/thresholds estimated separately for each group ($ df = $ shared items $ - 1 $). The test is liberal when the groups respond to different items or, with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}, are scored by different facet elements; a \textcolor{gray}{\texttt{UserWarning}} is raised, and the per-item tests in \textcolor{gray}{\texttt{self.dif\_\allowbreak table}} should be used instead. The joint test of items and facet effects, \textcolor{gray}{\texttt{self.df\_\allowbreak omnibus\_\allowbreak table}}, is withheld (\textcolor{gray}{\texttt{None}}) until it can be checked with conditional maximum likelihood. \textcolor{gray}{\texttt{test=\textquotesingle lr\textquotesingle /\textquotesingle both\textquotesingle }} is mildly liberal; \textcolor{gray}{\texttt{test=\textquotesingle wald\textquotesingle }} (the default) is preferred.
```
*Was:*
```latex
The omnibus test comes in two forms. \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}} tests items only: facet effects are held at their full-data values in both groups and items/thresholds estimated separately for each group ($ df = $ common items $ - 1 $). \textcolor{gray}{\texttt{self.df\_\allowbreak omnibus\_\allowbreak table}} tests items and facet effects together: items, thresholds and facet effects are all estimated separately for each group (same $ df $). \textcolor{gray}{\texttt{test=\textquotesingle lr\textquotesingle /\textquotesingle both\textquotesingle }} is mildly liberal; \textcolor{gray}{\texttt{test=\textquotesingle wald\textquotesingle }} (the default) is preferred.
```

### dif_test_mfrm (line 3983)

```latex
\textcolor{gray}{\texttt{selection\_\allowbreak method}} & String, one of \textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle none\textquotesingle }}: how the item-location scale is purified before testing. \textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}: sequential Wald tests of each item's difference, using \textcolor{gray}{\texttt{wald\_\allowbreak alpha}} and \textcolor{gray}{\texttt{min\_\allowbreak anchors}}. \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }}: excludes outlying items by the modified z-score procedure \citep{IglewiczHoaglin1993}, using \textcolor{gray}{\texttt{corr\_\allowbreak tol}}, \textcolor{gray}{\texttt{sd\_\allowbreak ratio\_\allowbreak tol}} and \textcolor{gray}{\texttt{min\_\allowbreak anchors}}. \textcolor{gray}{\texttt{\textquotesingle none\textquotesingle }}: mean shift over all shared items. Default is \textcolor{gray}{\texttt{selection\_\allowbreak method=\textquotesingle wald\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{selection\_\allowbreak method}} & String, one of \textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle none\textquotesingle }}: how the item-location scale is purified before testing. \textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}: sequential Wald tests of each item's difference, using \textcolor{gray}{\texttt{wald\_\allowbreak alpha}} and \textcolor{gray}{\texttt{min\_\allowbreak anchors}}. \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }}: excludes outlying items by the modified z-score procedure \citep{IglewiczHoaglin1993}, using \textcolor{gray}{\texttt{corr\_\allowbreak tol}}, \textcolor{gray}{\texttt{sd\_\allowbreak ratio\_\allowbreak tol}} and \textcolor{gray}{\texttt{min\_\allowbreak anchors}}. \textcolor{gray}{\texttt{\textquotesingle none\textquotesingle }}: mean shift over all common items. Default is \textcolor{gray}{\texttt{selection\_\allowbreak method=\textquotesingle wald\textquotesingle }}.\\
```

### dif_test_mfrm (line 3995)

```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}}, also computes the omnibus likelihood-ratio test for each focal group (\textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}}). Default is \textcolor{gray}{\texttt{omnibus=True}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}}, also computes the two omnibus likelihood-ratio tests for each focal group (\textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}} and \textcolor{gray}{\texttt{self.df\_\allowbreak omnibus\_\allowbreak table}}). Default is \textcolor{gray}{\texttt{omnibus=True}}.\\
```

### dif_test_mfrm (line 4082)

```latex
\textcolor{gray}{\texttt{self.df\_\allowbreak omnibus\_\allowbreak table}} & \textcolor{gray}{\texttt{None}}: the joint test of items and facet effects is withheld until it can be checked with conditional maximum likelihood.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.df\_\allowbreak omnibus\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: as \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}}, for the joint test of items and facet effects. \textcolor{gray}{\texttt{None}} if \textcolor{gray}{\texttt{omnibus=False}}.\\
```

### drf_test_mfrm (line 4111)

```latex
Each group is calibrated on the items its persons responded to, and only items shared by both groups are tested. Parameters that depend on which facet elements or items they are relative to (the \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }} threshold marginal, and the parameters of \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }}) are first anchored on the facet elements and items shared by both groups. A focal group that shares fewer than two facet elements, or (where the model needs them) no items, with the reference group (possible with more than two groups) is left out with a \textcolor{gray}{\texttt{UserWarning}} and recorded in \textcolor{gray}{\texttt{self.drf\_\allowbreak excluded\_\allowbreak groups}}.
```

---
*Not from this session:* `mfrm.tex` also differs from the 15 October backup in the `std_errors` description (sandwich fallback wording; NaN standard errors for facet elements that used only the lowest or highest category) and in `anchor_std_errors` (the same NaN sentence). Those edits were already in the file before this session and are not listed above.


# v1.5.0: conditional methods (2026-10-19)

## slm.tex

### calibrate_slm (line 447)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~robust=False, estimation=\textquotesingle pair\textquotesingle , solver=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~robust=False)}}\\
```

### calibrate_slm (line 444)

```latex
With \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, the item locations are conditional maximum likelihood (CML) estimates \citep{Andersen1970}, maximising the likelihood of the responses given each person's total score over the items they responded to, starting from the PAIR estimates. Items without CML estimates are removed from the data, with a \textcolor{gray}{\texttt{UserWarning}}.\\

\newpage
\textbf{Usage}
```
*Was:*
```latex
\newpage
\textbf{Usage}
```

### calibrate_slm (line 469)

```latex
{\texttt{FutureWarning}}. It will be removed in a future release. Default is \textcolor{gray}{\texttt{robust=False}}.\\
\midrule
\textcolor{gray}{\texttt{estimation}} & String: \textcolor{gray}{\texttt{\textquotesingle pair\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} (see Description). Default is \textcolor{gray}{\texttt{estimation=\textquotesingle pair\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{solver}} & String or \textcolor{gray}{\texttt{None}}: CML solver, \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle newton\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle lbfgs\textquotesingle }}; all reach the same estimates. \textcolor{gray}{\texttt{None}} resolves to \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }}. Ignored unless \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}. Default is \textcolor{gray}{\texttt{solver=None}}.\\
\end{longtable}
```
*Was:*
```latex
{\texttt{FutureWarning}}. It will be removed in a future release. Default is \textcolor{gray}{\texttt{robust=False}}.\\
\end{longtable}
```

### calibrate_slm (line 478)

```latex
Attribute \textcolor{gray}{\texttt{self.constant}}: the additive smoothing constant actually used, after resolving \textcolor{gray}{\texttt{None}}.\\
Attribute \textcolor{gray}{\texttt{self.estimation}}: the estimation method used.\\
Attribute \textcolor{gray}{\texttt{self.convergence}}: Dictionary or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, iterations, convergence, solver, conditional log-likelihood and the items removed (\textcolor{gray}{\texttt{\textquotesingle dropped\_\allowbreak items\textquotesingle }}, with the reason).\\
Attribute \textcolor{gray}{\texttt{self.cml\_\allowbreak dropped\_\allowbreak responses}}: pandas dataframe: with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, the responses of the items removed.\\
```
*Was:*
```latex
Attribute \textcolor{gray}{\texttt{self.constant}}: the additive smoothing constant actually used, after resolving \textcolor{gray}{\texttt{None}}.\\
```

### calibrate_slm (line 495)

```latex
To produce conditional maximum likelihood estimates:

\textcolor{gray}{\texttt{self.calibrate(estimation=\textquotesingle cml\textquotesingle )}}
```

### std_errors_slm (line 508)

```latex
For estimates from \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, \textcolor{gray}{\texttt{se\_\allowbreak method=None}} gives sandwich standard errors of the conditional likelihood, and \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle hessian\textquotesingle }} the inverse of its information matrix; the bootstrap refits with CML.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### std_errors_slm (line 541)

```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle hessian\textquotesingle }} (CML estimates only), \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```
*Was:*
```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```

### std_errors_slm (line 554)

```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle hessian\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```

### std_errors_slm (line 556)

```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak covariance}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: sandwich or Hessian covariance matrix of the item locations (sandwich or Hessian only; \textcolor{gray}{\texttt{None}} for the bootstrap).
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak covariance}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: sandwich covariance matrix of the item locations (sandwich only; \textcolor{gray}{\texttt{None}} for the bootstrap).
```

### calibrate_anchor_slm (line 598)

```latex
lik\_\allowbreak tol=0.000001, seed=None, fix\_\allowbreak anchors=None)}}\\
```
*Was:*
```latex
lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### calibrate_anchor_slm (line 593)

```latex
For CML estimates (\textcolor{gray}{\texttt{calibrate(estimation=\textquotesingle cml\textquotesingle )}}), the anchor items that keep their supplied values (see \textcolor{gray}{\texttt{overwrite\_\allowbreak anchors}}) are by default held at those values in a CML fit, and every other parameter is estimated given them (\textcolor{gray}{\texttt{fix\_\allowbreak anchors}}).\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### calibrate_anchor_slm (line 614)

```latex
(re-)runs \textcolor{gray}{\texttt{calibrate}}, with the estimation method and solver of the latest calibration, before anchoring.
```
*Was:*
```latex
(re-)runs \textcolor{gray}{\texttt{calibrate}} before anchoring.
```

### calibrate_anchor_slm (line 639)

```latex
}} only). \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\midrule
\textcolor{gray}{\texttt{fix\_\allowbreak anchors}} & Boolean or \textcolor{gray}{\texttt{None}}: if \textcolor{gray}{\texttt{True}}, fits CML with the kept anchor items held at their supplied values (see Description); if \textcolor{gray}{\texttt{False}}, applies the translation. \textcolor{gray}{\texttt{None}} resolves to \textcolor{gray}{\texttt{True}} for CML estimates, else \textcolor{gray}{\texttt{False}}; \textcolor{gray}{\texttt{True}} with other estimation methods raises a \textcolor{gray}{\texttt{ValueError}}. Default is \textcolor{gray}{\texttt{fix\_\allowbreak anchors=None}}.\\
\end{longtable}
```
*Was:*
```latex
}} only). \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\end{longtable}
```

### calibrate_anchor_slm (line 654)

```latex
\textcolor{gray}{\texttt{self.anchor\_\allowbreak items}} & pandas series: item locations on the anchor scale: shifted by the translation constant, or from the CML fit with the anchors fixed.
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.anchor\_\allowbreak items}} & pandas series: item locations shifted onto the anchor scale.
```

### calibrate_anchor_slm (line 673)

```latex
 results from an earlier call survive a later call with a different anchor set instead of being overwritten in place.\\
\midrule
\textcolor{gray}{\texttt{self.anchor\_\allowbreak cml}} & Namespace or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{fix\_\allowbreak anchors=True}}, the CML fit with the anchors fixed -- \textcolor{gray}{\texttt{fixed\_\allowbreak items}}, \textcolor{gray}{\texttt{items}}, \textcolor{gray}{\texttt{item\_\allowbreak se}} (0 for the fixed items), \textcolor{gray}{\texttt{log\_\allowbreak likelihood}} and \textcolor{gray}{\texttt{convergence}}; \textcolor{gray}{\texttt{None}} otherwise.\\
\end{longtable}
```
*Was:*
```latex
 results from an earlier call survive a later call with a different anchor set instead of being overwritten in place.\\
\end{longtable}
```

### global_fit_slm (line 1231)

```latex
For CML estimates (\textcolor{gray}{\texttt{calibrate(estimation=\textquotesingle cml\textquotesingle )}}), \textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} gives Andersen's conditional likelihood ratio test \citep{Andersen1973} between the two groups, and \textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }} Martin-L\"of's test of unidimensionality \citep{MartinLof1973} for two item subsets; the Wald test then uses the CML covariances (Section \ref{std_errors_slm}).
```
*Was:*
```latex
\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} is not yet available.
```

### global_fit_slm (line 1238)

```latex
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle wald\textquotesingle , split=None, covariate=None, welch=True, alpha=0.05, seed=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle , no\_\allowbreak of\_\allowbreak samples=500,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=\textquotesingle chi2\textquotesingle , mc\_\allowbreak exceedances=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle wald\textquotesingle , split=\textquotesingle score\textquotesingle , covariate=None, welch=True, alpha=0.05,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500)}}\\
```

### global_fit_slm (line 1252)

```latex
\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }} (CML estimates only for the last two).
```
*Was:*
```latex
\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} (raises \textcolor{gray}{\texttt{NotImplementedError}}).
```

### global_fit_slm (line 1254)

```latex
For \textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }}: \textcolor{gray}{\texttt{\textquotesingle median\textquotesingle }} (items split at the median item location) or a list of the items in the first subset. Default is \textcolor{gray}{\texttt{split=None}}, which resolves to \textcolor{gray}{\texttt{\textquotesingle score\textquotesingle }}, or \textcolor{gray}{\texttt{\textquotesingle median\textquotesingle }} for \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}.
```
*Was:*
```latex
Default is \textcolor{gray}{\texttt{split=\textquotesingle score\textquotesingle }}.
```

### global_fit_slm (line 1264)

```latex
seed for \textcolor{gray}{\texttt{split=\textquotesingle random\textquotesingle }} and the Monte Carlo samples.
```
*Was:*
```latex
seed for \textcolor{gray}{\texttt{split=\textquotesingle random\textquotesingle }}.
```

### global_fit_slm (line 1267)

```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}. Also the number of Monte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
```

### global_fit_slm (line 1267)

```latex
nte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle chi2\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} (\textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }} only): chi-square p-value, or a Monte Carlo p-value from \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets simulated from the CML fit given each person's total. A \textcolor{gray}{\texttt{UserWarning}} is issued when the score table is too sparse for the chi-square p-value. Default is \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle chi2\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{mc\_\allowbreak exceedances}} & Integer or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}, stops once this many simulated statistics reach the observed one. Default is \textcolor{gray}{\texttt{mc\_\allowbreak exceedances=None}}.\\
\end{longtable}
```
*Was:*
```latex
nte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
\end{longtable}
```

### global_fit_slm (line 1291)

```latex
bal\_\allowbreak fit\_\allowbreak groups}} & Dictionary: \textcolor{gray}{\texttt{\{group name: calibrated model\}}}.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak obsexp}} & pandas dataframe (\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }}): per item and group, persons, observed and expected mean item score under the pooled CML fit (given each person's total and items) and their standardised difference.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak item\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }}): per item, each group's CML location, their difference and Wald $ z $, and the item's conditional chi-square over the groups with its p-value and Benjamini--Hochberg-adjusted p-value \citep{BenjaminiHochberg1995}.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak results}} & Dictionary: every result, keyed by method and options.\\
\end{longtable}
```
*Was:*
```latex
bal\_\allowbreak fit\_\allowbreak groups}} & Dictionary: \textcolor{gray}{\texttt{\{group name: calibrated model\}}}.\\
\end{longtable}
```

### global_fit_slm (line 1308)

```latex
To run Andersen's test and Martin-L\"of's test on CML estimates:

\textcolor{gray}{\texttt{self.calibrate(estimation=\textquotesingle cml\textquotesingle )}}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle andersen-lr\textquotesingle )}}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle martin-lof\textquotesingle )}}
```

### dif_test_slm (line 1602)

```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}} (default), also runs an Andersen-style omnibus LR test (reference vs. focal, every common item jointly) per focal group, stored in \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}} -- answering the question ``is there any DIF at all?'' With CML estimates, the omnibus test is a conditional likelihood ratio between the pooled CML fit and the pooled fit with a location shift for each common item in the focal group.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}} (default), also runs an Andersen-style omnibus LR test (reference vs. focal, every common item jointly) per focal group, stored in \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}} -- answering the question ``is there any DIF at all?''\\
```

### res_corr_analysis_slm (line 1368)

```latex
pairwise correlations between item residuals (Q3; \citealp{Yen1984}).
```
*Was:*
```latex
pairwise correlations between item residuals.
```

### res_corr_analysis_slm (line 1368)

```latex
owbreak correlations}} & A pandas dataframe of pairwise correlations between item residuals (Q3; \citealp{Yen1984}). \\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak star}} & pandas dataframe: Q3* \citep{Christensen2017}, the residual correlations minus their mean off-diagonal value.\\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak star\_\allowbreak max}} & Float: the largest Q3*.\\
\end{longtable}
```
*Was:*
```latex
owbreak correlations}} & A pandas dataframe of pairwise correlations between item residuals (Q3; \citealp{Yen1984}). \\
\end{longtable}
```

### res_corr_analysis_slm (line 1321)

```latex
After \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} (Section \ref{fit_statistics_slm}), the residuals are conditional on each person's total.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### item_stats_slm (line 936)

```latex
After \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} (Section \ref{fit_statistics_slm}), the fit statistics are conditional and further columns are added.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### item_stats_slm (line 1033)

```latex
 biserial correlation. Only produced if \textcolor{gray}{\texttt{point\_\allowbreak measure\_\allowbreak corr=True}}.\\
\midrule
\textcolor{gray}{\texttt{Observed gamma}} & Item-restscore gamma \citep{Kreiner2011}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Expected gamma}} & Expected item-restscore gamma. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Restscore p (BH)}} & Benjamini--Hochberg-adjusted p-value of the item-restscore test. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\midrule
\textcolor{gray}{\texttt{Q1}} & Conditional item chi-square \citep{vandenWollenberg1982}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Q1 df}} & Degrees of freedom of Q1. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Q1 p (BH)}} & Benjamini--Hochberg-adjusted p-value of Q1. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\midrule
\textcolor{gray}{\texttt{Infit p (Westfall-Young)}} & Westfall--Young bootstrap p-value of the infit mean square \citep{WestfallYoung1993}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=...)}}.\\
\midrule
\textcolor{gray}{\texttt{Outfit p (Westfall-Young)}} & Westfall--Young bootstrap p-value of the outfit mean square. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=...)}}.\\
\midrule
\textcolor{gray}{\texttt{Target}} & Location of maximum item information. Only produced after \textcolor{gray}{\texttt{targeting}} (Section \ref{targeting_slm}) and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Max information}} & Maximum item information. Only produced after \textcolor{gray}{\texttt{targeting}} and with \textcolor{gray}{\texttt{full=True}}.\\
\end{longtable}
```
*Was:*
```latex
 biserial correlation. Only produced if \textcolor{gray}{\texttt{point\_\allowbreak measure\_\allowbreak corr=True}}.\\
\end{longtable}
```

### person_stats_slm (line 1150)

```latex
ttt{Outfit MS}} & Outfit mean square. \\
\midrule
\textcolor{gray}{\texttt{Outfit Z}} & Standardised outfit z-score. \\
\midrule
\textcolor{gray}{\texttt{Conditional p}} & Probability, given the person's total, of a response pattern no more likely than theirs. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\end{longtable}
```
*Was:*
```latex
ttt{Outfit MS}} & Outfit mean square. \\
\midrule
\textcolor{gray}{\texttt{Outfit Z}} & Standardised outfit z-score. \\
\end{longtable}
```

### icc_slm (line 1843)

```latex
\midrule
\textcolor{gray}{\texttt{cond\_\allowbreak obs}} & Boolean: with \textcolor{gray}{\texttt{obs=True}}, also plots each class interval's mean expected score given each person's total and items (hollow diamonds), on the grouping of the conditional item chi-square (Section \ref{fit_statistics_slm}). No effect if \textcolor{gray}{\texttt{obs=False}}. Default is \textcolor{gray}{\texttt{cond\_\allowbreak obs=False}}.\\
\midrule
\textcolor{gray}{\texttt{cond\_\allowbreak obs\_\allowbreak bars}} & Boolean: with \textcolor{gray}{\texttt{cond\_\allowbreak obs=True}}, bars of $ \pm 1.96 $ standard deviations of each class interval's observed mean under the model. Default is \textcolor{gray}{\texttt{cond\_\allowbreak obs\_\allowbreak bars=True}}.\\
\midrule
\textcolor{gray}{\texttt{obs\_\allowbreak groups}} & String, pandas series, list or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{obs=True}} and a single item, observed points split by group (a column of \textcolor{gray}{\texttt{self.exogenous}}, a series indexed by person or labels in response order), one colour and marker per group. Ignored for a list of items. Default is \textcolor{gray}{\texttt{obs\_\allowbreak groups=None}}.\\
```

### icc_slm (line 1826)

```latex
no\_\allowbreak of\_\allowbreak classes=5,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~cond\_\allowbreak obs=False, cond\_\allowbreak obs\_\allowbreak bars=True, obs\_\allowbreak groups=None, 
```
*Was:*
```latex
no\_\allowbreak of\_\allowbreak classes=5, 
```

### item_stats_slm (new section before it) (line 931)

```latex
\subsubsection{\textcolor{gray}{\texttt{fit\_\allowbreak statistics}}} \label{fit_statistics_slm}

\textbf{Description}\\
Computes the fit statistics used by \textcolor{gray}{\texttt{item\_\allowbreak stats\_\allowbreak df}}, \textcolor{gray}{\texttt{person\_\allowbreak stats\_\allowbreak df}} and \textcolor{gray}{\texttt{test\_\allowbreak stats\_\allowbreak df}}: expected scores, residuals, and infit and outfit for items and persons. With \textcolor{gray}{\texttt{conditional=True}}, expected scores, variances and residuals are conditional on each person's total score over the items they responded to, and do not involve the person locations \citep{ChristensenKreiner2013}; the item-restscore test \citep{Kreiner2011}, the conditional item chi-square Q1 \citep{vandenWollenberg1982} and R1c \citep{Glas1988} on groups of persons, and each person's conditional p-value are also computed. With \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, p-values and cutoffs for item fit, item-restscore, Q1, Q3 and the first residual contrast also come from that many datasets simulated from the model given each person's total, each recalibrated with the object's settings, with Westfall--Young familywise p-values \citep{WestfallYoung1993}.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(warm\_\allowbreak corr=True, se=True, test\_\allowbreak stats=True, trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~tolerance=0.00001, max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~robust=False, method=\textquotesingle log-lik\textquotesingle , matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500, interval=None, seed=None, conditional=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~conditional\_\allowbreak samples=1000, bootstrap\_\allowbreak samples=None, q1\_\allowbreak groups=None)}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{warm\_\allowbreak corr}} & Boolean: If \textcolor{gray}{\texttt{True}}, Warm's \parencite*{Warm1989} bias correction is applied during generation of person estimates (see Section \ref{person_slm}). Default is \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}. \\
\midrule
\textcolor{gray}{\texttt{se}} & Boolean: if \textcolor{gray}{\texttt{False}}, skips the bootstrap standard errors (the fit statistics do not depend on them). Default is \textcolor{gray}{\texttt{se=True}}.\\
\midrule
\textcolor{gray}{\texttt{test\_\allowbreak stats}} & Boolean: if \textcolor{gray}{\texttt{True}}, also computes the test-level statistics (separation, strata, reliability; Section \ref{test_stats_slm}). Requires \textcolor{gray}{\texttt{se=True}}. Default is \textcolor{gray}{\texttt{test\_\allowbreak stats=True}}.\\
\midrule
\textcolor{gray}{\texttt{trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict}} & Boolean: if \textcolor{gray}{\texttt{True}}, keeps category probabilities for persons with non-extreme scores only, to save memory. Default is \textcolor{gray}{\texttt{trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict=False}}.\\
\midrule
\textcolor{gray}{\texttt{tolerance}} & Float: convergence criterion for Newton-Raphson iterations during generation of person estimates (see Section \ref{person_slm}). Default is \textcolor{gray}{\texttt{tolerance=0.00001}}. \\
\midrule
\textcolor{gray}{\texttt{max\_\allowbreak iters}} & Integer: maximum number of Newton-Raphson iterations during generation of person estimates (see Section \ref{person_slm}). Default is \textcolor{gray}{\texttt{max\_\allowbreak iters=100}}. \\
\midrule
\textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} & Float: Value in range $ (0, 1) $ to ensure a estimate is returned  during generation of person location estimates (see Section \ref{person_slm}) if a person has an extreme score (all items responded to are correct or incorrect). Since there is no finite ML location estimate for extreme scores, this adjusts the person's score to \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if zero) or \textcolor{gray}the maximum possible score minus \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if maximum score) before estimating location. This applies to maximum likelihood estimation only: with \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}, extreme scores have finite Warm estimates and the adjustment only sets the starting value. \\
\midrule
\textcolor{gray}{\texttt{constant}} & Float: additive smoothing constant \citep{Elliott2022a} for item estimation (see section \ref{calibrate_slm}). Default is \textcolor{gray}{\texttt{constant=None}}, which resolves to $ \min(1, 0.548\, I^{-0.2 \log_{10} N}) $, where $ N $ is the number of persons with at least one response and $ I $ is the number of items.\\
\midrule
\textcolor{gray}{\texttt{robust}} & Boolean: deprecated and has no effect; passing \textcolor{gray}{\texttt{robust=True}} raises a \textcolor{gray}{\texttt{FutureWarning}}. It will be removed in a future release. Default is \textcolor{gray}{\texttt{robust=False}}.\\
\midrule
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of estimates from pairwise reciprocal matrix \citep{Elliott2022a} during item estimation (see section \ref{calibrate_slm}). Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} \\
\midrule
\textcolor{gray}{\texttt{matrix\_\allowbreak power}} & Integer or \textcolor{gray}{\texttt{None}}: power to which conditional category response frequency matrix \citep[991]{Elliott2022a} is raised for generation of item location estimates. Each additional power increases the number of indirect pairwise comparisons \citep{Choppin1985, Elliott2022a}. Default is \textcolor{gray}{\texttt{matrix\_\allowbreak power=None}}, which resolves to \textcolor{gray}{\texttt{0}} (no matrix powering) when \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} and to \textcolor{gray}{\texttt{5}} for every other method. Passing \textcolor{gray}{\texttt{matrix\_\allowbreak power=0}} with a method other than \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} issues a \textcolor{gray}{\texttt{UserWarning}}.\\
\midrule
\textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Float: convergence stopping criterion for log-likelihood method for generation of item location estimates (see Section \ref{calibrate_slm}). Ignored for other methods. \\
\midrule
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} &  Integer: Number of bootstrap samples to generate for standard error estimates (see Section \ref{std_errors_slm}). More samples lead to more accurate standard error estimates, but take correspondingly longer to compute. Unused if \textcolor{gray}{\texttt{se=False}}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}. \\
\midrule
\textcolor{gray}{\texttt{interval}} & Float. Empirical interval to define quantiles of estimates from bootstrap samples, as an alternative to a confidence interval (see Section \ref{std_errors_slm}). Defines a central interval of proportion $ p $, e.g.\ \textcolor{gray}{\texttt{interval=0.95}} defines quantiles at 2.5\% and 97.5\%. More stable with larger numbers of bootstrap samples. Ignored if \textcolor{gray}{\texttt{se=False}}. Default is \textcolor{gray}{\texttt{interval=None}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the bootstrap standard errors, the conditional p-values and the bootstrap datasets. \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\midrule
\textcolor{gray}{\texttt{conditional}} & Boolean: if \textcolor{gray}{\texttt{True}}, computes the statistics conditionally (see Description). Default is \textcolor{gray}{\texttt{conditional=False}}.\\
\midrule
\textcolor{gray}{\texttt{conditional\_\allowbreak samples}} & Integer: Monte Carlo samples for the persons' conditional p-values (\textcolor{gray}{\texttt{conditional=True}} only). Default is \textcolor{gray}{\texttt{conditional\_\allowbreak samples=1000}}.\\
\midrule
\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}} & Integer or \textcolor{gray}{\texttt{None}}: number of simulated datasets for the bootstrap p-values and cutoffs (see Description; at least 400 recommended). Requires \textcolor{gray}{\texttt{conditional=True}}. Default is \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples=None}}.\\
\midrule
\textcolor{gray}{\texttt{q1\_\allowbreak groups}} & Integer, pandas series, list or \textcolor{gray}{\texttt{None}}: groups for Q1 and R1c (\textcolor{gray}{\texttt{conditional=True}} only). \textcolor{gray}{\texttt{None}}: groups of about equal size ordered by person location, tied locations kept together (with complete data, groups of adjacent total scores), about one group per 60 persons, from 2 to 10, each of at least 30 persons. An integer sets the number of groups; labels per person (a series indexed by person, or a list in response order) supply them. Default is \textcolor{gray}{\texttt{q1\_\allowbreak groups=None}}.\\
\end{longtable}

\textbf{Returns}\\
\begin{longtable}{p{2.2in} p{3.8in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{self.item\_\allowbreak infit\_\allowbreak ms}} & pandas series: item infit mean squares; also \textcolor{gray}{\texttt{item\_\allowbreak outfit\_\allowbreak ms}}, \textcolor{gray}{\texttt{item\_\allowbreak infit\_\allowbreak zstd}}, \textcolor{gray}{\texttt{item\_\allowbreak outfit\_\allowbreak zstd}} and the person equivalents (\textcolor{gray}{\texttt{person\_\allowbreak infit\_\allowbreak ms}} etc.). Conditional with \textcolor{gray}{\texttt{conditional=True}}.\\
\midrule
\textcolor{gray}{\texttt{self.residual\_\allowbreak df}} & pandas dataframe: residuals; \textcolor{gray}{\texttt{std\_\allowbreak residual\_\allowbreak df}}: standardised residuals.\\
\midrule
\textcolor{gray}{\texttt{self.fit\_\allowbreak conditional}} & Boolean: whether the statistics are conditional.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak restscore\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item, observed and expected gamma, SE, p and Benjamini--Hochberg-adjusted p; with \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, also bootstrap and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item, Q1, df, p and Benjamini--Hochberg-adjusted p; with \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, also bootstrap and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak groups}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item and group, persons, observed and expected mean item score, standardised difference and the group's location range.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak r1c}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): R1c, with rows \textcolor{gray}{\texttt{\textquotesingle item scores\textquotesingle }}: statistic, df, p, number of groups and the estimates used.\\
\midrule
\textcolor{gray}{\texttt{self.person\_\allowbreak conditional\_\allowbreak p}} & pandas series (\textcolor{gray}{\texttt{conditional=True}}): each person's probability, given their total, of a response pattern no more likely than theirs.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak fit\_\allowbreak bootstrap}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item, infit and outfit mean squares, 95\% bands, p and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak bootstrap}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item pair, Q3, expected Q3, 95\% cutoff, p and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.residual\_\allowbreak contrast\_\allowbreak bootstrap}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): first residual contrast, its expected value, 95\% cutoff and p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak fit\_\allowbreak bootstrap\_\allowbreak info}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): samples used, samples skipped and seed.\\
\end{longtable}

\textbf{Examples}\\
To compute conditional fit statistics:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(conditional=True)}}

To add bootstrap p-values from 1000 simulated datasets, reproducibly:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=1000, seed=1)}}

\newpage
```

### save_stats_slm (new section before it) (line 1531)

```latex
\subsubsection{\textcolor{gray}{\texttt{local\_\allowbreak dependence\_\allowbreak gamma}}} \label{local_dependence_gamma_slm}

\textbf{Description}\\
Tests every ordered pair of items for local dependence with the partial gamma \citep{KreinerChristensen2004}: Goodman and Kruskal's gamma between the two items' scores, counting concordant and discordant pairs of persons only within strata of the first item's rest score (the person's total over the other items they responded to, and which items those were). Under the model the partial gamma is zero; a positive value indicates local dependence. No parameter estimates are needed.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma(alpha=0.05, p\_\allowbreak method=\textquotesingle permutation\textquotesingle , samples=1000, seed=None)}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{alpha}} & Float: level for the \textcolor{gray}{\texttt{Flagged}} column (Benjamini--Hochberg-adjusted p-values). Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
\midrule
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle permutation\textquotesingle }}: normal p-value from the exact mean and variance of the concordant minus discordant count under permutation of the first item's responses within strata \citep{Kendall1970}; \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }}: p-value from \textcolor{gray}{\texttt{samples}} such permutations; \textcolor{gray}{\texttt{\textquotesingle asymptotic\textquotesingle }}: from the asymptotic standard error of the partial gamma \citep{Davis1967}, liberal with small samples or many missingness patterns (use \textcolor{gray}{\texttt{\textquotesingle permutation\textquotesingle }}). Default is \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle permutation\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{samples}} & Integer: permutations (\textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} only). Default is \textcolor{gray}{\texttt{samples=1000}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the permutations (\textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} only). Default is \textcolor{gray}{\texttt{seed=None}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma\_\allowbreak table}}, a pandas dataframe with one row per ordered item pair, sorted by gamma: \textcolor{gray}{\texttt{Item}} (the item whose rest score is conditioned on), \textcolor{gray}{\texttt{Paired item}}, \textcolor{gray}{\texttt{Gamma}}, \textcolor{gray}{\texttt{z}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{p (BH)}} and \textcolor{gray}{\texttt{Flagged}}. \textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma\_\allowbreak info}}: dictionary of \textcolor{gray}{\texttt{p\_\allowbreak method}}, \textcolor{gray}{\texttt{samples}} and \textcolor{gray}{\texttt{seed}}.\\

\textbf{Examples}\\
To test all item pairs for local dependence:\\
\textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma()}}

\newpage
\subsubsection{\textcolor{gray}{\texttt{targeting}}} \label{targeting_slm}

\textbf{Description}\\
Targeting and measurement properties of the test (all items) in the study population \citep{KreinerChristensen2013}: a normal distribution of person locations is fitted to the observed total scores (each person's total over the items they responded to, item parameters fixed), and the test information, the exact bias, SEM and RMSE of the weighted likelihood estimates \citep{Warm1989} and reliabilities are computed over it.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.targeting(grid=(-6.0, 6.0, 0.05))}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{grid}} & Tuple: (from, to, step) of the locations for \textcolor{gray}{\texttt{self.targeting\_\allowbreak curves}}. Default is \textcolor{gray}{\texttt{grid=(-6.0, 6.0, 0.05)}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.targeting\_\allowbreak summary}}, a pandas series:

\begin{longtable}{p{2.2in} p{3.8in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{Population mean}} & Mean of the fitted normal distribution.\\
\midrule
\textcolor{gray}{\texttt{Population SD}} & SD of the fitted normal distribution.\\
\midrule
\textcolor{gray}{\texttt{Target}} & Location of maximum test information.\\
\midrule
\textcolor{gray}{\texttt{Target - population mean}} & Distance of the target from the population mean.\\
\midrule
\textcolor{gray}{\texttt{Maximum information}} & Test information at the target.\\
\midrule
\textcolor{gray}{\texttt{Expected information}} & Test information averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected / maximum information}} & Their ratio.\\
\midrule
\textcolor{gray}{\texttt{Test difficulty}} & Location at which the expected total is half the maximum.\\
\midrule
\textcolor{gray}{\texttt{Expected |bias|}} & Absolute bias of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected SEM}} & SEM of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected RMSE}} & RMSE of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{rho1}} & $ V(T)/V(R) $: true-score over total-score variance.\\
\midrule
\textcolor{gray}{\texttt{rho2}} & $ \mathrm{corr}(R_1, R_2) $ for two independent administrations.\\
\midrule
\textcolor{gray}{\texttt{rho3}} & $ V(\theta)/V(\hat\theta) $.\\
\midrule
\textcolor{gray}{\texttt{rho4}} & $ \mathrm{corr}(\hat\theta_1, \hat\theta_2) $.\\
\midrule
\textcolor{gray}{\texttt{rho5}} & $ \mathrm{corr}(\theta, \hat\theta) $.\\
\midrule
\textcolor{gray}{\texttt{rho6}} & $ P(R_1 < R_2 \mid \theta_1 < \theta_2) $: probability that two random persons' totals order them correctly.\\
\end{longtable}

Also \textcolor{gray}{\texttt{self.targeting\_\allowbreak curves}} (by location: test information, asymptotic SE, bias, SEM and RMSE of the WLE, population density), \textcolor{gray}{\texttt{self.targeting\_\allowbreak score\_\allowbreak table}} (by total: WLE, model and observed proportions, observed among persons who responded to every item) and \textcolor{gray}{\texttt{self.targeting\_\allowbreak items}} (per item: target and maximum information).\\

\textbf{Examples}\\
To assess the targeting of the test:\\
\textcolor{gray}{\texttt{self.targeting()}}

\newpage
\subsubsection{\textcolor{gray}{\texttt{global\_\allowbreak stats\_\allowbreak df}}} \label{global_stats_slm}

\textbf{Description}\\
Produces a table of whole-test results in the form of a pandas dataframe: global tests (R1c, Andersen's test, Martin-L\"of's test, the Wald test), the first residual contrast, the largest Q3*, the numbers of flagged item pairs, items and persons, and the targeting and reliability indices of \textcolor{gray}{\texttt{targeting}}. By default it collects the results already computed and lists the others as not run. With \textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests=True}}, it runs those not yet run, each with its own defaults unless overridden in \textcolor{gray}{\texttt{test\_\allowbreak kwargs}}; Andersen's and Martin-L\"of's tests are left out for estimates other than CML. Running the conditional fit this way makes the fit statistics in \textcolor{gray}{\texttt{self.item\_\allowbreak stats}} and \textcolor{gray}{\texttt{self.person\_\allowbreak stats}} conditional.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df(run\_\allowbreak all\_\allowbreak tests=False, test\_\allowbreak kwargs=None, alpha=0.05, dp=3, filename=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~file\_\allowbreak format=\textquotesingle csv\textquotesingle )}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests}} & Boolean: if \textcolor{gray}{\texttt{True}}, runs the analyses not yet run (see Description), including \textcolor{gray}{\texttt{targeting}}. Default is \textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests=False}}.\\
\midrule
\textcolor{gray}{\texttt{test\_\allowbreak kwargs}} & Dictionary or \textcolor{gray}{\texttt{None}}: arguments for the analyses run, keyed by \textcolor{gray}{\texttt{\textquotesingle fit\_\allowbreak statistics\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle res\_\allowbreak corr\_\allowbreak analysis\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle local\_\allowbreak dependence\_\allowbreak gamma\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle targeting\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}. Default is \textcolor{gray}{\texttt{test\_\allowbreak kwargs=None}}.\\
\midrule
\textcolor{gray}{\texttt{alpha}} & Float: level for the counts of flagged items, item pairs and persons. Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
\midrule
\textcolor{gray}{\texttt{dp}} & Integer: Number of decimal places reported in output.\\
\midrule
\textcolor{gray}{\texttt{filename}} & String: The filename for the saved table, with no suffix for format. If \textcolor{gray}{\texttt{None}}, no file will be saved. Default is \textcolor{gray}{\texttt{filename=None}}.\\
\midrule
\textcolor{gray}{\texttt{file\_\allowbreak format}} & The format of the file: \textcolor{gray}{\texttt{csv}} or \textcolor{gray}{\texttt{xlsx}}. Default is \textcolor{gray}{\texttt{file\_\allowbreak format=csv}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.global\_\allowbreak stats}}, a pandas dataframe with one row per result and columns \textcolor{gray}{\texttt{Section}}, \textcolor{gray}{\texttt{Value}}, \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{p}} and \textcolor{gray}{\texttt{Details}} (the options defining the result, or the call needed to produce it).\\

\textbf{Examples}\\
To collect the results already computed:\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df()}}

To run all tests, with bootstrap p-values from 1000 simulated datasets:\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df(run\_\allowbreak all\_\allowbreak tests=True, test\_\allowbreak kwargs=\{\textquotesingle fit\_\allowbreak statistics\textquotesingle : \{\textquotesingle bootstrap\_\allowbreak samples\textquotesingle : 1000, \textquotesingle seed\textquotesingle : 1\}\})}}

\newpage
```

## pcm.tex

### calibrate_pcm (line 573)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~robust=False, estimation=\textquotesingle pair\textquotesingle , solver=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~robust=False)}}\\
```

### calibrate_pcm (line 571)

```latex
With \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, the uncentred thresholds are conditional maximum likelihood (CML) estimates \citep{Andersen1970}, maximising the likelihood of the responses given each person's total score over the items they responded to, starting from the PAIR estimates. Items without CML estimates are removed from the data, with a \textcolor{gray}{\texttt{UserWarning}}.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### calibrate_pcm (line 595)

```latex
oat: convergence criterion for log-likelihood method for central item location estimates. Ignored for other methods. \\
\midrule
\textcolor{gray}{\texttt{estimation}} & String: \textcolor{gray}{\texttt{\textquotesingle pair\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} (see Description). Default is \textcolor{gray}{\texttt{estimation=\textquotesingle pair\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{solver}} & String or \textcolor{gray}{\texttt{None}}: CML solver, \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle newton\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle lbfgs\textquotesingle }}; all reach the same estimates. \textcolor{gray}{\texttt{None}} resolves to \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }}. Ignored unless \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}. Default is \textcolor{gray}{\texttt{solver=None}}.\\
\end{longtable}
```
*Was:*
```latex
oat: convergence criterion for log-likelihood method for central item location estimates. Ignored for other methods. \\
\end{longtable}
```

### calibrate_pcm (line 618)

```latex
es, one row per item, one column per threshold position. NaN-padded for items with fewer thresholds than the maximum.\\
\midrule
\textcolor{gray}{\texttt{self.estimation}} & String: the estimation method used.\\
\midrule
\textcolor{gray}{\texttt{self.convergence}} & Dictionary or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, iterations, convergence, solver, conditional log-likelihood and the items removed (\textcolor{gray}{\texttt{\textquotesingle dropped\_\allowbreak items\textquotesingle }}, with the reason).\\
\midrule
\textcolor{gray}{\texttt{self.cml\_\allowbreak dropped\_\allowbreak responses}} & pandas dataframe: with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, the responses of the items removed.\\
\end{longtable}
```
*Was:*
```latex
es, one row per item, one column per threshold position. NaN-padded for items with fewer thresholds than the maximum.\\
\end{longtable}
```

### calibrate_pcm (line 639)

```latex
To produce conditional maximum likelihood estimates:

\textcolor{gray}{\texttt{self.calibrate(estimation=\textquotesingle cml\textquotesingle )}}
```

### std_errors_pcm (line 652)

```latex
For estimates from \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, \textcolor{gray}{\texttt{se\_\allowbreak method=None}} gives sandwich standard errors of the conditional likelihood, and \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle hessian\textquotesingle }} the inverse of its information matrix; the bootstrap refits with CML.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### std_errors_pcm (line 685)

```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle hessian\textquotesingle }} (CML estimates only), \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```
*Was:*
```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```

### std_errors_pcm (line 699)

```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle hessian\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```

### std_errors_pcm (line 701)

```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak covariance}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: sandwich or Hessian covariance matrix of the uncentred thresholds (sandwich or Hessian only; \textcolor{gray}{\texttt{None}} for the bootstrap).
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak covariance}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: sandwich covariance matrix of the uncentred thresholds (sandwich only; \textcolor{gray}{\texttt{None}} for the bootstrap).
```

### calibrate_anchor_pcm (line 761)

```latex
lik\_\allowbreak tol=0.000001, seed=None, fix\_\allowbreak anchors=None)}}\\
```
*Was:*
```latex
lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### calibrate_anchor_pcm (line 753)

```latex
For CML estimates (\textcolor{gray}{\texttt{calibrate(estimation=\textquotesingle cml\textquotesingle )}}), the anchor items that keep their supplied values (see \textcolor{gray}{\texttt{overwrite\_\allowbreak anchors}}) are by default held at those values in a CML fit, and every other parameter is estimated given them (\textcolor{gray}{\texttt{fix\_\allowbreak anchors}}).\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### calibrate_anchor_pcm (line 777)

```latex
(re-)runs \textcolor{gray}{\texttt{calibrate}}, with the estimation method and solver of the latest calibration, before anchoring.
```
*Was:*
```latex
(re-)runs \textcolor{gray}{\texttt{calibrate}} before anchoring.
```

### calibrate_anchor_pcm (line 813)

```latex
}} only). \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\midrule
\textcolor{gray}{\texttt{fix\_\allowbreak anchors}} & Boolean or \textcolor{gray}{\texttt{None}}: if \textcolor{gray}{\texttt{True}}, fits CML with the kept anchor items held at their supplied values (see Description); if \textcolor{gray}{\texttt{False}}, applies the translation. \textcolor{gray}{\texttt{None}} resolves to \textcolor{gray}{\texttt{True}} for CML estimates, else \textcolor{gray}{\texttt{False}}; \textcolor{gray}{\texttt{True}} with other estimation methods raises a \textcolor{gray}{\texttt{ValueError}}. Default is \textcolor{gray}{\texttt{fix\_\allowbreak anchors=None}}.\\
\end{longtable}
```
*Was:*
```latex
}} only). \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\end{longtable}
```

### calibrate_anchor_pcm (line 829)

```latex
\textcolor{gray}{\texttt{self.anchor\_\allowbreak items}} & pandas series: item locations on the anchor scale: shifted by the translation constant, or from the CML fit with the anchors fixed.
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.anchor\_\allowbreak items}} & pandas series: item locations shifted onto the anchor scale.
```

### calibrate_anchor_pcm (line 848)

```latex
 results from an earlier call survive a later call with a different anchor set instead of being overwritten in place.\\
\midrule
\textcolor{gray}{\texttt{self.anchor\_\allowbreak cml}} & Namespace or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{fix\_\allowbreak anchors=True}}, the CML fit with the anchors fixed -- \textcolor{gray}{\texttt{fixed\_\allowbreak items}}, \textcolor{gray}{\texttt{items}}, \textcolor{gray}{\texttt{item\_\allowbreak se}} (0 for the fixed items), thresholds (uncentred), \textcolor{gray}{\texttt{log\_\allowbreak likelihood}} and \textcolor{gray}{\texttt{convergence}}; \textcolor{gray}{\texttt{None}} otherwise.\\
\end{longtable}
```
*Was:*
```latex
 results from an earlier call survive a later call with a different anchor set instead of being overwritten in place.\\
\end{longtable}
```

### global_fit_pcm (line 1666)

```latex
For CML estimates (\textcolor{gray}{\texttt{calibrate(estimation=\textquotesingle cml\textquotesingle )}}), \textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} gives Andersen's conditional likelihood ratio test \citep{Andersen1973} between the two groups, and \textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }} Martin-L\"of's test of unidimensionality \citep{MartinLof1973} for two item subsets; the Wald test then uses the CML covariances (Section \ref{std_errors_pcm}).
```
*Was:*
```latex
\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} is not yet available.
```

### global_fit_pcm (line 1675)

```latex
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle wald\textquotesingle , split=None, covariate=None, welch=True, alpha=0.05, seed=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle , no\_\allowbreak of\_\allowbreak samples=500,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=\textquotesingle chi2\textquotesingle , mc\_\allowbreak exceedances=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle wald\textquotesingle , split=\textquotesingle score\textquotesingle , covariate=None, welch=True, alpha=0.05, seed=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500)}}\\
```

### global_fit_pcm (line 1689)

```latex
\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }} (CML estimates only for the last two).
```
*Was:*
```latex
\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} (raises \textcolor{gray}{\texttt{NotImplementedError}}).
```

### global_fit_pcm (line 1691)

```latex
For \textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }}: \textcolor{gray}{\texttt{\textquotesingle median\textquotesingle }} (items split at the median item location) or a list of the items in the first subset. Default is \textcolor{gray}{\texttt{split=None}}, which resolves to \textcolor{gray}{\texttt{\textquotesingle score\textquotesingle }}, or \textcolor{gray}{\texttt{\textquotesingle median\textquotesingle }} for \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}.
```
*Was:*
```latex
Default is \textcolor{gray}{\texttt{split=\textquotesingle score\textquotesingle }}.
```

### global_fit_pcm (line 1701)

```latex
seed for \textcolor{gray}{\texttt{split=\textquotesingle random\textquotesingle }} and the Monte Carlo samples.
```
*Was:*
```latex
seed for \textcolor{gray}{\texttt{split=\textquotesingle random\textquotesingle }}.
```

### global_fit_pcm (line 1704)

```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}. Also the number of Monte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
```

### global_fit_pcm (line 1704)

```latex
nte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle chi2\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} (\textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }} only): chi-square p-value, or a Monte Carlo p-value from \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets simulated from the CML fit given each person's total. A \textcolor{gray}{\texttt{UserWarning}} is issued when the score table is too sparse for the chi-square p-value. Default is \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle chi2\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{mc\_\allowbreak exceedances}} & Integer or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}, stops once this many simulated statistics reach the observed one. Default is \textcolor{gray}{\texttt{mc\_\allowbreak exceedances=None}}.\\
\end{longtable}
```
*Was:*
```latex
nte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
\end{longtable}
```

### global_fit_pcm (line 1728)

```latex
bal\_\allowbreak fit\_\allowbreak groups}} & Dictionary: \textcolor{gray}{\texttt{\{group name: calibrated model\}}}.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak obsexp}} & pandas dataframe (\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }}): per item and group, persons, observed and expected mean item score under the pooled CML fit (given each person's total and items) and their standardised difference.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak item\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }}): per item, each group's CML location, their difference and Wald $ z $, and the item's conditional chi-square over the groups with its p-value and Benjamini--Hochberg-adjusted p-value \citep{BenjaminiHochberg1995}.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak results}} & Dictionary: every result, keyed by method and options.\\
\end{longtable}
```
*Was:*
```latex
bal\_\allowbreak fit\_\allowbreak groups}} & Dictionary: \textcolor{gray}{\texttt{\{group name: calibrated model\}}}.\\
\end{longtable}
```

### global_fit_pcm (line 1745)

```latex
To run Andersen's test and Martin-L\"of's test on CML estimates:

\textcolor{gray}{\texttt{self.calibrate(estimation=\textquotesingle cml\textquotesingle )}}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle andersen-lr\textquotesingle )}}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle martin-lof\textquotesingle )}}
```

### dif_test_pcm (line 2042)

```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}} (default), also runs an Andersen-style omnibus LR test (reference vs. focal, every common item jointly) per focal group for the item-location component, stored in \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}}. Cheap relative to the per-item LR test -- one extra combined-group model fit per focal group, not per item. See \textcolor{gray}{\texttt{omnibus\_\allowbreak scope}} below for how \textcolor{gray}{\texttt{PCM}}'s per-item threshold freedom is accounted for. With CML estimates, the omnibus test is a conditional likelihood ratio between the pooled CML fit and the pooled fit with a location shift for each common item in the focal group.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}} (default), also runs an Andersen-style omnibus LR test (reference vs. focal, every common item jointly) per focal group for the item-location component, stored in \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}}. Cheap relative to the per-item LR test -- one extra combined-group model fit per focal group, not per item. See \textcolor{gray}{\texttt{omnibus\_\allowbreak scope}} below for how \textcolor{gray}{\texttt{PCM}}'s per-item threshold freedom is accounted for.\\
```

### res_corr_analysis_pcm (line 1805)

```latex
pairwise correlations between item residuals (Q3; \citealp{Yen1984}).
```
*Was:*
```latex
pairwise correlations between item standardised residuals.
```

### res_corr_analysis_pcm (line 1805)

```latex
owbreak correlations}} & A pandas dataframe of pairwise correlations between item residuals (Q3; \citealp{Yen1984}). \\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak star}} & pandas dataframe: Q3* \citep{Christensen2017}, the residual correlations minus their mean off-diagonal value.\\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak star\_\allowbreak max}} & Float: the largest Q3*.\\
\end{longtable}
```
*Was:*
```latex
owbreak correlations}} & A pandas dataframe of pairwise correlations between item residuals (Q3; \citealp{Yen1984}). \\
\end{longtable}
```

### res_corr_analysis_pcm (line 1758)

```latex
After \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} (Section \ref{fit_statistics_pcm}), the residuals are conditional on each person's total.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### item_stats_pcm (line 1115)

```latex
After \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} (Section \ref{fit_statistics_pcm}), the fit statistics are conditional and further columns are added.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### item_stats_pcm (line 1210)

```latex
exttt{True}}, threshold estimates indicate the presence of disordered thresholds within the item \citep{Andrich2010}.\\
\midrule
\textcolor{gray}{\texttt{Observed gamma}} & Item-restscore gamma \citep{Kreiner2011}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Expected gamma}} & Expected item-restscore gamma. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Restscore p (BH)}} & Benjamini--Hochberg-adjusted p-value of the item-restscore test. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\midrule
\textcolor{gray}{\texttt{Q1}} & Conditional item chi-square \citep{vandenWollenberg1982}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Q1 df}} & Degrees of freedom of Q1. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Q1 p (BH)}} & Benjamini--Hochberg-adjusted p-value of Q1. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\midrule
\textcolor{gray}{\texttt{Infit p (Westfall-Young)}} & Westfall--Young bootstrap p-value of the infit mean square \citep{WestfallYoung1993}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=...)}}.\\
\midrule
\textcolor{gray}{\texttt{Outfit p (Westfall-Young)}} & Westfall--Young bootstrap p-value of the outfit mean square. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=...)}}.\\
\midrule
\textcolor{gray}{\texttt{Target}} & Location of maximum item information. Only produced after \textcolor{gray}{\texttt{targeting}} (Section \ref{targeting_pcm}) and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Max information}} & Maximum item information. Only produced after \textcolor{gray}{\texttt{targeting}} and with \textcolor{gray}{\texttt{full=True}}.\\
\end{longtable}
```
*Was:*
```latex
exttt{True}}, threshold estimates indicate the presence of disordered thresholds within the item \citep{Andrich2010}.\\
\end{longtable}
```

### person_stats_pcm (line 1508)

```latex
ttt{Outfit MS}} & Outfit mean square. \\
\midrule
\textcolor{gray}{\texttt{Outfit Z}} & Standardised outfit z-score. \\
\midrule
\textcolor{gray}{\texttt{Conditional p}} & Probability, given the person's total, of a response pattern no more likely than theirs. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\end{longtable}
```
*Was:*
```latex
ttt{Outfit MS}} & Outfit mean square. \\
\midrule
\textcolor{gray}{\texttt{Outfit Z}} & Standardised outfit z-score. \\
\end{longtable}
```

### icc (line 2303)

```latex
\midrule
\textcolor{gray}{\texttt{cond\_\allowbreak obs}} & Boolean: with \textcolor{gray}{\texttt{obs=True}}, also plots each class interval's mean expected score given each person's total and items (hollow diamonds), on the grouping of the conditional item chi-square (Section \ref{fit_statistics_pcm}). No effect if \textcolor{gray}{\texttt{obs=False}}. Default is \textcolor{gray}{\texttt{cond\_\allowbreak obs=False}}.\\
\midrule
\textcolor{gray}{\texttt{cond\_\allowbreak obs\_\allowbreak bars}} & Boolean: with \textcolor{gray}{\texttt{cond\_\allowbreak obs=True}}, bars of $ \pm 1.96 $ standard deviations of each class interval's observed mean under the model. Default is \textcolor{gray}{\texttt{cond\_\allowbreak obs\_\allowbreak bars=True}}.\\
\midrule
\textcolor{gray}{\texttt{obs\_\allowbreak groups}} & String, pandas series, list or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{obs=True}} and a single item, observed points split by group (a column of \textcolor{gray}{\texttt{self.exogenous}}, a series indexed by person or labels in response order), one colour and marker per group. Ignored for a list of items. Default is \textcolor{gray}{\texttt{obs\_\allowbreak groups=None}}.\\
```

### icc (line 2285)

```latex
no\_\allowbreak of\_\allowbreak classes=5,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~cond\_\allowbreak obs=False, cond\_\allowbreak obs\_\allowbreak bars=True, obs\_\allowbreak groups=None, 
```
*Was:*
```latex
no\_\allowbreak of\_\allowbreak classes=5, 
```

### item_stats_pcm (new section before it) (line 1110)

```latex
\subsubsection{\textcolor{gray}{\texttt{fit\_\allowbreak statistics}}} \label{fit_statistics_pcm}

\textbf{Description}\\
Computes the fit statistics used by \textcolor{gray}{\texttt{item\_\allowbreak stats\_\allowbreak df}}, \textcolor{gray}{\texttt{person\_\allowbreak stats\_\allowbreak df}} and \textcolor{gray}{\texttt{test\_\allowbreak stats\_\allowbreak df}}: expected scores, residuals, and infit and outfit for items and persons. With \textcolor{gray}{\texttt{conditional=True}}, expected scores, variances and residuals are conditional on each person's total score over the items they responded to, and do not involve the person locations \citep{ChristensenKreiner2013}; the item-restscore test \citep{Kreiner2011}, the conditional item chi-square Q1 \citep{vandenWollenberg1982} and R1c \citep{Glas1988} on groups of persons, and each person's conditional p-value are also computed. With \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, p-values and cutoffs for item fit, item-restscore, Q1, Q3 and the first residual contrast also come from that many datasets simulated from the model given each person's total, each recalibrated with the object's settings, with Westfall--Young familywise p-values \citep{WestfallYoung1993}.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(warm\_\allowbreak corr=True, se=True, test\_\allowbreak stats=True, trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~tolerance=0.00001, max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , robust=False, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500, interval=None, seed=None, conditional=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~conditional\_\allowbreak samples=1000, bootstrap\_\allowbreak samples=None, q1\_\allowbreak groups=None)}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{warm\_\allowbreak corr}} & Boolean: If \textcolor{gray}{\texttt{True}}, Warm's bias correction \citep{Warm1989} is applied during generation of person location estimates (see Section \ref{person_pcm}). Default is \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}. \\
\midrule
\textcolor{gray}{\texttt{se}} & Boolean: if \textcolor{gray}{\texttt{False}}, skips the bootstrap standard errors (the fit statistics do not depend on them). Default is \textcolor{gray}{\texttt{se=True}}.\\
\midrule
\textcolor{gray}{\texttt{test\_\allowbreak stats}} & Boolean: if \textcolor{gray}{\texttt{True}}, also computes the test-level statistics (separation, strata, reliability; Section \ref{test_stats_pcm}). Requires \textcolor{gray}{\texttt{se=True}}. Default is \textcolor{gray}{\texttt{test\_\allowbreak stats=True}}.\\
\midrule
\textcolor{gray}{\texttt{trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict}} & Boolean: if \textcolor{gray}{\texttt{True}}, keeps category probabilities for persons with non-extreme scores only, to save memory. Default is \textcolor{gray}{\texttt{trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict=False}}.\\
\midrule
\textcolor{gray}{\texttt{tolerance}} & Float: convergence stopping criterion for Newton-Raphson iterations during generation of person estimates (see Section \ref{person_pcm}). Default is \textcolor{gray}{\texttt{tolerance=0.00001}}. \\
\midrule
\textcolor{gray}{\texttt{max\_\allowbreak iters}} & Integer: maximum number of Newton-Raphson iterations during generation of person estimates (see Section \ref{person_pcm}). Default is \textcolor{gray}{\texttt{max\_\allowbreak iters=100}}. \\
\midrule
\textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} & Float: Value in range $ (0, 1) $ to ensure an item location estimate is returned if a person has an extreme score (all responses correct or incorrect). Since there is no finite ML location estimate for extreme scores, adjusts the person's score to \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if zero) or \textcolor{gray}the maximum possible score minus \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if maximum score) before estimation. This applies to maximum likelihood estimation only: with \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}, extreme scores have finite Warm estimates and the adjustment only sets the starting value. \\
\midrule
\textcolor{gray}{\texttt{constant}} & Float or \textcolor{gray}{\texttt{None}}: additive smoothing constant \citep{Elliott2022a} for item estimation. Default is \textcolor{gray}{\texttt{constant=None}}, which resolves to $ 0.1\, N^{0.15} (1-p)^{0.75} \bar{m}^{-1.4} $, where $ N $ is the number of persons with at least one response, $ p $ is the proportion of missing responses among them and $ \bar{m} $ is the mean maximum score across items (see Section \ref{calibrate_pcm}).\\
\midrule
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of estimates from pairwise reciprocal matrix \citep{Elliott2022a} during item estimation (see section \ref{calibrate_pcm}). Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} \\
\midrule
\textcolor{gray}{\texttt{robust}} & Boolean: deprecated and has no effect; passing \textcolor{gray}{\texttt{robust=True}} raises a \textcolor{gray}{\texttt{FutureWarning}}. It will be removed in a future release. Default is \textcolor{gray}{\texttt{robust=False}}.\\
\midrule
\textcolor{gray}{\texttt{matrix\_\allowbreak power}} & Integer or \textcolor{gray}{\texttt{None}}: power to which conditional category response frequency matrix \citep[991]{Elliott2022a} is raised for generation of item location estimates. Each additional power increases the number of indirect pairwise comparisons \citep{Choppin1985, Elliott2022a}. Default is \textcolor{gray}{\texttt{matrix\_\allowbreak power=None}}, which resolves to \textcolor{gray}{\texttt{0}} (no matrix powering) when \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} and to \textcolor{gray}{\texttt{5}} for every other method. Passing \textcolor{gray}{\texttt{matrix\_\allowbreak power=0}} with a method other than \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} issues a \textcolor{gray}{\texttt{UserWarning}}.\\
\midrule
\textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Float: convergence stopping criterion for log-likelihood method for generation of item location estimates (see Section \ref{calibrate_pcm}). Ignored for other methods. \\
\midrule
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} &  Integer: Number of bootstrap samples to generate for standard error estimates (see Section \ref{std_errors_pcm}). More samples lead to more accurate standard error estimates, but take correspondingly longer to compute. Unused if \textcolor{gray}{\texttt{se=False}}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}. \\
\midrule
\textcolor{gray}{\texttt{interval}} & Float. Empirical interval to define quantiles of estimates from bootstrap samples, as an alternative to a confidence interval (see Section \ref{std_errors_pcm}). Defines a central interval of proportion $ p $, e.g.\ \textcolor{gray}{\texttt{interval=0.95}} defines quantiles at 2.5\% and 97.5\%. More stable with larger numbers of bootstrap samples. Ignored if \textcolor{gray}{\texttt{se=False}}. Default is \textcolor{gray}{\texttt{interval=None}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the bootstrap standard errors, the conditional p-values and the bootstrap datasets. \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\midrule
\textcolor{gray}{\texttt{conditional}} & Boolean: if \textcolor{gray}{\texttt{True}}, computes the statistics conditionally (see Description). Default is \textcolor{gray}{\texttt{conditional=False}}.\\
\midrule
\textcolor{gray}{\texttt{conditional\_\allowbreak samples}} & Integer: Monte Carlo samples for the persons' conditional p-values (\textcolor{gray}{\texttt{conditional=True}} only). Default is \textcolor{gray}{\texttt{conditional\_\allowbreak samples=1000}}.\\
\midrule
\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}} & Integer or \textcolor{gray}{\texttt{None}}: number of simulated datasets for the bootstrap p-values and cutoffs (see Description; at least 400 recommended). Requires \textcolor{gray}{\texttt{conditional=True}}. Default is \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples=None}}.\\
\midrule
\textcolor{gray}{\texttt{q1\_\allowbreak groups}} & Integer, pandas series, list or \textcolor{gray}{\texttt{None}}: groups for Q1 and R1c (\textcolor{gray}{\texttt{conditional=True}} only). \textcolor{gray}{\texttt{None}}: groups of about equal size ordered by person location, tied locations kept together (with complete data, groups of adjacent total scores), about one group per 60 persons, from 2 to 10, each of at least 30 persons. An integer sets the number of groups; labels per person (a series indexed by person, or a list in response order) supply them. Default is \textcolor{gray}{\texttt{q1\_\allowbreak groups=None}}.\\
\end{longtable}

\textbf{Returns}\\
\begin{longtable}{p{2.2in} p{3.8in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{self.item\_\allowbreak infit\_\allowbreak ms}} & pandas series: item infit mean squares; also \textcolor{gray}{\texttt{item\_\allowbreak outfit\_\allowbreak ms}}, \textcolor{gray}{\texttt{item\_\allowbreak infit\_\allowbreak zstd}}, \textcolor{gray}{\texttt{item\_\allowbreak outfit\_\allowbreak zstd}} and the person equivalents (\textcolor{gray}{\texttt{person\_\allowbreak infit\_\allowbreak ms}} etc.). Conditional with \textcolor{gray}{\texttt{conditional=True}}.\\
\midrule
\textcolor{gray}{\texttt{self.residual\_\allowbreak df}} & pandas dataframe: residuals; \textcolor{gray}{\texttt{std\_\allowbreak residual\_\allowbreak df}}: standardised residuals.\\
\midrule
\textcolor{gray}{\texttt{self.fit\_\allowbreak conditional}} & Boolean: whether the statistics are conditional.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak restscore\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item, observed and expected gamma, SE, p and Benjamini--Hochberg-adjusted p; with \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, also bootstrap and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item, Q1, df, p and Benjamini--Hochberg-adjusted p; with \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, also bootstrap and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak groups}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item and group, persons, observed and expected mean item score, standardised difference and the group's location range.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak r1c}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): R1c, with rows \textcolor{gray}{\texttt{\textquotesingle item scores\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }} (Glas's category-level form, liberal when category counts within groups are small: use \textcolor{gray}{\texttt{\textquotesingle item scores\textquotesingle }}): statistic, df, p, number of groups and the estimates used.\\
\midrule
\textcolor{gray}{\texttt{self.person\_\allowbreak conditional\_\allowbreak p}} & pandas series (\textcolor{gray}{\texttt{conditional=True}}): each person's probability, given their total, of a response pattern no more likely than theirs.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak fit\_\allowbreak bootstrap}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item, infit and outfit mean squares, 95\% bands, p and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak bootstrap}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item pair, Q3, expected Q3, 95\% cutoff, p and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.residual\_\allowbreak contrast\_\allowbreak bootstrap}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): first residual contrast, its expected value, 95\% cutoff and p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak fit\_\allowbreak bootstrap\_\allowbreak info}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): samples used, samples skipped and seed.\\
\end{longtable}

\textbf{Examples}\\
To compute conditional fit statistics:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(conditional=True)}}

To add bootstrap p-values from 1000 simulated datasets, reproducibly:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=1000, seed=1)}}

\newpage
```

### save_stats_pcm (new section before it) (line 1969)

```latex
\subsubsection{\textcolor{gray}{\texttt{local\_\allowbreak dependence\_\allowbreak gamma}}} \label{local_dependence_gamma_pcm}

\textbf{Description}\\
Tests every ordered pair of items for local dependence with the partial gamma \citep{KreinerChristensen2004}: Goodman and Kruskal's gamma between the two items' scores, counting concordant and discordant pairs of persons only within strata of the first item's rest score (the person's total over the other items they responded to, and which items those were). Under the model the partial gamma is zero; a positive value indicates local dependence. No parameter estimates are needed.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma(alpha=0.05, p\_\allowbreak method=\textquotesingle permutation\textquotesingle , samples=1000, seed=None)}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{alpha}} & Float: level for the \textcolor{gray}{\texttt{Flagged}} column (Benjamini--Hochberg-adjusted p-values). Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
\midrule
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle permutation\textquotesingle }}: normal p-value from the exact mean and variance of the concordant minus discordant count under permutation of the first item's responses within strata \citep{Kendall1970}; \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }}: p-value from \textcolor{gray}{\texttt{samples}} such permutations; \textcolor{gray}{\texttt{\textquotesingle asymptotic\textquotesingle }}: from the asymptotic standard error of the partial gamma \citep{Davis1967}, liberal with small samples or many missingness patterns (use \textcolor{gray}{\texttt{\textquotesingle permutation\textquotesingle }}). Default is \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle permutation\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{samples}} & Integer: permutations (\textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} only). Default is \textcolor{gray}{\texttt{samples=1000}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the permutations (\textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} only). Default is \textcolor{gray}{\texttt{seed=None}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma\_\allowbreak table}}, a pandas dataframe with one row per ordered item pair, sorted by gamma: \textcolor{gray}{\texttt{Item}} (the item whose rest score is conditioned on), \textcolor{gray}{\texttt{Paired item}}, \textcolor{gray}{\texttt{Gamma}}, \textcolor{gray}{\texttt{z}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{p (BH)}} and \textcolor{gray}{\texttt{Flagged}}. \textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma\_\allowbreak info}}: dictionary of \textcolor{gray}{\texttt{p\_\allowbreak method}}, \textcolor{gray}{\texttt{samples}} and \textcolor{gray}{\texttt{seed}}.\\

\textbf{Examples}\\
To test all item pairs for local dependence:\\
\textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma()}}

\newpage
\subsubsection{\textcolor{gray}{\texttt{targeting}}} \label{targeting_pcm}

\textbf{Description}\\
Targeting and measurement properties of the test (all items) in the study population \citep{KreinerChristensen2013}: a normal distribution of person locations is fitted to the observed total scores (each person's total over the items they responded to, item parameters fixed), and the test information, the exact bias, SEM and RMSE of the weighted likelihood estimates \citep{Warm1989} and reliabilities are computed over it.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.targeting(grid=(-6.0, 6.0, 0.05))}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{grid}} & Tuple: (from, to, step) of the locations for \textcolor{gray}{\texttt{self.targeting\_\allowbreak curves}}. Default is \textcolor{gray}{\texttt{grid=(-6.0, 6.0, 0.05)}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.targeting\_\allowbreak summary}}, a pandas series:

\begin{longtable}{p{2.2in} p{3.8in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{Population mean}} & Mean of the fitted normal distribution.\\
\midrule
\textcolor{gray}{\texttt{Population SD}} & SD of the fitted normal distribution.\\
\midrule
\textcolor{gray}{\texttt{Target}} & Location of maximum test information.\\
\midrule
\textcolor{gray}{\texttt{Target - population mean}} & Distance of the target from the population mean.\\
\midrule
\textcolor{gray}{\texttt{Maximum information}} & Test information at the target.\\
\midrule
\textcolor{gray}{\texttt{Expected information}} & Test information averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected / maximum information}} & Their ratio.\\
\midrule
\textcolor{gray}{\texttt{Test difficulty}} & Location at which the expected total is half the maximum.\\
\midrule
\textcolor{gray}{\texttt{Expected |bias|}} & Absolute bias of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected SEM}} & SEM of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected RMSE}} & RMSE of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{rho1}} & $ V(T)/V(R) $: true-score over total-score variance.\\
\midrule
\textcolor{gray}{\texttt{rho2}} & $ \mathrm{corr}(R_1, R_2) $ for two independent administrations.\\
\midrule
\textcolor{gray}{\texttt{rho3}} & $ V(\theta)/V(\hat\theta) $.\\
\midrule
\textcolor{gray}{\texttt{rho4}} & $ \mathrm{corr}(\hat\theta_1, \hat\theta_2) $.\\
\midrule
\textcolor{gray}{\texttt{rho5}} & $ \mathrm{corr}(\theta, \hat\theta) $.\\
\midrule
\textcolor{gray}{\texttt{rho6}} & $ P(R_1 < R_2 \mid \theta_1 < \theta_2) $: probability that two random persons' totals order them correctly.\\
\end{longtable}

Also \textcolor{gray}{\texttt{self.targeting\_\allowbreak curves}} (by location: test information, asymptotic SE, bias, SEM and RMSE of the WLE, population density), \textcolor{gray}{\texttt{self.targeting\_\allowbreak score\_\allowbreak table}} (by total: WLE, model and observed proportions, observed among persons who responded to every item) and \textcolor{gray}{\texttt{self.targeting\_\allowbreak items}} (per item: target and maximum information).\\

\textbf{Examples}\\
To assess the targeting of the test:\\
\textcolor{gray}{\texttt{self.targeting()}}

\newpage
\subsubsection{\textcolor{gray}{\texttt{global\_\allowbreak stats\_\allowbreak df}}} \label{global_stats_pcm}

\textbf{Description}\\
Produces a table of whole-test results in the form of a pandas dataframe: global tests (R1c, Andersen's test, Martin-L\"of's test, the Wald test), the first residual contrast, the largest Q3*, the numbers of flagged item pairs, items and persons, and the targeting and reliability indices of \textcolor{gray}{\texttt{targeting}}. By default it collects the results already computed and lists the others as not run. With \textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests=True}}, it runs those not yet run, each with its own defaults unless overridden in \textcolor{gray}{\texttt{test\_\allowbreak kwargs}}; Andersen's and Martin-L\"of's tests are left out for estimates other than CML. Running the conditional fit this way makes the fit statistics in \textcolor{gray}{\texttt{self.item\_\allowbreak stats}} and \textcolor{gray}{\texttt{self.person\_\allowbreak stats}} conditional.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df(run\_\allowbreak all\_\allowbreak tests=False, test\_\allowbreak kwargs=None, alpha=0.05, dp=3, filename=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~file\_\allowbreak format=\textquotesingle csv\textquotesingle )}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests}} & Boolean: if \textcolor{gray}{\texttt{True}}, runs the analyses not yet run (see Description), including \textcolor{gray}{\texttt{targeting}}. Default is \textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests=False}}.\\
\midrule
\textcolor{gray}{\texttt{test\_\allowbreak kwargs}} & Dictionary or \textcolor{gray}{\texttt{None}}: arguments for the analyses run, keyed by \textcolor{gray}{\texttt{\textquotesingle fit\_\allowbreak statistics\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle res\_\allowbreak corr\_\allowbreak analysis\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle local\_\allowbreak dependence\_\allowbreak gamma\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle targeting\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}. Default is \textcolor{gray}{\texttt{test\_\allowbreak kwargs=None}}.\\
\midrule
\textcolor{gray}{\texttt{alpha}} & Float: level for the counts of flagged items, item pairs and persons. Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
\midrule
\textcolor{gray}{\texttt{dp}} & Integer: Number of decimal places reported in output.\\
\midrule
\textcolor{gray}{\texttt{filename}} & String: The filename for the saved table, with no suffix for format. If \textcolor{gray}{\texttt{None}}, no file will be saved. Default is \textcolor{gray}{\texttt{filename=None}}.\\
\midrule
\textcolor{gray}{\texttt{file\_\allowbreak format}} & The format of the file: \textcolor{gray}{\texttt{csv}} or \textcolor{gray}{\texttt{xlsx}}. Default is \textcolor{gray}{\texttt{file\_\allowbreak format=csv}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.global\_\allowbreak stats}}, a pandas dataframe with one row per result and columns \textcolor{gray}{\texttt{Section}}, \textcolor{gray}{\texttt{Value}}, \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{p}} and \textcolor{gray}{\texttt{Details}} (the options defining the result, or the call needed to produce it).\\

\textbf{Examples}\\
To collect the results already computed:\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df()}}

To run all tests, with bootstrap p-values from 1000 simulated datasets:\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df(run\_\allowbreak all\_\allowbreak tests=True, test\_\allowbreak kwargs=\{\textquotesingle fit\_\allowbreak statistics\textquotesingle : \{\textquotesingle bootstrap\_\allowbreak samples\textquotesingle : 1000, \textquotesingle seed\textquotesingle : 1\}\})}}

\newpage
```

### model_selection_pcm (line 1725)

```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05, min\_\allowbreak effect=0, sampling=\textquotesingle auto\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~likelihood=\textquotesingle conditional\textquotesingle , warm\_\allowbreak corr=True, tolerance=0.00001,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None, method=\textquotesingle log-lik\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~robust=False, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05, min\_\allowbreak effect=0,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~sampling=\textquotesingle dynamic\textquotesingle , warm\_\allowbreak corr=True, tolerance=0.00001, max\_\allowbreak iters=100,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None, robust=False, method=\textquotesingle log-lik\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### model_selection_pcm (line 1724)

```latex
By default (\textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}) each model is scored by the conditional log-likelihood given each person's total score, after one Newton step from its estimates; \textcolor{gray}{\texttt{likelihood=\textquotesingle joint\textquotesingle }} uses the log-likelihood given the person estimates.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### model_selection_pcm (line 1750)

```latex
\textcolor{gray}{\texttt{sampling}} & \textcolor{gray}{\texttt{None}}, \textcolor{gray}{\texttt{\textquotesingle auto\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }}, or Integer: Subsamples $ T $ non-extreme persons before computing log-likelihoods: at very large $ N $, the LR/AIC/BIC comparison becomes biased toward the more complex (PCM) model, since log-likelihood differences scale with $ N $; capping the effective $ N $ mitigates this. An integer defines $ T $.\textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} uses $ T = \min(20(I-1)(m-1), 1500) $, where $ I $ is the number of items and $ m $ the shared maximum score; subsampling is skipped whenever the actual non-extreme sample size is $ \leq T $. \textcolor{gray}{\texttt{None}} applies no subsampling. \textcolor{gray}{\texttt{\textquotesingle auto\textquotesingle }} resolves to \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} with \textcolor{gray}{\texttt{likelihood=\textquotesingle joint\textquotesingle }} and to \textcolor{gray}{\texttt{None}} with \textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}. Default is \textcolor{gray}{\texttt{sampling=\textquotesingle auto\textquotesingle }}.
```
*Was:*
```latex
\textcolor{gray}{\texttt{sampling}} & \textcolor{gray}{\texttt{None}}, \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }}, or Integer: Subsamples $ T $ non-extreme persons before computing log-likelihoods: at very large $ N $, the LR/AIC/BIC comparison becomes biased toward the more complex (PCM) model, since log-likelihood differences scale with $ N $; capping the effective $ N $ mitigates this. An integer defines $ T $.\textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} uses $ T = \min(20(I-1)(m-1), 1500) $, where $ I $ is the number of items and $ m $ the shared maximum score; subsampling is skipped whenever the actual non-extreme sample size is $ \leq T $. \textcolor{gray}{\texttt{None}} applies no subsampling. Default is \textcolor{gray}{\texttt{sampling=\textquotesingle dynamic\textquotesingle }}.
```

### model_selection_pcm (line 1751)

```latex
\midrule
\textcolor{gray}{\texttt{likelihood}} & String: \textcolor{gray}{\texttt{\textquotesingle conditional\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle joint\textquotesingle }} (see Description). Default is \textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}.\\
```

## rsm.tex

### calibrate_rsm (line 402)

```latex
There are four approaches to estimation available, selectable through the \textcolor{gray}{\texttt{estimation}} argument: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}.
```
*Was:*
```latex
There are three approaches to estimation available, selectable through the \textcolor{gray}{\texttt{estimation}} argument: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle}}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle}} and \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle}}.
```

### calibrate_rsm (line 431)

```latex
\textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }} gives conditional maximum likelihood (CML) estimates \citep{Andersen1970}, maximising the likelihood of the responses given each person's total score over the items they responded to, starting from the \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates. Items without CML estimates are removed from the data, with a \textcolor{gray}{\texttt{UserWarning}}; a category unused by every person with a non-extreme total raises a \textcolor{gray}{\texttt{ValueError}}.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### calibrate_rsm (line 435)

```latex
estimation=None, solver=None)
```
*Was:*
```latex
estimation=None)
```

### calibrate_rsm (line 462)

```latex
\textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} (see Description).
```
*Was:*
```latex
\textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} (see Description).
```

### calibrate_rsm (line 463)

```latex
\midrule
\textcolor{gray}{\texttt{solver}} & String or \textcolor{gray}{\texttt{None}}: CML solver, \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle newton\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle lbfgs\textquotesingle }}; all reach the same estimates. \textcolor{gray}{\texttt{None}} resolves to \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }}. Ignored unless \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}. Default is \textcolor{gray}{\texttt{solver=None}}.\\
```

### calibrate_rsm (line 493)

```latex
\textcolor{gray}{\texttt{self.convergence}} & Dictionary or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, the number of Newton--Raphson iterations (\textcolor{gray}{\texttt{\textquotesingle iterations\textquotesingle }}) and whether the fit converged (\textcolor{gray}{\texttt{\textquotesingle converged\textquotesingle }}); a \textcolor{gray}{\texttt{UserWarning}} is issued if it did not. \textcolor{gray}{\texttt{None}} with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. With \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, also the solver, the conditional log-likelihood and the items removed (\textcolor{gray}{\texttt{\textquotesingle dropped\_\allowbreak items\textquotesingle }}, with the reason).\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.convergence}} & Dictionary or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, the number of Newton--Raphson iterations (\textcolor{gray}{\texttt{\textquotesingle iterations\textquotesingle }}) and whether the fit converged (\textcolor{gray}{\texttt{\textquotesingle converged\textquotesingle }}); a \textcolor{gray}{\texttt{UserWarning}} is issued if it did not. \textcolor{gray}{\texttt{None}} with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}.\\
```

### calibrate_rsm (line 493)

```latex
s removed (\textcolor{gray}{\texttt{\textquotesingle dropped\_\allowbreak items\textquotesingle }}, with the reason).\\
\midrule
\textcolor{gray}{\texttt{self.cml\_\allowbreak dropped\_\allowbreak responses}} & pandas dataframe: with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, the responses of the items removed.\\
\end{longtable}
```
*Was:*
```latex
s removed (\textcolor{gray}{\texttt{\textquotesingle dropped\_\allowbreak items\textquotesingle }}, with the reason).\\
\end{longtable}
```

### calibrate_rsm (line 527)

```latex
To produce conditional maximum likelihood estimates:

\textcolor{gray}{\texttt{self.calibrate(estimation=\textquotesingle cml\textquotesingle )}}
```

### std_errors_rsm (line 540)

```latex
For estimates from \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, \textcolor{gray}{\texttt{se\_\allowbreak method=None}} gives sandwich standard errors of the conditional likelihood, and \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle hessian\textquotesingle }} the inverse of its information matrix; the bootstrap refits with CML.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### std_errors_rsm (line 573)

```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle hessian\textquotesingle }} (CML estimates only), \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```
*Was:*
```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```

### std_errors_rsm (line 587)

```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle hessian\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}
```

### std_errors_rsm (line 589)

```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak covariance}} & Dictionary or \textcolor{gray}{\texttt{None}}: sandwich or Hessian covariance matrices, \textcolor{gray}{\texttt{\{\textquotesingle items\textquotesingle : pandas dataframe, \textquotesingle cat\_\allowbreak widths\textquotesingle : pandas dataframe\}}} (sandwich or Hessian only; \textcolor{gray}{\texttt{None}} for the bootstrap).
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.se\_\allowbreak covariance}} & Dictionary or \textcolor{gray}{\texttt{None}}: sandwich covariance matrices, \textcolor{gray}{\texttt{\{\textquotesingle items\textquotesingle : pandas dataframe, \textquotesingle cat\_\allowbreak widths\textquotesingle : pandas dataframe\}}} (sandwich only; \textcolor{gray}{\texttt{None}} for the bootstrap).
```

### calibrate_anchor_rsm (line 661)

```latex
None, threshold\_\allowbreak method=None, fix\_\allowbreak anchors=None)}}\\
```
*Was:*
```latex
None, threshold\_\allowbreak method=None)}}\\
```

### calibrate_anchor_rsm (line 651)

```latex
For CML estimates (\textcolor{gray}{\texttt{calibrate(estimation=\textquotesingle cml\textquotesingle )}}), the anchor items that keep their supplied values (see \textcolor{gray}{\texttt{overwrite\_\allowbreak anchors}}) are by default held at those values in a CML fit, and every other parameter is estimated given them (\textcolor{gray}{\texttt{fix\_\allowbreak anchors}}).\\

\newpage
\textbf{Usage}
```
*Was:*
```latex
\newpage
\textbf{Usage}
```

### calibrate_anchor_rsm (line 677)

```latex
(re-)runs \textcolor{gray}{\texttt{calibrate}}, with the estimation method and solver of the latest calibration, before anchoring.
```
*Was:*
```latex
(re-)runs \textcolor{gray}{\texttt{calibrate}} before anchoring.
```

### calibrate_anchor_rsm (line 711)

```latex
}} only). \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\midrule
\textcolor{gray}{\texttt{fix\_\allowbreak anchors}} & Boolean or \textcolor{gray}{\texttt{None}}: if \textcolor{gray}{\texttt{True}}, fits CML with the kept anchor items held at their supplied values (see Description); if \textcolor{gray}{\texttt{False}}, applies the translation. \textcolor{gray}{\texttt{None}} resolves to \textcolor{gray}{\texttt{True}} for CML estimates, else \textcolor{gray}{\texttt{False}}; \textcolor{gray}{\texttt{True}} with other estimation methods raises a \textcolor{gray}{\texttt{ValueError}}. Default is \textcolor{gray}{\texttt{fix\_\allowbreak anchors=None}}.\\
\end{longtable}
```
*Was:*
```latex
}} only). \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\end{longtable}
```

### calibrate_anchor_rsm (line 726)

```latex
\textcolor{gray}{\texttt{self.anchor\_\allowbreak items}} & pandas series: item locations on the anchor scale: shifted by the translation constant, or from the CML fit with the anchors fixed.
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.anchor\_\allowbreak items}} & pandas series: item locations shifted onto the anchor scale.
```

### calibrate_anchor_rsm (line 745)

```latex
 results from an earlier call survive a later call with a different anchor set instead of being overwritten in place.\\
\midrule
\textcolor{gray}{\texttt{self.anchor\_\allowbreak cml}} & Namespace or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{fix\_\allowbreak anchors=True}}, the CML fit with the anchors fixed -- \textcolor{gray}{\texttt{fixed\_\allowbreak items}}, \textcolor{gray}{\texttt{items}}, \textcolor{gray}{\texttt{item\_\allowbreak se}} (0 for the fixed items), thresholds (shared), \textcolor{gray}{\texttt{log\_\allowbreak likelihood}} and \textcolor{gray}{\texttt{convergence}}; \textcolor{gray}{\texttt{None}} otherwise.\\
\end{longtable}
```
*Was:*
```latex
 results from an earlier call survive a later call with a different anchor set instead of being overwritten in place.\\
\end{longtable}
```

### global_fit_rsm (line 1546)

```latex
For CML estimates (\textcolor{gray}{\texttt{calibrate(estimation=\textquotesingle cml\textquotesingle )}}), \textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} gives Andersen's conditional likelihood ratio test \citep{Andersen1973} between the two groups, and \textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }} Martin-L\"of's test of unidimensionality \citep{MartinLof1973} for two item subsets; the Wald test then uses the CML covariances (Section \ref{std_errors_rsm}).
```
*Was:*
```latex
\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} is not yet available.
```

### global_fit_rsm (line 1553)

```latex
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle wald\textquotesingle , split=None, covariate=None, welch=True, alpha=0.05, seed=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle , no\_\allowbreak of\_\allowbreak samples=500,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=\textquotesingle chi2\textquotesingle , mc\_\allowbreak exceedances=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle wald\textquotesingle , split=\textquotesingle score\textquotesingle , covariate=None, welch=True, alpha=0.05,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, se\_\allowbreak correction=\textquotesingle kauermann-carroll\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=500)}}\\
```

### global_fit_rsm (line 1567)

```latex
\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }} (CML estimates only for the last two).
```
*Was:*
```latex
\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} (raises \textcolor{gray}{\texttt{NotImplementedError}}).
```

### global_fit_rsm (line 1569)

```latex
For \textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }}: \textcolor{gray}{\texttt{\textquotesingle median\textquotesingle }} (items split at the median item location) or a list of the items in the first subset. Default is \textcolor{gray}{\texttt{split=None}}, which resolves to \textcolor{gray}{\texttt{\textquotesingle score\textquotesingle }}, or \textcolor{gray}{\texttt{\textquotesingle median\textquotesingle }} for \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}.
```
*Was:*
```latex
Default is \textcolor{gray}{\texttt{split=\textquotesingle score\textquotesingle }}.
```

### global_fit_rsm (line 1579)

```latex
seed for \textcolor{gray}{\texttt{split=\textquotesingle random\textquotesingle }} and the Monte Carlo samples.
```
*Was:*
```latex
seed for \textcolor{gray}{\texttt{split=\textquotesingle random\textquotesingle }}.
```

### global_fit_rsm (line 1582)

```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}. Also the number of Monte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples per group, used when the object was calibrated with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
```

### global_fit_rsm (line 1582)

```latex
nte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle chi2\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} (\textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }} only): chi-square p-value, or a Monte Carlo p-value from \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets simulated from the CML fit given each person's total. A \textcolor{gray}{\texttt{UserWarning}} is issued when the score table is too sparse for the chi-square p-value. Default is \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle chi2\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{mc\_\allowbreak exceedances}} & Integer or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}, stops once this many simulated statistics reach the observed one. Default is \textcolor{gray}{\texttt{mc\_\allowbreak exceedances=None}}.\\
\end{longtable}
```
*Was:*
```latex
nte Carlo samples for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
\end{longtable}
```

### global_fit_rsm (line 1606)

```latex
bal\_\allowbreak fit\_\allowbreak groups}} & Dictionary: \textcolor{gray}{\texttt{\{group name: calibrated model\}}}.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak obsexp}} & pandas dataframe (\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }}): per item and group, persons, observed and expected mean item score under the pooled CML fit (given each person's total and items) and their standardised difference.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak item\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }}): per item, each group's CML location, their difference and Wald $ z $, and the item's conditional chi-square over the groups with its p-value and Benjamini--Hochberg-adjusted p-value \citep{BenjaminiHochberg1995}.\\
\midrule
\textcolor{gray}{\texttt{self.global\_\allowbreak fit\_\allowbreak results}} & Dictionary: every result, keyed by method and options.\\
\end{longtable}
```
*Was:*
```latex
bal\_\allowbreak fit\_\allowbreak groups}} & Dictionary: \textcolor{gray}{\texttt{\{group name: calibrated model\}}}.\\
\end{longtable}
```

### global_fit_rsm (line 1623)

```latex
To run Andersen's test and Martin-L\"of's test on CML estimates:

\textcolor{gray}{\texttt{self.calibrate(estimation=\textquotesingle cml\textquotesingle )}}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle andersen-lr\textquotesingle )}}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak fit(method=\textquotesingle martin-lof\textquotesingle )}}
```

### dif_test_rsm (line 1919)

```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}} (default), also runs an Andersen-style omnibus LR test (reference vs. focal, every common item jointly) per focal group for the item-location component, stored in \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}}. With CML estimates, the omnibus test is a conditional likelihood ratio between the pooled CML fit and the pooled fit with a location shift for each common item in the focal group.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean: if \textcolor{gray}{\texttt{True}} (default), also runs an Andersen-style omnibus LR test (reference vs. focal, every common item jointly) per focal group for the item-location component, stored in \textcolor{gray}{\texttt{self.dif\_\allowbreak omnibus\_\allowbreak table}}.\\
```

### res_corr_analysis_rsm (line 1683)

```latex
pairwise correlations between item residuals (Q3; \citealp{Yen1984}).
```
*Was:*
```latex
pairwise correlations between item standardised residuals.
```

### res_corr_analysis_rsm (line 1683)

```latex
owbreak correlations}} & A pandas dataframe of pairwise correlations between item residuals (Q3; \citealp{Yen1984}). \\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak star}} & pandas dataframe: Q3* \citep{Christensen2017}, the residual correlations minus their mean off-diagonal value.\\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak star\_\allowbreak max}} & Float: the largest Q3*.\\
\end{longtable}
```
*Was:*
```latex
owbreak correlations}} & A pandas dataframe of pairwise correlations between item residuals (Q3; \citealp{Yen1984}). \\
\end{longtable}
```

### res_corr_analysis_rsm (line 1636)

```latex
After \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} (Section \ref{fit_statistics_rsm}), the residuals are conditional on each person's total.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### item_stats_rsm (line 1003)

```latex
After \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} (Section \ref{fit_statistics_rsm}), the fit statistics are conditional and further columns are added.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### item_stats_rsm (line 1098)

```latex
exttt{True}}, threshold estimates indicate the presence of disordered thresholds within the item \citep{Andrich2010}.\\
\midrule
\textcolor{gray}{\texttt{Observed gamma}} & Item-restscore gamma \citep{Kreiner2011}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Expected gamma}} & Expected item-restscore gamma. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Restscore p (BH)}} & Benjamini--Hochberg-adjusted p-value of the item-restscore test. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\midrule
\textcolor{gray}{\texttt{Q1}} & Conditional item chi-square \citep{vandenWollenberg1982}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Q1 df}} & Degrees of freedom of Q1. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}} and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Q1 p (BH)}} & Benjamini--Hochberg-adjusted p-value of Q1. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\midrule
\textcolor{gray}{\texttt{Infit p (Westfall-Young)}} & Westfall--Young bootstrap p-value of the infit mean square \citep{WestfallYoung1993}. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=...)}}.\\
\midrule
\textcolor{gray}{\texttt{Outfit p (Westfall-Young)}} & Westfall--Young bootstrap p-value of the outfit mean square. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=...)}}.\\
\midrule
\textcolor{gray}{\texttt{Target}} & Location of maximum item information. Only produced after \textcolor{gray}{\texttt{targeting}} (Section \ref{targeting_rsm}) and with \textcolor{gray}{\texttt{full=True}}.\\
\midrule
\textcolor{gray}{\texttt{Max information}} & Maximum item information. Only produced after \textcolor{gray}{\texttt{targeting}} and with \textcolor{gray}{\texttt{full=True}}.\\
\end{longtable}
```
*Was:*
```latex
exttt{True}}, threshold estimates indicate the presence of disordered thresholds within the item \citep{Andrich2010}.\\
\end{longtable}
```

### person_stats_rsm (line 1394)

```latex
ttt{Outfit MS}} & Outfit mean square. \\
\midrule
\textcolor{gray}{\texttt{Outfit Z}} & Standardised outfit z-score. \\
\midrule
\textcolor{gray}{\texttt{Conditional p}} & Probability, given the person's total, of a response pattern no more likely than theirs. Only produced after \textcolor{gray}{\texttt{fit\_\allowbreak statistics(conditional=True)}}.\\
\end{longtable}
```
*Was:*
```latex
ttt{Outfit MS}} & Outfit mean square. \\
\midrule
\textcolor{gray}{\texttt{Outfit Z}} & Standardised outfit z-score. \\
\end{longtable}
```

### icc (line 2180)

```latex
\midrule
\textcolor{gray}{\texttt{cond\_\allowbreak obs}} & Boolean: with \textcolor{gray}{\texttt{obs=True}}, also plots each class interval's mean expected score given each person's total and items (hollow diamonds), on the grouping of the conditional item chi-square (Section \ref{fit_statistics_rsm}). No effect if \textcolor{gray}{\texttt{obs=False}}. Default is \textcolor{gray}{\texttt{cond\_\allowbreak obs=False}}.\\
\midrule
\textcolor{gray}{\texttt{cond\_\allowbreak obs\_\allowbreak bars}} & Boolean: with \textcolor{gray}{\texttt{cond\_\allowbreak obs=True}}, bars of $ \pm 1.96 $ standard deviations of each class interval's observed mean under the model. Default is \textcolor{gray}{\texttt{cond\_\allowbreak obs\_\allowbreak bars=True}}.\\
\midrule
\textcolor{gray}{\texttt{obs\_\allowbreak groups}} & String, pandas series, list or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{obs=True}} and a single item, observed points split by group (a column of \textcolor{gray}{\texttt{self.exogenous}}, a series indexed by person or labels in response order), one colour and marker per group. Ignored for a list of items. Default is \textcolor{gray}{\texttt{obs\_\allowbreak groups=None}}.\\
```

### icc (line 2162)

```latex
no\_\allowbreak of\_\allowbreak classes=5,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~cond\_\allowbreak obs=False, cond\_\allowbreak obs\_\allowbreak bars=True, obs\_\allowbreak groups=None, 
```
*Was:*
```latex
no\_\allowbreak of\_\allowbreak classes=5, 
```

### item_stats_rsm (new section before it) (line 998)

```latex
\subsubsection{\textcolor{gray}{\texttt{fit\_\allowbreak statistics}}} \label{fit_statistics_rsm}

\textbf{Description}\\
Computes the fit statistics used by \textcolor{gray}{\texttt{item\_\allowbreak stats\_\allowbreak df}}, \textcolor{gray}{\texttt{person\_\allowbreak stats\_\allowbreak df}} and \textcolor{gray}{\texttt{test\_\allowbreak stats\_\allowbreak df}}: expected scores, residuals, and infit and outfit for items and persons. With \textcolor{gray}{\texttt{conditional=True}}, expected scores, variances and residuals are conditional on each person's total score over the items they responded to, and do not involve the person locations \citep{ChristensenKreiner2013}; the item-restscore test \citep{Kreiner2011}, the conditional item chi-square Q1 \citep{vandenWollenberg1982} and R1c \citep{Glas1988} on groups of persons, and each person's conditional p-value are also computed. With \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, p-values and cutoffs for item fit, item-restscore, Q1, Q3 and the first residual contrast also come from that many datasets simulated from the model given each person's total, each recalibrated with the object's settings, with Westfall--Young familywise p-values \citep{WestfallYoung1993}.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(warm\_\allowbreak corr=True, se=True, test\_\allowbreak stats=True, trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~tolerance=0.00001, max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , constant=None, threshold\_\allowbreak constant=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, no\_\allowbreak of\_\allowbreak samples=500, log\_\allowbreak lik\_\allowbreak tol=0.000001, interval=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~seed=None, conditional=False, conditional\_\allowbreak samples=1000,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~bootstrap\_\allowbreak samples=None, q1\_\allowbreak groups=None)}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{warm\_\allowbreak corr}} & Boolean: If \textcolor{gray}{\texttt{True}}, Warm's \parencite*{Warm1989} bias correction applied during generation of person estimates (see Section \ref{person_rsm}). Default \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}. \\
\midrule
\textcolor{gray}{\texttt{se}} & Boolean: if \textcolor{gray}{\texttt{False}}, skips the bootstrap standard errors (the fit statistics do not depend on them). Default is \textcolor{gray}{\texttt{se=True}}.\\
\midrule
\textcolor{gray}{\texttt{test\_\allowbreak stats}} & Boolean: if \textcolor{gray}{\texttt{True}}, also computes the test-level statistics (separation, strata, reliability; Section \ref{test_stats_rsm}). Requires \textcolor{gray}{\texttt{se=True}}. Default is \textcolor{gray}{\texttt{test\_\allowbreak stats=True}}.\\
\midrule
\textcolor{gray}{\texttt{trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict}} & Boolean: if \textcolor{gray}{\texttt{True}}, keeps category probabilities for persons with non-extreme scores only, to save memory. Default is \textcolor{gray}{\texttt{trim\_\allowbreak cat\_\allowbreak prob\_\allowbreak dict=False}}.\\
\midrule
\textcolor{gray}{\texttt{tolerance}} & Float: convergence stopping criterion for Newton-Raphson iterations during generation of person estimates (see Section \ref{person_rsm}). Default is \textcolor{gray}{\texttt{tolerance=0.00001}}. \\
\midrule
\textcolor{gray}{\texttt{max\_\allowbreak iters}} & Integer: maximum number of Newton-Raphson iterations during generation of person estimates (see Section \ref{person_rsm}). Default is \textcolor{gray}{\texttt{max\_\allowbreak iters=100}}. \\
\midrule
\textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} & Float: Value in range $ (0, 1) $ to ensure a estimate is returned  during generation of person estimates (see Section \ref{person_rsm}) if a person has an extreme score (all responses correct or incorrect). Adjusts the person's score to \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if zero) or \textcolor{gray}the maximum possible score minus \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if maximum score) before estimation. \\
\midrule
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of estimates from pairwise reciprocal matrix \citep{Elliott2022a} during item estimation (see section \ref{calibrate_rsm}). Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. Default is \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} \\
\midrule
\textcolor{gray}{\texttt{constant}} & Float: additive smoothing constant \citep{Elliott2022a} for item estimation (see section \ref{calibrate_rsm}). Default is \textcolor{gray}{\texttt{constant=None}}, which resolves to $ \min(1, 0.548\, I^{-0.2 \log_{10} Nm}) $, where $ N $ is the number of persons with at least one response, $ I $ is the number of items and $ m $ is the maximum score. Calibration uses the estimation method of the latest calibration (\textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} if none).\\
\midrule
\textcolor{gray}{\texttt{threshold\_\allowbreak constant}} & Float or \textcolor{gray}{\texttt{None}}: additive smoothing constant for the Stage 2 CPAT threshold distances (see Section \ref{calibrate_rsm}). \textcolor{gray}{\texttt{None}} (default) resolves to \textcolor{gray}{\texttt{0.03}}.\\
\midrule
\textcolor{gray}{\texttt{matrix\_\allowbreak power}} & Integer or \textcolor{gray}{\texttt{None}}: power to which conditional category response frequency matrix \citep[991]{Elliott2022a} is raised for generation of item location estimates. Each additional power increases the number of indirect pairwise comparisons \citep{Choppin1985, Elliott2022a}. Default is \textcolor{gray}{\texttt{matrix\_\allowbreak power=None}}, which resolves to \textcolor{gray}{\texttt{0}} (no matrix powering) when \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} and to \textcolor{gray}{\texttt{5}} for every other method. Passing \textcolor{gray}{\texttt{matrix\_\allowbreak power=0}} with a method other than \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} issues a \textcolor{gray}{\texttt{UserWarning}}.\\
\midrule
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} &  Integer: Number of bootstrap samples to generate for standard error estimates (see Section \ref{std_errors_rsm}). More samples lead to more accurate standard error estimates, but take correspondingly longer to compute. Unused if \textcolor{gray}{\texttt{se=False}}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}. \\
\midrule
\textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Float: convergence stopping criterion for log-likelihood method for generation of item location estimates (see Section \ref{calibrate_rsm}). Ignored for other methods. \\
\midrule
\textcolor{gray}{\texttt{interval}} & Float. Empirical interval to define quantiles of estimates from bootstrap samples, as an alternative to a confidence interval (see Section \ref{std_errors_rsm}). Defines a central interval of proportion $ p $, e.g.\ \textcolor{gray}{\texttt{interval=0.95}} defines quantiles at 2.5\% and 97.5\%. More stable with larger numbers of bootstrap samples. Ignored if \textcolor{gray}{\texttt{se=False}}. Default is \textcolor{gray}{\texttt{interval=None}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the bootstrap standard errors, the conditional p-values and the bootstrap datasets. \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\midrule
\textcolor{gray}{\texttt{conditional}} & Boolean: if \textcolor{gray}{\texttt{True}}, computes the statistics conditionally (see Description). Default is \textcolor{gray}{\texttt{conditional=False}}.\\
\midrule
\textcolor{gray}{\texttt{conditional\_\allowbreak samples}} & Integer: Monte Carlo samples for the persons' conditional p-values (\textcolor{gray}{\texttt{conditional=True}} only). Default is \textcolor{gray}{\texttt{conditional\_\allowbreak samples=1000}}.\\
\midrule
\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}} & Integer or \textcolor{gray}{\texttt{None}}: number of simulated datasets for the bootstrap p-values and cutoffs (see Description; at least 400 recommended). Requires \textcolor{gray}{\texttt{conditional=True}}. Default is \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples=None}}.\\
\midrule
\textcolor{gray}{\texttt{q1\_\allowbreak groups}} & Integer, pandas series, list or \textcolor{gray}{\texttt{None}}: groups for Q1 and R1c (\textcolor{gray}{\texttt{conditional=True}} only). \textcolor{gray}{\texttt{None}}: groups of about equal size ordered by person location, tied locations kept together (with complete data, groups of adjacent total scores), about one group per 60 persons, from 2 to 10, each of at least 30 persons. An integer sets the number of groups; labels per person (a series indexed by person, or a list in response order) supply them. Default is \textcolor{gray}{\texttt{q1\_\allowbreak groups=None}}.\\
\end{longtable}

\textbf{Returns}\\
\begin{longtable}{p{2.2in} p{3.8in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{self.item\_\allowbreak infit\_\allowbreak ms}} & pandas series: item infit mean squares; also \textcolor{gray}{\texttt{item\_\allowbreak outfit\_\allowbreak ms}}, \textcolor{gray}{\texttt{item\_\allowbreak infit\_\allowbreak zstd}}, \textcolor{gray}{\texttt{item\_\allowbreak outfit\_\allowbreak zstd}} and the person equivalents (\textcolor{gray}{\texttt{person\_\allowbreak infit\_\allowbreak ms}} etc.). Conditional with \textcolor{gray}{\texttt{conditional=True}}.\\
\midrule
\textcolor{gray}{\texttt{self.residual\_\allowbreak df}} & pandas dataframe: residuals; \textcolor{gray}{\texttt{std\_\allowbreak residual\_\allowbreak df}}: standardised residuals.\\
\midrule
\textcolor{gray}{\texttt{self.fit\_\allowbreak conditional}} & Boolean: whether the statistics are conditional.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak restscore\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item, observed and expected gamma, SE, p and Benjamini--Hochberg-adjusted p; with \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, also bootstrap and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak table}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item, Q1, df, p and Benjamini--Hochberg-adjusted p; with \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, also bootstrap and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak groups}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item and group, persons, observed and expected mean item score, standardised difference and the group's location range.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q1\_\allowbreak r1c}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): R1c, with rows \textcolor{gray}{\texttt{\textquotesingle item scores\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }} (Glas's category-level form, liberal when category counts within groups are small: use \textcolor{gray}{\texttt{\textquotesingle item scores\textquotesingle }}): statistic, df, p, number of groups and the estimates used.\\
\midrule
\textcolor{gray}{\texttt{self.person\_\allowbreak conditional\_\allowbreak p}} & pandas series (\textcolor{gray}{\texttt{conditional=True}}): each person's probability, given their total, of a response pattern no more likely than theirs.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak fit\_\allowbreak bootstrap}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item, infit and outfit mean squares, 95\% bands, p and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak bootstrap}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item pair, Q3, expected Q3, 95\% cutoff, p and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.residual\_\allowbreak contrast\_\allowbreak bootstrap}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): first residual contrast, its expected value, 95\% cutoff and p.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak fit\_\allowbreak bootstrap\_\allowbreak info}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): samples used, samples skipped and seed.\\
\end{longtable}

\textbf{Examples}\\
To compute conditional fit statistics:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(conditional=True)}}

To add bootstrap p-values from 1000 simulated datasets, reproducibly:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(conditional=True, bootstrap\_\allowbreak samples=1000, seed=1)}}

\newpage
```

### save_stats_rsm (new section before it) (line 1848)

```latex
\subsubsection{\textcolor{gray}{\texttt{local\_\allowbreak dependence\_\allowbreak gamma}}} \label{local_dependence_gamma_rsm}

\textbf{Description}\\
Tests every ordered pair of items for local dependence with the partial gamma \citep{KreinerChristensen2004}: Goodman and Kruskal's gamma between the two items' scores, counting concordant and discordant pairs of persons only within strata of the first item's rest score (the person's total over the other items they responded to, and which items those were). Under the model the partial gamma is zero; a positive value indicates local dependence. No parameter estimates are needed.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma(alpha=0.05, p\_\allowbreak method=\textquotesingle permutation\textquotesingle , samples=1000, seed=None)}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{alpha}} & Float: level for the \textcolor{gray}{\texttt{Flagged}} column (Benjamini--Hochberg-adjusted p-values). Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
\midrule
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle permutation\textquotesingle }}: normal p-value from the exact mean and variance of the concordant minus discordant count under permutation of the first item's responses within strata \citep{Kendall1970}; \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }}: p-value from \textcolor{gray}{\texttt{samples}} such permutations; \textcolor{gray}{\texttt{\textquotesingle asymptotic\textquotesingle }}: from the asymptotic standard error of the partial gamma \citep{Davis1967}, liberal with small samples or many missingness patterns (use \textcolor{gray}{\texttt{\textquotesingle permutation\textquotesingle }}). Default is \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle permutation\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{samples}} & Integer: permutations (\textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} only). Default is \textcolor{gray}{\texttt{samples=1000}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the permutations (\textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} only). Default is \textcolor{gray}{\texttt{seed=None}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma\_\allowbreak table}}, a pandas dataframe with one row per ordered item pair, sorted by gamma: \textcolor{gray}{\texttt{Item}} (the item whose rest score is conditioned on), \textcolor{gray}{\texttt{Paired item}}, \textcolor{gray}{\texttt{Gamma}}, \textcolor{gray}{\texttt{z}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{p (BH)}} and \textcolor{gray}{\texttt{Flagged}}. \textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma\_\allowbreak info}}: dictionary of \textcolor{gray}{\texttt{p\_\allowbreak method}}, \textcolor{gray}{\texttt{samples}} and \textcolor{gray}{\texttt{seed}}.\\

\textbf{Examples}\\
To test all item pairs for local dependence:\\
\textcolor{gray}{\texttt{self.local\_\allowbreak dependence\_\allowbreak gamma()}}

\newpage
\subsubsection{\textcolor{gray}{\texttt{targeting}}} \label{targeting_rsm}

\textbf{Description}\\
Targeting and measurement properties of the test (all items) in the study population \citep{KreinerChristensen2013}: a normal distribution of person locations is fitted to the observed total scores (each person's total over the items they responded to, item parameters fixed), and the test information, the exact bias, SEM and RMSE of the weighted likelihood estimates \citep{Warm1989} and reliabilities are computed over it.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.targeting(grid=(-6.0, 6.0, 0.05))}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{grid}} & Tuple: (from, to, step) of the locations for \textcolor{gray}{\texttt{self.targeting\_\allowbreak curves}}. Default is \textcolor{gray}{\texttt{grid=(-6.0, 6.0, 0.05)}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.targeting\_\allowbreak summary}}, a pandas series:

\begin{longtable}{p{2.2in} p{3.8in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{Population mean}} & Mean of the fitted normal distribution.\\
\midrule
\textcolor{gray}{\texttt{Population SD}} & SD of the fitted normal distribution.\\
\midrule
\textcolor{gray}{\texttt{Target}} & Location of maximum test information.\\
\midrule
\textcolor{gray}{\texttt{Target - population mean}} & Distance of the target from the population mean.\\
\midrule
\textcolor{gray}{\texttt{Maximum information}} & Test information at the target.\\
\midrule
\textcolor{gray}{\texttt{Expected information}} & Test information averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected / maximum information}} & Their ratio.\\
\midrule
\textcolor{gray}{\texttt{Test difficulty}} & Location at which the expected total is half the maximum.\\
\midrule
\textcolor{gray}{\texttt{Expected |bias|}} & Absolute bias of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected SEM}} & SEM of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{Expected RMSE}} & RMSE of the WLE averaged over the population.\\
\midrule
\textcolor{gray}{\texttt{rho1}} & $ V(T)/V(R) $: true-score over total-score variance.\\
\midrule
\textcolor{gray}{\texttt{rho2}} & $ \mathrm{corr}(R_1, R_2) $ for two independent administrations.\\
\midrule
\textcolor{gray}{\texttt{rho3}} & $ V(\theta)/V(\hat\theta) $.\\
\midrule
\textcolor{gray}{\texttt{rho4}} & $ \mathrm{corr}(\hat\theta_1, \hat\theta_2) $.\\
\midrule
\textcolor{gray}{\texttt{rho5}} & $ \mathrm{corr}(\theta, \hat\theta) $.\\
\midrule
\textcolor{gray}{\texttt{rho6}} & $ P(R_1 < R_2 \mid \theta_1 < \theta_2) $: probability that two random persons' totals order them correctly.\\
\end{longtable}

Also \textcolor{gray}{\texttt{self.targeting\_\allowbreak curves}} (by location: test information, asymptotic SE, bias, SEM and RMSE of the WLE, population density), \textcolor{gray}{\texttt{self.targeting\_\allowbreak score\_\allowbreak table}} (by total: WLE, model and observed proportions, observed among persons who responded to every item) and \textcolor{gray}{\texttt{self.targeting\_\allowbreak items}} (per item: target and maximum information).\\

\textbf{Examples}\\
To assess the targeting of the test:\\
\textcolor{gray}{\texttt{self.targeting()}}

\newpage
\subsubsection{\textcolor{gray}{\texttt{global\_\allowbreak stats\_\allowbreak df}}} \label{global_stats_rsm}

\textbf{Description}\\
Produces a table of whole-test results in the form of a pandas dataframe: global tests (R1c, Andersen's test, Martin-L\"of's test, the Wald test), the first residual contrast, the largest Q3*, the numbers of flagged item pairs, items and persons, and the targeting and reliability indices of \textcolor{gray}{\texttt{targeting}}. By default it collects the results already computed and lists the others as not run. With \textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests=True}}, it runs those not yet run, each with its own defaults unless overridden in \textcolor{gray}{\texttt{test\_\allowbreak kwargs}}; Andersen's and Martin-L\"of's tests are left out for estimates other than CML. Running the conditional fit this way makes the fit statistics in \textcolor{gray}{\texttt{self.item\_\allowbreak stats}} and \textcolor{gray}{\texttt{self.person\_\allowbreak stats}} conditional.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df(run\_\allowbreak all\_\allowbreak tests=False, test\_\allowbreak kwargs=None, alpha=0.05, dp=3, filename=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~file\_\allowbreak format=\textquotesingle csv\textquotesingle )}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests}} & Boolean: if \textcolor{gray}{\texttt{True}}, runs the analyses not yet run (see Description), including \textcolor{gray}{\texttt{targeting}}. Default is \textcolor{gray}{\texttt{run\_\allowbreak all\_\allowbreak tests=False}}.\\
\midrule
\textcolor{gray}{\texttt{test\_\allowbreak kwargs}} & Dictionary or \textcolor{gray}{\texttt{None}}: arguments for the analyses run, keyed by \textcolor{gray}{\texttt{\textquotesingle fit\_\allowbreak statistics\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle res\_\allowbreak corr\_\allowbreak analysis\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle local\_\allowbreak dependence\_\allowbreak gamma\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle targeting\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}. Default is \textcolor{gray}{\texttt{test\_\allowbreak kwargs=None}}.\\
\midrule
\textcolor{gray}{\texttt{alpha}} & Float: level for the counts of flagged items, item pairs and persons. Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
\midrule
\textcolor{gray}{\texttt{dp}} & Integer: Number of decimal places reported in output.\\
\midrule
\textcolor{gray}{\texttt{filename}} & String: The filename for the saved table, with no suffix for format. If \textcolor{gray}{\texttt{None}}, no file will be saved. Default is \textcolor{gray}{\texttt{filename=None}}.\\
\midrule
\textcolor{gray}{\texttt{file\_\allowbreak format}} & The format of the file: \textcolor{gray}{\texttt{csv}} or \textcolor{gray}{\texttt{xlsx}}. Default is \textcolor{gray}{\texttt{file\_\allowbreak format=csv}}.\\
\end{longtable}

\textbf{Returns}\\
Returned and stored as attribute \textcolor{gray}{\texttt{self.global\_\allowbreak stats}}, a pandas dataframe with one row per result and columns \textcolor{gray}{\texttt{Section}}, \textcolor{gray}{\texttt{Value}}, \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{p}} and \textcolor{gray}{\texttt{Details}} (the options defining the result, or the call needed to produce it).\\

\textbf{Examples}\\
To collect the results already computed:\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df()}}

To run all tests, with bootstrap p-values from 1000 simulated datasets:\\
\textcolor{gray}{\texttt{self.global\_\allowbreak stats\_\allowbreak df(run\_\allowbreak all\_\allowbreak tests=True, test\_\allowbreak kwargs=\{\textquotesingle fit\_\allowbreak statistics\textquotesingle : \{\textquotesingle bootstrap\_\allowbreak samples\textquotesingle : 1000, \textquotesingle seed\textquotesingle : 1\}\})}}

\newpage
```

### model_selection_rsm (line 1606)

```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05, min\_\allowbreak effect=0, sampling=\textquotesingle auto\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~likelihood=\textquotesingle conditional\textquotesingle , warm\_\allowbreak corr=True, tolerance=0.00001,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~threshold\_\allowbreak constant=None, method=\textquotesingle log-lik\textquotesingle , log\_\allowbreak lik\_\allowbreak tol=0.000001,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05, min\_\allowbreak effect=0,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~sampling=\textquotesingle dynamic\textquotesingle , warm\_\allowbreak corr=True, tolerance=0.00001, max\_\allowbreak iters=100,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None, threshold\_\allowbreak constant=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### model_selection_rsm (line 1605)

```latex
By default (\textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}) each model is scored by the conditional log-likelihood given each person's total score, after one Newton step from its estimates; \textcolor{gray}{\texttt{likelihood=\textquotesingle joint\textquotesingle }} uses the log-likelihood given the person estimates.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### model_selection_rsm (line 1632)

```latex
\textcolor{gray}{\texttt{sampling}} & \textcolor{gray}{\texttt{None}}, \textcolor{gray}{\texttt{\textquotesingle auto\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }}, or Integer: Subsamples $ T $ non-extreme persons before computing log-likelihoods: at very large $ N $, the LR/AIC/BIC comparison becomes biased toward the more complex (PCM) model, since log-likelihood differences scale with $ N $; capping the effective $ N $ mitigates this. An integer defines $ T $. \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} uses $ T = \min(20(I-1)(m-1), 1500) $, where $ I $ is the number of items and $ m $ the shared maximum score; subsampling is skipped whenever the actual non-extreme sample size is $ \leq T $. \textcolor{gray}{\texttt{None}} applies no subsampling. \textcolor{gray}{\texttt{\textquotesingle auto\textquotesingle }} resolves to \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} with \textcolor{gray}{\texttt{likelihood=\textquotesingle joint\textquotesingle }} and to \textcolor{gray}{\texttt{None}} with \textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}. Default is \textcolor{gray}{\texttt{sampling=\textquotesingle auto\textquotesingle }}.
```
*Was:*
```latex
\textcolor{gray}{\texttt{sampling}} & \textcolor{gray}{\texttt{None}}, \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }}, or Integer: Subsamples $ T $ non-extreme persons before computing log-likelihoods: at very large $ N $, the LR/AIC/BIC comparison becomes biased toward the more complex (PCM) model, since log-likelihood differences scale with $ N $; capping the effective $ N $ mitigates this. An integer defines $ T $. \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} uses $ T = \min(20(I-1)(m-1), 1500) $, where $ I $ is the number of items and $ m $ the shared maximum score; subsampling is skipped whenever the actual non-extreme sample size is $ \leq T $. \textcolor{gray}{\texttt{None}} applies no subsampling. Default is \textcolor{gray}{\texttt{sampling=\textquotesingle dynamic\textquotesingle }}.
```

### model_selection_rsm (line 1633)

```latex
\midrule
\textcolor{gray}{\texttt{likelihood}} & String: \textcolor{gray}{\texttt{\textquotesingle conditional\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle joint\textquotesingle }} (see Description). Default is \textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}.\\
```

### calibrate_rsm (line 435)

```latex
no\_\allowbreak info=\textquotesingle mean\textquotesingle , estimation=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~solver=None)}}\\
```
*Was:*
```latex
no\_\allowbreak info=\textquotesingle mean\textquotesingle , estimation=None, solver=None)}}\\
```

## mfrm.tex

### model_selection_mfrm (line 2396)

```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , models=None, aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05, min\_\allowbreak effect=0,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~sampling=\textquotesingle auto\textquotesingle , likelihood=\textquotesingle conditional\textquotesingle , warm\_\allowbreak corr=True,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~tolerance=0.00001, max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~constant=None, threshold\_\allowbreak constant=None, method=None, robust=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , models=None, aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~min\_\allowbreak effect=0, sampling=\textquotesingle dynamic\textquotesingle , warm\_\allowbreak corr=True, tolerance=0.00001,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5, constant=None, }\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~threshold\_\allowbreak constant=None,} method=None, robust=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### model_selection_mfrm (line 2395)

```latex
By default (\textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}) each model is scored by the conditional log-likelihood given each person's total over all the ratings they received (each facet element $ \times $ item treated as an item), after one Newton step from its estimates; \textcolor{gray}{\texttt{likelihood=\textquotesingle joint\textquotesingle }} uses the log-likelihood given the person estimates.\\

\textbf{Usage}
```
*Was:*
```latex
\textbf{Usage}
```

### model_selection_mfrm (line 2424)

```latex
\textcolor{gray}{\texttt{sampling}} & \textcolor{gray}{\texttt{None}}, \textcolor{gray}{\texttt{\textquotesingle auto\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }}, or Integer: subsamples non-extreme persons before computing log-likelihoods -- parameters are always estimated on the full data; only the log-likelihood evaluation itself is subsampled. Not primarily about speed: at very large $ N $, the LR/AIC/BIC comparison becomes biased toward the more complex facet element representation, since log-likelihood differences scale with $ N $; capping the effective $ N $ mitigates this. \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} uses $ T = \min(20R(I-1)(m-1), 1500) $, where $ I $ is the number of items, $ R $ the number of facet elements and $ m $ the maximum score ($ m-1 $ is taken as 1 for dichotomous data); an integer fixes $ T $ directly; \textcolor{gray}{\texttt{None}} disables subsampling. \textcolor{gray}{\texttt{\textquotesingle auto\textquotesingle }} resolves to \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} with \textcolor{gray}{\texttt{likelihood=\textquotesingle joint\textquotesingle }} and to \textcolor{gray}{\texttt{None}} with \textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}. Default is \textcolor{gray}{\texttt{sampling=\textquotesingle auto\textquotesingle }}.
```
*Was:*
```latex
\textcolor{gray}{\texttt{sampling}} & \textcolor{gray}{\texttt{None}}, \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }}, or Integer: subsamples non-extreme persons before computing log-likelihoods -- parameters are always estimated on the full data; only the log-likelihood evaluation itself is subsampled. Not primarily about speed: at very large $ N $, the LR/AIC/BIC comparison becomes biased toward the more complex facet element representation, since log-likelihood differences scale with $ N $; capping the effective $ N $ mitigates this. \textcolor{gray}{\texttt{\textquotesingle dynamic\textquotesingle }} uses $ T = \min(20R(I-1)(m-1), 1500) $, where $ I $ is the number of items, $ R $ the number of facet elements and $ m $ the maximum score ($ m-1 $ is taken as 1 for dichotomous data); an integer fixes $ T $ directly; \textcolor{gray}{\texttt{None}} disables subsampling. Default is \textcolor{gray}{\texttt{sampling=\textquotesingle dynamic\textquotesingle }}.
```

### model_selection_mfrm (line 2425)

```latex
\midrule
\textcolor{gray}{\texttt{likelihood}} & String: \textcolor{gray}{\texttt{\textquotesingle conditional\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle joint\textquotesingle }} (see Description). Default is \textcolor{gray}{\texttt{likelihood=\textquotesingle conditional\textquotesingle }}.\\
```

### item_stats_mfrm (new section before it) (line 2582)

```latex
\subsubsection{\textcolor{gray}{\texttt{fit\_\allowbreak statistics}}} \label{fit_statistics_mfrm}

\textbf{Description}\\
Computes the fit statistics used by \textcolor{gray}{\texttt{item\_\allowbreak stats\_\allowbreak df}}, \textcolor{gray}{\texttt{rater\_\allowbreak stats\_\allowbreak df}} and \textcolor{gray}{\texttt{person\_\allowbreak stats\_\allowbreak df}} for \textcolor{gray}{\texttt{model}}. With \textcolor{gray}{\texttt{conditional=True}}, expected scores, variances and residuals are conditional on each person's total over all the ratings they received, treating each facet element $ \times $ item as an item, and do not involve the person locations \citep{ChristensenKreiner2013}; item-restscore tests per item and per facet element and each person's conditional p-value are also computed. Threshold fit is unchanged. With \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, p-values and cutoffs for item and facet element fit, item-restscore, Q3 and the first residual contrast also come from that many datasets simulated from the model given each person's total, each recalibrated with the object's settings, with Westfall--Young familywise p-values.\\

\textbf{Usage}\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(model=\textquotesingle global\textquotesingle , anchors=None, warm\_\allowbreak corr=True, se=True, test\_\allowbreak stats=True,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~ext\_\allowbreak scores=True, tolerance=0.00001, max\_\allowbreak iters=100,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~ext\_\allowbreak score\_\allowbreak adjustment=0.5, method=None, constant=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~threshold\_\allowbreak constant=None, robust=False, matrix\_\allowbreak power=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, no\_\allowbreak of\_\allowbreak samples=500, interval=None, seed=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~conditional=False, conditional\_\allowbreak samples=1000, bootstrap\_\allowbreak samples=None)}}\\

\textbf{Arguments}
\begin{longtable}{p{1.5in} p{4.5in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{model}} & String: the facet element representation (Section \ref{calibrate_mfrm}). Default is \textcolor{gray}{\texttt{model=\textquotesingle global\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{anchors}} & List: If \textcolor{gray}{\texttt{None}}, unanchored central item location estimates and standard errors will be used. If a list of facet element names is passed, anchored estimates and standard errors, anchored to the facet elements passed, will be used. Other statistics are unaffected by \textcolor{gray}{\texttt{anchors}}. Default is \textcolor{gray}{\texttt{anchors=None}}. \\
\midrule
\textcolor{gray}{\texttt{warm\_\allowbreak corr}} & Boolean: If \textcolor{gray}{\texttt{True}}, Warm's \parencite*{Warm1989} bias correction applied during generation of person estimates (see Section \ref{person_mfrm}). Default is \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}. \\
\midrule
\textcolor{gray}{\texttt{se}} & Boolean: if \textcolor{gray}{\texttt{False}}, skips the bootstrap standard errors (the fit statistics do not depend on them). Default is \textcolor{gray}{\texttt{se=True}}.\\
\midrule
\textcolor{gray}{\texttt{test\_\allowbreak stats}} & Boolean: if \textcolor{gray}{\texttt{True}}, also computes the test-level statistics. Requires \textcolor{gray}{\texttt{se=True}}. Default is \textcolor{gray}{\texttt{test\_\allowbreak stats=True}}.\\
\midrule
\textcolor{gray}{\texttt{ext\_\allowbreak scores}} & Boolean: if \textcolor{gray}{\texttt{True}}, persons with extreme scores are included in the person estimates. Default is \textcolor{gray}{\texttt{ext\_\allowbreak scores=True}}.\\
\midrule
\textcolor{gray}{\texttt{tolerance}} & Float: convergence stopping criterion for Newton-Raphson iterations during generation of person estimates (see Section \ref{person_mfrm}). Default is \textcolor{gray}{\texttt{tolerance=0.00001}}. \\
\midrule
\textcolor{gray}{\texttt{max\_\allowbreak iters}} & Integer: maximum number of Newton-Raphson iterations during generation of person estimates (see Section \ref{person_mfrm}). Default is \textcolor{gray}{\texttt{max\_\allowbreak iters=100}}. \\
\midrule
\textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} & Float: Value in range $ (0, 1) $ to ensure a estimate is returned  during generation of person location estimates (see Section \ref{person_mfrm}) if a person has an extreme score (all items responded to are correct or incorrect). Since there is no finite ML location estimate for extreme scores, this adjusts the person's score to \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if zero) or \textcolor{gray}the maximum possible score minus \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} (if maximum score) before estimating location. This applies to maximum likelihood estimation only: with \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}, extreme scores have finite Warm estimates and the adjustment only sets the starting value. \\
\midrule
\textcolor{gray}{\texttt{method}} & String: method for derivation of vector of estimates from pairwise reciprocal matrix \citep{Elliott2022a} during item estimation (see section \ref{calibrate_mfrm}). Options are \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} for least squares \citep{Choppin1968, Choppin1985}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }} for the eigenvector method\citep{Garner2002}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} for cosine maximisation \citep{Kou2014} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} for the Bradley--Terry log-likelihood method \citep{Bradley1952}. \textcolor{gray}{\texttt{None}} (default) resolves per representation (see Section \ref{calibrate_mfrm}). \\
\midrule
\textcolor{gray}{\texttt{constant}} & Float or \textcolor{gray}{\texttt{None}}: additive smoothing constant \citep{Elliott2022a} for item estimation. \textcolor{gray}{\texttt{None}} (default) resolves to the simulation-tuned per-stage defaults (see Section \ref{calibrate_mfrm}). Calibration uses the estimation method of the latest calibration (\textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} if none).\\
\midrule
\textcolor{gray}{\texttt{threshold\_\allowbreak constant}} & Float or \textcolor{gray}{\texttt{None}}: smoothing constant for the threshold-location (CPAT) stage alone; \textcolor{gray}{\texttt{None}} (default) follows \textcolor{gray}{\texttt{constant}}. See Section \ref{calibrate_mfrm}.\\
\midrule
\textcolor{gray}{\texttt{robust}} & Boolean, default \textcolor{gray}{\texttt{False}}: selects the more conservative per-representation calibration for rough data. See Section \ref{calibrate_mfrm}.\\
\midrule
\textcolor{gray}{\texttt{matrix\_\allowbreak power}} & Integer or \textcolor{gray}{\texttt{None}}: power to which conditional category response frequency matrix \citep[991]{Elliott2022a} is raised for generation of item location estimates. Each additional power increases the number of indirect pairwise comparisons \citep{Choppin1985, Elliott2022a}. Default is \textcolor{gray}{\texttt{matrix\_\allowbreak power=None}}, which resolves to \textcolor{gray}{\texttt{0}} (no matrix powering) when \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }} and to \textcolor{gray}{\texttt{5}} for every other method. Passing \textcolor{gray}{\texttt{matrix\_\allowbreak power=0}} with a method other than \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} issues a \textcolor{gray}{\texttt{UserWarning}}.\\
\midrule
\textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Float: convergence stopping criterion for log-likelihood method for generation of item location estimates (see Section \ref{calibrate_mfrm}). Ignored for other methods. \\
\midrule
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} &  Integer: Number of bootstrap samples for standard error estimates (see Section \ref{std_errors_mfrm}). More samples lead to more accurate standard error estimates, but take longer to compute. Default \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}. \\
\midrule
\textcolor{gray}{\texttt{interval}} & Float. Empirical interval to define quantiles of estimates from bootstrap samples, as an alternative to a confidence interval (see Section \ref{std_errors_mfrm}). Defines a central interval of proportion $ p $, e.g.\ \textcolor{gray}{\texttt{interval=0.95}} defines quantiles at 2.5\% and 97.5\%. More stable with larger numbers of bootstrap samples. Default is \textcolor{gray}{\texttt{interval=None}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the bootstrap standard errors, the conditional p-values and the bootstrap datasets. \textcolor{gray}{\texttt{None}} (default) draws fresh entropy each call. See Section \ref{reproducibility}.\\
\midrule
\textcolor{gray}{\texttt{conditional}} & Boolean: if \textcolor{gray}{\texttt{True}}, computes the statistics conditionally (see Description). Default is \textcolor{gray}{\texttt{conditional=False}}.\\
\midrule
\textcolor{gray}{\texttt{conditional\_\allowbreak samples}} & Integer: Monte Carlo samples for the persons' conditional p-values (\textcolor{gray}{\texttt{conditional=True}} only). Default is \textcolor{gray}{\texttt{conditional\_\allowbreak samples=1000}}.\\
\midrule
\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}} & Integer or \textcolor{gray}{\texttt{None}}: number of simulated datasets for the bootstrap p-values (see Description; at least 400 recommended). Requires \textcolor{gray}{\texttt{conditional=True}}. Default is \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples=None}}.\\
\end{longtable}

\textbf{Returns}\\
\begin{longtable}{p{2.6in} p{3.4in}}
\toprule
\endfirsthead
\toprule
\endhead
\midrule\multicolumn{2}{r}{\textit{Continued on next page...}}\\
\endfoot
\bottomrule
\endlastfoot
\textcolor{gray}{\texttt{self.item\_\allowbreak infit\_\allowbreak ms\_\allowbreak \{model\}}} & pandas series: item infit mean squares, and the outfit, facet element and person equivalents, as in \textcolor{gray}{\texttt{item\_\allowbreak stats\_\allowbreak df}} (Section \ref{item_stats_mfrm}). Conditional with \textcolor{gray}{\texttt{conditional=True}}.\\
\midrule
\textcolor{gray}{\texttt{self.fit\_\allowbreak conditional\_\allowbreak \{model\}}} & Boolean: whether the statistics are conditional.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak restscore\_\allowbreak table\_\allowbreak \{model\}}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): per item, over facet elements, observed and expected item-restscore gamma \citep{Kreiner2011}, SE, p and Benjamini--Hochberg-adjusted p; with \textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}, also bootstrap and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.rater\_\allowbreak restscore\_\allowbreak table\_\allowbreak \{model\}}} & pandas dataframe (\textcolor{gray}{\texttt{conditional=True}}): the same per facet element, over items.\\
\midrule
\textcolor{gray}{\texttt{self.person\_\allowbreak conditional\_\allowbreak p\_\allowbreak \{model\}}} & pandas series (\textcolor{gray}{\texttt{conditional=True}}): each person's probability, given their total, of a rating pattern no more likely than theirs.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak fit\_\allowbreak bootstrap\_\allowbreak \{model\}}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item, infit and outfit mean squares, 95\% bands, p and Westfall--Young p \citep{WestfallYoung1993}; \textcolor{gray}{\texttt{self.rater\_\allowbreak fit\_\allowbreak bootstrap\_\allowbreak \{model\}}}: the same per facet element.\\
\midrule
\textcolor{gray}{\texttt{self.q3\_\allowbreak bootstrap\_\allowbreak \{model\}}} & pandas dataframe (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): per item pair, Q3, expected Q3, 95\% cutoff, p and Westfall--Young p.\\
\midrule
\textcolor{gray}{\texttt{self.residual\_\allowbreak contrast\_\allowbreak bootstrap\_\allowbreak \{model\}}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): first residual contrast, its expected value, 95\% cutoff and p.\\
\midrule
\textcolor{gray}{\texttt{self.fit\_\allowbreak bootstrap\_\allowbreak info\_\allowbreak \{model\}}} & Dictionary (\textcolor{gray}{\texttt{bootstrap\_\allowbreak samples}}): samples used, samples skipped and seed.\\
\end{longtable}

\textbf{Examples}\\
To compute conditional fit statistics for the \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }} model:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(model=\textquotesingle global\textquotesingle , conditional=True)}}

To add bootstrap p-values from 1000 simulated datasets, reproducibly:\\
\textcolor{gray}{\texttt{self.fit\_\allowbreak statistics(model=\textquotesingle global\textquotesingle , conditional=True, bootstrap\_\allowbreak samples=1000, seed=1)}}

\newpage
```

### item_res_corr_analysis_mfrm (line 3654)

```latex
& between item residuals (Q3; \citealp{Yen1984}). \\
```
*Was:*
```latex
& between item standardised residuals. \\
```

### item_res_corr_analysis_mfrm (line 3659)

```latex
r & \\
\textcolor{gray}{\texttt{self.item\_\allowbreak residual\_\allowbreak correlations\_\allowbreak bistretch}} & \\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q3\_\allowbreak star\_\allowbreak \{model\}}} & pandas dataframe: Q3* \citep{Christensen2017}, the item residual correlations minus their mean off-diagonal value.\\
\midrule
\textcolor{gray}{\texttt{self.item\_\allowbreak q3\_\allowbreak star\_\allowbreak max\_\allowbreak \{model\}}} & Float: the largest Q3*.\\
\end{longtable}
```
*Was:*
```latex
r & \\
\textcolor{gray}{\texttt{self.item\_\allowbreak residual\_\allowbreak correlations\_\allowbreak bistretch}} & \\
\end{longtable}
```


## model_selection (RSM/PCM rewrite, 2026-10-28)
Replaces parts of the RSM/PCM `model_selection` sections (rsm.tex line 1001, pcm.tex line 1108). The first Description paragraph is unchanged. Validation numbers stay out of the manual (CHANGELOG and docstrings only).

### model_selection_rsm (line 1001): Description, paragraphs 2 and 3 (replace)
```latex
Uses a likelihood ratio test, AIC, or BIC to determine which model is the more appropriate, parsimonious fit for a given data set. The comparison is symmetric: both \textcolor{gray}{\texttt{RSM.model\_\allowbreak selection}} and \textcolor{gray}{\texttt{PCM.model\_\allowbreak selection}} set the same \textcolor{gray}{\texttt{self.model\_\allowbreak comparison\_\allowbreak rsm\_\allowbreak pcm\_\allowbreak *}} attribute names regardless of which class the method is called from. The \textcolor{gray}{\texttt{PCM}} model is fitted to the same response data with the estimation \textcolor{gray}{\texttt{self}} was calibrated with, and the test follows that estimation; no person estimates are needed. Auto-triggers \textcolor{gray}{\texttt{calibrate}} (Section \ref{calibrate_rsm}) on \textcolor{gray}{\texttt{self}} if not run.\\

\begin{itemize}
\item \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}: the conditional log-likelihoods of the two CML fits are compared (with PCM's ridge, the unpenalised log-likelihoods at the ridge estimates); the LR test has $ (I-1)(K-1) $ degrees of freedom, the difference in free parameters ($ I $ items, maximum score $ K $). If either fit removes an item, a \textcolor{gray}{\texttt{ValueError}} is raised; use \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, which keeps every item.
\item \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}: both models are scored by the pairwise conditional likelihood of every item pair's response patterns, $ (a, b) $ against $ (a-1, b+1) $. Each person contributes to many pairs, so the LR statistic is referred to a weighted sum of $ \chi^2_1 $ variables (Kent, 1982), with weights from the likelihood's sensitivity $ H $ and its variability over persons $ J $ at the PCM fit, and a Satterthwaite \textit{p}-value. AIC and BIC replace the number of parameters by $ \textrm{tr}(J H^{-1}) $ (Varin and Vidoni, 2005).
\item PAIR with \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }}: the same test on PAIR's Bradley--Terry likelihood of those comparisons, with RSM as that likelihood's maximum under the rating scale restriction.
\item PAIR with \textcolor{gray}{\texttt{method=\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, and \textcolor{gray}{\texttt{estimation=\textquotesingle cpat-rcl\textquotesingle }}: there is no likelihood, so the \textit{p}-value is from a conditional parametric bootstrap: \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets are simulated given each person's total under the fitted RSM, PCM is refitted to each (with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} for \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}), and the sum of squared item-specific threshold deviations of the PCM fit is compared with its simulated values. Only \textcolor{gray}{\texttt{test=\textquotesingle LR\textquotesingle }} is available; AIC and BIC raise a \textcolor{gray}{\texttt{ValueError}}.
\end{itemize}

When a middle score category is rarely used or unused, the comparison is not reliable (it tends to favour PCM), and a \textcolor{gray}{\texttt{UserWarning}} names the categories.\\
```

### model_selection_rsm: Usage (replace)
```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05, min\_\allowbreak effect=0,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=199, constant=None, threshold\_\allowbreak constant=None, method=\textquotesingle log-lik\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### model_selection_rsm: Arguments (replace the sampling, likelihood, person-estimation, calibration and seed rows; no_of_samples goes after min_effect)
```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: number of simulated datasets for the bootstrap test (PAIR with \textcolor{gray}{\texttt{method=\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, or \textcolor{gray}{\texttt{estimation=\textquotesingle cpat-rcl\textquotesingle }}). Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=199}}.\\
\midrule
\textcolor{gray}{\texttt{sampling}}, \textcolor{gray}{\texttt{likelihood}}, \textcolor{gray}{\texttt{warm\_\allowbreak corr}}, & Deprecated: no effect, and raise a \textcolor{gray}{\texttt{FutureWarning}}. To be removed \\
\textcolor{gray}{\texttt{tolerance}}, \textcolor{gray}{\texttt{max\_\allowbreak iters}}, \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} & in a future release.\\
\midrule
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{threshold\_\allowbreak constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration keyword arguments, used only if \textcolor{gray}{\texttt{calibrate}} (Section \ref{calibrate_rsm}) has not already been run on \textcolor{gray}{\texttt{self}}. See Section \ref{calibrate_rsm} for details of each.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the bootstrap test's simulations. Pass an integer for a reproducible result; \textcolor{gray}{\texttt{None}} (default) draws fresh entropy. See Section \ref{reproducibility}.\\
```

### model_selection_rsm: Returns (add after the BIC row)
```latex
\midrule
\textcolor{gray}{\texttt{self.model\_\allowbreak comparison\_\allowbreak rsm\_\allowbreak pcm\_\allowbreak info}} & Dictionary: the test used (\textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle composite\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}), the estimation, and, for the composite test, the range of its $ \chi^2_1 $ weights.\\
\midrule
\textcolor{gray}{\texttt{self.model\_\allowbreak comparison\_\allowbreak rsm\_\allowbreak pcm\_\allowbreak effect}} & Float: the largest absolute difference, in logits, between an item's centred PCM thresholds and RSM's shared thresholds (the effect \textcolor{gray}{\texttt{min\_\allowbreak effect}} is compared with).\\
```

### model_selection_rsm: Example (replace the second example)
```latex
To run a likelihood ratio test (for an object calibrated with PAIR and \textcolor{gray}{\texttt{method=\textquotesingle cos\textquotesingle }}, the bootstrap test, here with 499 simulated datasets):

\textcolor{gray}{\texttt{my\_\allowbreak rsm.model\_\allowbreak selection(test=\textquotesingle LR\textquotesingle , no\_\allowbreak of\_\allowbreak samples=499, seed=1)}}
```

### model_selection_pcm (line 1108): Description, paragraphs 2 and 3 (replace)
```latex
Uses a likelihood ratio test, AIC, or BIC to determine which model is the more appropriate, parsimonious fit for a given data set. Requires uniform maximum scores across items (a precondition for RSM itself to apply at all). The \textcolor{gray}{\texttt{RSM}} model is fitted to the same response data with the estimation \textcolor{gray}{\texttt{self}} was calibrated with, and the test follows that estimation; no person estimates are needed. Auto-triggers \textcolor{gray}{\texttt{calibrate}} (Section \ref{calibrate_pcm}) on \textcolor{gray}{\texttt{self}} if not run.\\

\begin{itemize}
\item \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}: the conditional log-likelihoods of the two CML fits are compared (with PCM's ridge, the unpenalised log-likelihoods at the ridge estimates); the LR test has $ (I-1)(K-1) $ degrees of freedom, the difference in free parameters ($ I $ items, maximum score $ K $). If either fit removes an item, a \textcolor{gray}{\texttt{ValueError}} is raised; use \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, which keeps every item.
\item \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}: both models are scored by the pairwise conditional likelihood of every item pair's response patterns, $ (a, b) $ against $ (a-1, b+1) $. Each person contributes to many pairs, so the LR statistic is referred to a weighted sum of $ \chi^2_1 $ variables (Kent, 1982), with weights from the likelihood's sensitivity $ H $ and its variability over persons $ J $ at the PCM fit, and a Satterthwaite \textit{p}-value. AIC and BIC replace the number of parameters by $ \textrm{tr}(J H^{-1}) $ (Varin and Vidoni, 2005).
\item PAIR with \textcolor{gray}{\texttt{method=\textquotesingle log-lik\textquotesingle }}: the same test on PAIR's Bradley--Terry likelihood of those comparisons, with RSM as that likelihood's maximum under the rating scale restriction.
\item PAIR with \textcolor{gray}{\texttt{method=\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}: there is no likelihood, so the \textit{p}-value is from a conditional parametric bootstrap: \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets are simulated given each person's total under the fitted RSM, PCM is refitted to each, and the sum of squared item-specific threshold deviations of the PCM fit is compared with its simulated values. Only \textcolor{gray}{\texttt{test=\textquotesingle LR\textquotesingle }} is available; AIC and BIC raise a \textcolor{gray}{\texttt{ValueError}}.
\end{itemize}

When a middle score category is rarely used or unused, the comparison is not reliable (it tends to favour PCM), and a \textcolor{gray}{\texttt{UserWarning}} names the categories.\\
```

### model_selection_pcm: Usage (replace)
```latex
\textcolor{gray}{\texttt{self.model\_\allowbreak selection(test=\textquotesingle AIC\textquotesingle , aic\_\allowbreak sig\_\allowbreak test=True, alpha=0.05, min\_\allowbreak effect=0,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak of\_\allowbreak samples=199, constant=None, method=\textquotesingle log-lik\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### model_selection_pcm: Arguments (replace the sampling, likelihood, person-estimation, calibration and seed rows; no_of_samples goes after min_effect)
```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: number of simulated datasets for the bootstrap test (PAIR with \textcolor{gray}{\texttt{method=\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}). Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=199}}.\\
\midrule
\textcolor{gray}{\texttt{sampling}}, \textcolor{gray}{\texttt{likelihood}}, \textcolor{gray}{\texttt{warm\_\allowbreak corr}}, & Deprecated: no effect, and raise a \textcolor{gray}{\texttt{FutureWarning}}. To be removed \\
\textcolor{gray}{\texttt{tolerance}}, \textcolor{gray}{\texttt{max\_\allowbreak iters}}, \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} & in a future release.\\
\midrule
\textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Calibration keyword arguments, used only if \textcolor{gray}{\texttt{calibrate}} (Section \ref{calibrate_pcm}) has not already been run on \textcolor{gray}{\texttt{self}}. See Section \ref{calibrate_pcm} for details of each.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the bootstrap test's simulations. Pass an integer for a reproducible result; \textcolor{gray}{\texttt{None}} (default) draws fresh entropy. See Section \ref{reproducibility}.\\
```

### model_selection_pcm: Returns (add after the BIC row)
```latex
\midrule
\textcolor{gray}{\texttt{self.model\_\allowbreak comparison\_\allowbreak rsm\_\allowbreak pcm\_\allowbreak info}} & Dictionary: the test used (\textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle composite\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }}), the estimation, and, for the composite test, the range of its $ \chi^2_1 $ weights.\\
\midrule
\textcolor{gray}{\texttt{self.model\_\allowbreak comparison\_\allowbreak rsm\_\allowbreak pcm\_\allowbreak effect}} & Float: the largest absolute difference, in logits, between an item's centred PCM thresholds and RSM's shared thresholds (the effect \textcolor{gray}{\texttt{min\_\allowbreak effect}} is compared with).\\
```

### model_selection_pcm: Example (replace the second example)
```latex
To run a likelihood ratio test (for an object calibrated with PAIR and \textcolor{gray}{\texttt{method=\textquotesingle cos\textquotesingle }}, the bootstrap test, here with 499 simulated datasets):

\textcolor{gray}{\texttt{my\_\allowbreak pcm.model\_\allowbreak selection(test=\textquotesingle LR\textquotesingle , no\_\allowbreak of\_\allowbreak samples=499, seed=1)}}
```

### references.bib (check before adding)
```bibtex
@article{Kent1982, author={Kent, John T.}, title={Robust properties of likelihood ratio tests}, journal={Biometrika}, volume={69}, number={1}, pages={19--27}, year={1982}}
@article{Varin2005, author={Varin, Cristiano and Vidoni, Paolo}, title={A note on composite likelihood inference and model selection}, journal={Biometrika}, volume={92}, number={3}, pages={519--528}, year={2005}}
```

## global_fit: Andersen's test under cp-full-rcl (2026-10-29)
Same text in slm.tex, pcm.tex and rsm.tex (`global_fit_slm`, `_pcm`, `_rsm`; change the `std_errors_*` label). Replaces the Description paragraph on CML tests and the `method` and `p_method` rows, and adds a note to the `global_fit_item_table` row. Validation numbers stay out of the manual (CHANGELOG and docstrings only).

### Description: the CML-tests paragraph (replace)

```latex
For CML estimates (\textcolor{gray}{\texttt{calibrate(estimation=\textquotesingle cml\textquotesingle )}}), \textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} gives Andersen's conditional likelihood ratio test \citep{Andersen1973} between the groups, and \textcolor{gray}{\texttt{method=\textquotesingle martin-lof\textquotesingle }} Martin-L\"of's test of unidimensionality \citep{MartinLof1973} for two item subsets; the Wald test then uses the CML covariances (Section \ref{std_errors_slm}). For \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates (the default), \textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} gives Andersen's test on the pairwise conditional likelihood those estimates maximise: the groups and their pooled persons are refitted with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, and by default the p-value comes from datasets simulated from the pooled fit given each person's total, with every fit repeated, for the composite likelihood ratio adjusted for persons appearing in many item pairs \citep{PaceSalvanSartori2011}. \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle chi2\textquotesingle }} gives a faster p-value from the unadjusted ratio \citep{Kent1982}; for PCM it rejects too often, and a \textcolor{gray}{\texttt{UserWarning}} says so. CML's test has more power with a few hundred persons, so calibrate with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }} when power matters most.
```

### Arguments: `method` row (replace the description)

```latex
\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}. \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} needs CML or \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates, \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }} CML estimates.
```

### Arguments: `p_method` row (replace)

```latex
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle chi2\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} (\textcolor{gray}{\texttt{method=\textquotesingle andersen-lr\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle martin-lof\textquotesingle }}): the chi-square p-value, or a Monte Carlo p-value from \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets simulated given each person's total. \textcolor{gray}{\texttt{None}} chooses \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }} for \textcolor{gray}{\texttt{\textquotesingle andersen-lr\textquotesingle }} with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates or PCM's CML ridge (stopping once 10 simulated statistics reach the observed one), and \textcolor{gray}{\texttt{\textquotesingle chi2\textquotesingle }} otherwise. Default is \textcolor{gray}{\texttt{p\_\allowbreak method=None}}.\\
```

### Returns: `global_fit_item_table` row (append)

```latex
\textcolor{gray}{\texttt{None}} for \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates.
```

### Returns: `global_fit_summary` row (append)

```latex
For \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates it also holds the composite likelihood ratio, its adjusted value (the statistic the Monte Carlo p-value refers to) and the chi-square p-value.\n```\n\n### references.bib (add)

```bibtex
@article{PaceSalvanSartori2011, author={Pace, Luigi and Salvan, Alessandra and Sartori, Nicola}, title={Adjusting composite likelihood ratio statistics}, journal={Statistica Sinica}, volume={21}, number={1}, pages={129--148}, year={2011}}
```

## loaders.tex: missing codes, sheets, separators, facet name, JSON layouts, warnings (2026-11-01)

### Usage lines (replace)

```latex
\textcolor{gray}{\texttt{loadup\_\allowbreak slm(filename, item\_\allowbreak names=True, person\_\allowbreak names=True, long=False, missing=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~sheet\_\allowbreak name=0, sep=\textquotesingle ,\textquotesingle )}}\\
```

```latex
\textcolor{gray}{\texttt{loadup\_\allowbreak pcm(filename, max\_\allowbreak score\_\allowbreak vector=None, item\_\allowbreak names=True, person\_\allowbreak names=True,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~long=False, missing=None, sheet\_\allowbreak name=0, sep=\textquotesingle ,\textquotesingle )}}\\
```

```latex
\textcolor{gray}{\texttt{loadup\_\allowbreak rsm(filename, max\_\allowbreak score=None, item\_\allowbreak names=True, person\_\allowbreak names=True,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~long=False, missing=None, sheet\_\allowbreak name=0, sep=\textquotesingle ,\textquotesingle )}}\\
```

```latex
\textcolor{gray}{\texttt{loadup\_\allowbreak mfrm\_\allowbreak single(filename, max\_\allowbreak score=None, item\_\allowbreak names=True, long=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~missing=None, facet\_\allowbreak name=\textquotesingle Rater\textquotesingle , sheet\_\allowbreak name=0, sep=\textquotesingle ,\textquotesingle )}}
```

```latex
\textcolor{gray}{\texttt{loadup\_\allowbreak mfrm\_\allowbreak xlsx\_\allowbreak tabs(filename, max\_\allowbreak score, item\_\allowbreak names=True, missing=None, long=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~facet\_\allowbreak name=\textquotesingle Rater\textquotesingle )}}
```

```latex
\textcolor{gray}{\texttt{loadup\_\allowbreak mfrm\_\allowbreak multiple(filename\_\allowbreak dict, max\_\allowbreak score, item\_\allowbreak names=True, missing=None, long=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~facet\_\allowbreak name=\textquotesingle Rater\textquotesingle , sep=\textquotesingle ,\textquotesingle )}}
```

```latex
\textcolor{gray}{\texttt{loadup\_\allowbreak exogenous(filename, person\_\allowbreak names=True, sheet\_\allowbreak name=0, sep=\textquotesingle ,\textquotesingle )}}\\
```

### Arguments: new rows

`missing` row: add after the `long` row in `loadup_slm`, `loadup_pcm`, `loadup_rsm` and `loadup_mfrm_single`; replace the existing `missing` row in `loadup_mfrm_xlsx_tabs` and `loadup_mfrm_multiple`.

```latex
\midrule
\textcolor{gray}{\texttt{missing}} & String, int, float, list or \textcolor{gray}{\texttt{None}}: value(s) to treat as missing, e.g.\ \textcolor{gray}{\texttt{99}}, \textcolor{gray}{\texttt{-9}} or \textcolor{gray}{\texttt{\textquotesingle .\textquotesingle }}. A number matches however it is written in the file (\textcolor{gray}{\texttt{99}}, \textcolor{gray}{\texttt{99.0}} or \textcolor{gray}{\texttt{\textquotesingle 99\textquotesingle }}). Default is \textcolor{gray}{\texttt{missing=None}}.\\
```

`facet_name` row: add after the `missing` row in the three MFRM loaders.

```latex
\midrule
\textcolor{gray}{\texttt{facet\_\allowbreak name}} & String: the name of the facet, used as the name of the first index level (and, for \textcolor{gray}{\texttt{loadup\_\allowbreak mfrm\_\allowbreak single}} in long format, the name of the facet column). Default is \textcolor{gray}{\texttt{facet\_\allowbreak name=\textquotesingle Rater\textquotesingle }}.\\
```

`sheet_name` row: add last in `loadup_slm`, `loadup_pcm`, `loadup_rsm`, `loadup_mfrm_single` and `loadup_exogenous`.

```latex
\midrule
\textcolor{gray}{\texttt{sheet\_\allowbreak name}} & String or int: xlsx files only, the sheet to read, by name or position. Default is \textcolor{gray}{\texttt{sheet\_\allowbreak name=0}} (the first sheet).\\
```

`sep` row: add last in `loadup_slm`, `loadup_pcm`, `loadup_rsm`, `loadup_mfrm_single`, `loadup_mfrm_multiple` and `loadup_exogenous`.

```latex
\midrule
\textcolor{gray}{\texttt{sep}} & String or \textcolor{gray}{\texttt{None}}: delimited text files only, the column separator, e.g.\ \textcolor{gray}{\texttt{\textquotesingle ;\textquotesingle }}. \textcolor{gray}{\texttt{None}} detects it. A \textcolor{gray}{\texttt{.tsv}} file is read with tabs unless \textcolor{gray}{\texttt{sep}} is given. Default is \textcolor{gray}{\texttt{sep=\textquotesingle ,\textquotesingle }}.\\
```

`max_score` row in `loadup_mfrm_xlsx_tabs` and `loadup_mfrm_multiple` (replace: the argument is required there, with no default)

```latex
\textcolor{gray}{\texttt{max\_\allowbreak score}} & The maximum score available.\\
```

### Input file format paragraph (SLM, PCM, RSM, mfrm_single): append

```latex
A json file can be a list of records, one per person (one per rater and person for \textcolor{gray}{\texttt{loadup\_\allowbreak mfrm\_\allowbreak single}}), with the person names as the first field, or an object of columns, \textcolor{gray}{\texttt{\{item: \{person: score\}\}}}, as written by pandas' \textcolor{gray}{\texttt{DataFrame.to\_\allowbreak json()}}. Values that are not valid scores (whole numbers from 0 to the maximum score) are set to missing, with a \textcolor{gray}{\texttt{UserWarning}} naming them and the items they occur in; pass missing-value codes as \textcolor{gray}{\texttt{missing}}. In long format, a response that appears more than once keeps the lowest score, with a \textcolor{gray}{\texttt{UserWarning}}.
```

### Input file format paragraph (mfrm_xlsx_tabs, mfrm_multiple): append

```latex
Values that are not valid scores (whole numbers from 0 to \textcolor{gray}{\texttt{max\_\allowbreak score}}) are set to missing, with a \textcolor{gray}{\texttt{UserWarning}} naming them and the items they occur in; pass missing-value codes as \textcolor{gray}{\texttt{missing}}. In long format, each sheet or file has the columns \textcolor{gray}{\texttt{Person}}, \textcolor{gray}{\texttt{Item}} and \textcolor{gray}{\texttt{Score}}, and a response that appears more than once keeps the lowest score, with a \textcolor{gray}{\texttt{UserWarning}}.
```

### Returns (SLM, PCM, RSM): first item (replace)

```latex
    \item pandas dataframe of responses in required format for class \textcolor{gray}{\texttt{SLM}}, with the index named \textcolor{gray}{\texttt{Person}}
```
(`PCM`/`RSM` in their sections.) MFRM loaders: the index levels are named `facet_name` and `Person`. `loadup_exogenous`: the index is named `Person`.

## references.bib: new entries (from memory: check before adding)

```bibtex
@article{Andersen1970, author={Andersen, Erling B.}, title={Asymptotic properties of conditional maximum-likelihood estimators}, journal={Journal of the Royal Statistical Society: Series B}, volume={32}, number={2}, pages={283--301}, year={1970}}
@article{Andersen1973, author={Andersen, Erling B.}, title={A goodness of fit test for the {R}asch model}, journal={Psychometrika}, volume={38}, number={1}, pages={123--140}, year={1973}}
@book{MartinLof1973, author={Martin-L{\"o}f, Per}, title={Statistiska modeller}, publisher={Institutet f{\"o}r f{\"o}rs{\"a}kringsmatematik och matematisk statistik vid Stockholms universitet}, address={Stockholm}, year={1973}}
@article{Glas1988, author={Glas, Cees A. W.}, title={The derivation of some tests for the {R}asch model from the multinomial distribution}, journal={Psychometrika}, volume={53}, number={4}, pages={525--546}, year={1988}}
@article{vandenWollenberg1982, author={van den Wollenberg, Arnold L.}, title={Two new test statistics for the {R}asch model}, journal={Psychometrika}, volume={47}, number={2}, pages={123--140}, year={1982}}
@article{Kreiner2011, author={Kreiner, Svend}, title={A note on item--restscore association in {R}asch models}, journal={Applied Psychological Measurement}, volume={35}, number={7}, pages={557--561}, year={2011}}
@article{KreinerChristensen2004, author={Kreiner, Svend and Christensen, Karl Bang}, title={Analysis of local dependence and multidimensionality in graphical loglinear {R}asch models}, journal={Communications in Statistics -- Theory and Methods}, volume={33}, number={6}, pages={1239--1276}, year={2004}}
@incollection{ChristensenKreiner2013, author={Christensen, Karl Bang and Kreiner, Svend}, title={Item fit statistics}, booktitle={Rasch Models in Health}, editor={Christensen, Karl Bang and Kreiner, Svend and Mesbah, Mounir}, publisher={ISTE and Wiley}, address={London and Hoboken, NJ}, year={2013}}
@incollection{KreinerChristensen2013, author={Kreiner, Svend and Christensen, Karl Bang}, title={Person parameter estimation and measurement in {R}asch models}, booktitle={Rasch Models in Health}, editor={Christensen, Karl Bang and Kreiner, Svend and Mesbah, Mounir}, publisher={ISTE and Wiley}, address={London and Hoboken, NJ}, year={2013}, chapter={4}}
@article{Christensen2017, author={Christensen, Karl Bang and Makransky, Guido and Horton, Mike}, title={Critical values for {Y}en's {Q}3: Identification of local dependence in the {R}asch model using residual correlations}, journal={Applied Psychological Measurement}, volume={41}, number={3}, pages={178--194}, year={2017}}
@article{Yen1984, author={Yen, Wendy M.}, title={Effects of local item dependence on the fit and equating performance of the three-parameter logistic model}, journal={Applied Psychological Measurement}, volume={8}, number={2}, pages={125--145}, year={1984}}
@book{WestfallYoung1993, author={Westfall, Peter H. and Young, S. Stanley}, title={Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment}, publisher={Wiley}, address={New York}, year={1993}}
@article{Davis1967, author={Davis, James A.}, title={A partial coefficient for {G}oodman and {K}ruskal's gamma}, journal={Journal of the American Statistical Association}, volume={62}, number={317}, pages={189--193}, year={1967}}
@book{Kendall1970, author={Kendall, Maurice G.}, title={Rank Correlation Methods}, edition={4}, publisher={Griffin}, address={London}, year={1970}}
```

# Manual changes for v2.0.0, 2026-11-05 (not yet in Overleaf)

Current local files against the versions before today's edits. Line numbers are in the current files. For each change: the section, the new text, then *Was:* (omitted for insertions).

## slm.tex

### calibrate_slm (line 451)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~estimation=None, solver=None, ridge=0, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~estimation=None, solver=None, ridge=0)}}\\
```

### calibrate_slm (line 476)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the CML fit. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### std_errors_slm (line 537)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~seed=None, se\_\allowbreak method=None, no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~seed=None, se\_\allowbreak method=None, no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle )}}\\
```

### std_errors_slm (line 566)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### calibrate_anchor_slm (line 626)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, fix\_\allowbreak anchors=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, fix\_\allowbreak anchors=None)}}\\
```

### calibrate_anchor_slm (line 668)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the fit. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### fit_statistics_slm (line 970)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~q1\_\allowbreak groups=None, q1\_\allowbreak overall\_\allowbreak samples=99, q1\_\allowbreak overall\_\allowbreak exceedances=10, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~q1\_\allowbreak groups=None, q1\_\allowbreak overall\_\allowbreak samples=99, q1\_\allowbreak overall\_\allowbreak exceedances=10)}}\\
```

### fit_statistics_slm (line 1021)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### global_fit_slm (line 1394)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=None, mc\_\allowbreak exceedances=None, unused\_\allowbreak categories=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=None, mc\_\allowbreak exceedances=None, unused\_\allowbreak categories=None)}}\\
```

### global_fit_slm (line 1428)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and Monte Carlo draws. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### dif_test_slm (line 1889)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None)}}\\
```

### dif_test_slm (line 1951)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### recovery_stats_slm (line 2009)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~plot=True, **plot\_\allowbreak kwargs, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~plot=True, **plot\_\allowbreak kwargs)}}\\
```

### recovery_stats_slm (line 2033)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the simulated datasets. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

## pcm.tex

### calibrate_pcm (line 571)

```latex
The method above is \textcolor{gray}{\texttt{estimation=\textquotesingle pair\textquotesingle }}, the default when \textcolor{gray}{\texttt{method}} is \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. With \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} (the default otherwise), uncentred thresholds maximise the pairwise conditional likelihood: every item pair and every pair of response patterns with the same pair total, $ (a, b) $ against $ (a-1, b+1) $, is a comparison of two thresholds, and the likelihood of all of them is maximised from the PAIR estimates. This generalises the Bradley--Terry likelihood \citep{Bradley1952, Zermelo1929} (which it reduces to for dichotomous items, as in the SLM's \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}) to polytomous items. \textcolor{gray}{\texttt{estimation=\textquotesingle cpat-rcl\textquotesingle }} restricts it to CPAT's patterns, the pattern pairs whose two thresholds have the same number or adjacent numbers. With \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, uncentred thresholds are conditional maximum likelihood (CML) estimates \citep{Andersen1970}, which maximise the likelihood of the responses given each person's total score over the items they responded to, starting from the \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates (from the PAIR estimates with \textcolor{gray}{\texttt{ridge=0}} or \textcolor{gray}{\texttt{unused\_\allowbreak categories=\textquotesingle null\textquotesingle }}). For both, \textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}} and \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} apply only to the PAIR estimates they start from (and, through them, to the default ridge weight below).
```
*Was:*
```latex
The method above is \textcolor{gray}{\texttt{estimation=\textquotesingle pair\textquotesingle }}, the default when \textcolor{gray}{\texttt{method}} is \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}. With \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} (the default otherwise), uncentred thresholds maximise the pairwise conditional likelihood: every item pair and every pair of response patterns with the same pair total, $ (a, b) $ against $ (a-1, b+1) $, is a comparison of two thresholds, and the likelihood of all of them is maximised from the PAIR estimates. This generalises the Bradley--Terry likelihood \citep{Bradley1952, Zermelo1929} (which it reduces to for dichotomous items, as in the SLM's \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}) to polytomous items. With \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, uncentred thresholds are conditional maximum likelihood (CML) estimates \citep{Andersen1970}, which maximise the likelihood of the responses given each person's total score over the items they responded to, starting from the \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} estimates (from the PAIR estimates with \textcolor{gray}{\texttt{ridge=0}} or \textcolor{gray}{\texttt{unused\_\allowbreak categories=\textquotesingle null\textquotesingle }}). For both, \textcolor{gray}{\texttt{constant}}, \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}} and \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} apply only to the PAIR estimates they start from (and, through them, to the default ridge weight below).
```

### calibrate_pcm (line 577)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~estimation=None, solver=None, unused\_\allowbreak categories=None, ridge=\textquotesingle minimise\_\allowbreak bias\textquotesingle , progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~estimation=None, solver=None, unused\_\allowbreak categories=None, ridge=\textquotesingle minimise\_\allowbreak bias\textquotesingle )}}\\
```

### calibrate_pcm (line 597)

```latex
\textcolor{gray}{\texttt{estimation}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pair\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} (see Description). Default is \textcolor{gray}{\texttt{estimation=None}}, which resolves to \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, or to \textcolor{gray}{\texttt{\textquotesingle pair\textquotesingle }} when \textcolor{gray}{\texttt{method}} is \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{estimation}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pair\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} (see Description). Default is \textcolor{gray}{\texttt{estimation=None}}, which resolves to \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, or to \textcolor{gray}{\texttt{\textquotesingle pair\textquotesingle }} when \textcolor{gray}{\texttt{method}} is \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}.\\
```

### calibrate_pcm (line 604)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the CML fit. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### std_errors_pcm (line 675)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~seed=None, se\_\allowbreak method=None, no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~seed=None, se\_\allowbreak method=None, no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle )}}\\
```

### std_errors_pcm (line 704)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### calibrate_anchor_pcm (line 783)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, fix\_\allowbreak anchors=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, fix\_\allowbreak anchors=None)}}\\
```

### calibrate_anchor_pcm (line 836)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the fit. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### model_selection_pcm (line 1139)

```latex
\item \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}: the same composite likelihood ratio test, on CPAT's patterns only (the pattern pairs whose two thresholds have the same number or adjacent numbers).
```

### model_selection_pcm (line 1151)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### model_selection_pcm (line 1178)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the fits and bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### fit_statistics_pcm (line 1238)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~q1\_\allowbreak groups=None, q1\_\allowbreak overall\_\allowbreak samples=99, q1\_\allowbreak overall\_\allowbreak exceedances=10, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~q1\_\allowbreak groups=None, q1\_\allowbreak overall\_\allowbreak samples=99, q1\_\allowbreak overall\_\allowbreak exceedances=10)}}\\
```

### fit_statistics_pcm (line 1289)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### global_fit_pcm (line 1846)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=None, mc\_\allowbreak exceedances=None, unused\_\allowbreak categories=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=None, mc\_\allowbreak exceedances=None, unused\_\allowbreak categories=None)}}\\
```

### global_fit_pcm (line 1880)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and Monte Carlo draws. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### dif_test_pcm (line 2346)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None)}}\\
```

### dif_test_pcm (line 2409)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### recovery_stats_pcm (line 2480)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~plot=True, **plot\_\allowbreak kwargs, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~plot=True, **plot\_\allowbreak kwargs)}}\\
```

### recovery_stats_pcm (line 2503)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the simulated datasets. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

## rsm.tex

### calibrate_rsm (line 405)

```latex
The default, \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, is regularised composite likelihood (RCL). For two items, every pair of response patterns $ (a, b) $ and $ (a-1, b+1) $ with the same total is a binomial comparison whose log-odds depend only on the two items' locations and two thresholds. This generalises the Bradley--Terry likelihood \citep{Bradley1952, Zermelo1929} (which it reduces to for dichotomous items, as in the SLM's \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}) to polytomous items. Both fit all their patterns jointly, as one composite likelihood, by Newton-Raphson, with additive smoothing constants on the item comparisons and CPAT threshold patterns ($ (k, k) $ against $ (k-1, k+1) $ on two items). A category width with fewer than 20 threshold patterns, or with all its patterns on one side, is pulled toward the mean of the other widths by a ridge of $ 0.5\,(1 - T/20) $ ($ 0.5 $ if one-sided), where $ T $ is its pattern count. The widths next to an empty middle category have no response patterns of their own (every pattern that would inform them includes the empty category): they are left out of the joint fit and set by CPAT's threshold estimator (\textcolor{gray}{\texttt{threshold\_\allowbreak method=\textquotesingle cpat-cl\textquotesingle }}), given the item locations from the joint fit, which smooths them heavily.
```
*Was:*
```latex
The default, \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, is regularised composite likelihood (RCL). For two items, every pair of response patterns $ (a, b) $ and $ (a-1, b+1) $ with the same total is a binomial comparison whose log-odds depend only on the two items' locations and two thresholds. This generalises the Bradley--Terry likelihood \citep{Bradley1952, Zermelo1929} (which it reduces to for dichotomous items, as in the SLM's \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}) to polytomous items. Both fit all their patterns jointly, as one composite likelihood, by Newton-Raphson, with additive smoothing constants on the item comparisons and CPAT threshold patterns ($ (k, k) $ against $ (k-1, k+1) $ on two items). A category width with fewer than 20 threshold patterns, or with all its patterns on one side, is pulled toward the mean of the other widths by a ridge of $ 0.5\,(1 - T/20) $ ($ 0.5 $ if one-sided), where $ T $ is its pattern count. The widths next to an empty middle category are left out of the joint fit and estimated separately by conditional maximum likelihood from their threshold patterns (\textcolor{gray}{\texttt{threshold\_\allowbreak method=\textquotesingle cpat-cml\textquotesingle }}), given the item locations from the joint fit.
```

### calibrate_rsm (line 436)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~solver=None, unused\_\allowbreak categories=\textquotesingle drop\textquotesingle , progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~solver=None, unused\_\allowbreak categories=\textquotesingle drop\textquotesingle )}}\\
```

### calibrate_rsm (line 459)

```latex
\textcolor{gray}{\texttt{threshold\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} only. Estimator for the threshold stage. \textcolor{gray}{\texttt{\textquotesingle cpat-cl\textquotesingle }}: for each distance between adjacent thresholds, the maximum of the composite likelihood of the threshold patterns ($ (k, k) $ against $ (k-1, k+1) $ on two items) over item pairs, given the item locations. \textcolor{gray}{\texttt{\textquotesingle cpat-info-weighted\textquotesingle }}: the information-weighted mean of the item pairs' log-odds \citep{Elliott2022a}. Default \textcolor{gray}{\texttt{threshold\_\allowbreak method=None}} resolves to \textcolor{gray}{\texttt{\textquotesingle cpat-cl\textquotesingle }} (called \textcolor{gray}{\texttt{\textquotesingle cpat-cml\textquotesingle }} before v2.0.0; still accepted, with a warning).\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{threshold\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} only. Estimator for the threshold stage, \textcolor{gray}{\texttt{\textquotesingle cpat-cml\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-info-weighted\textquotesingle }} \citep{Elliott2022a}. Default \textcolor{gray}{\texttt{threshold\_\allowbreak method=None}} resolves to \textcolor{gray}{\texttt{\textquotesingle cpat-cml\textquotesingle }}.\\
```

### calibrate_rsm (line 468)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the CML fit. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### std_errors_rsm (line 552)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~se\_\allowbreak method=None, no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~se\_\allowbreak method=None)}}\\
```

### std_errors_rsm (line 581)

```latex
\midrule
\textcolor{gray}{\texttt{no\_\allowbreak data\_\allowbreak se}} & String: for the thresholds and category widths the data do not inform (listed in \textcolor{gray}{\texttt{self.no\_\allowbreak data\_\allowbreak parameters}}), \textcolor{gray}{\texttt{\textquotesingle posterior\textquotesingle }}, the standard error from the information including the ridge, or \textcolor{gray}{\texttt{\textquotesingle nan\textquotesingle }}, which leaves them out of confidence intervals and tests. Those set by a rule outside the likelihood (an empty score category) get \textcolor{gray}{\texttt{NaN}} either way. Default is \textcolor{gray}{\texttt{no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### calibrate_anchor_rsm (line 674)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~estimation=None, threshold\_\allowbreak method=None, fix\_\allowbreak anchors=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~estimation=None, threshold\_\allowbreak method=None, fix\_\allowbreak anchors=None)}}\\
```

### calibrate_anchor_rsm (line 725)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the fit. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### model_selection_rsm (line 1018)

```latex
\item \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}: the same composite likelihood ratio test, on CPAT's patterns only (the pattern pairs whose two thresholds have the same number or adjacent numbers).
```

### model_selection_rsm (line 1020)

```latex
\item \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}: these have no likelihood, so the test is a conditional parametric bootstrap \citep{Davison1997}. \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets are simulated from RSM's \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} fit, keeping each person's total score and the items they responded to. The statistic compares the two fitted models: each item's thresholds are taken relative to that item's central location in the RSM fit and in the PCM fit, and the squared differences between the two are summed over items and thresholds. It is computed for the fits to the data and recomputed on each simulated dataset with both models refitted (PCM with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} when \textcolor{gray}{\texttt{self}} was calibrated with \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, which has no PCM version). The \textit{p}-value is (1 + the number of simulated statistics at least as large as the observed one) / (1 + the number of simulated datasets). Only \textcolor{gray}{\texttt{test=\textquotesingle LR\textquotesingle }} is available.
```
*Was:*
```latex
\item \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} with \textcolor{gray}{\texttt{method}} \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}: these have no likelihood, so the test is a conditional parametric bootstrap \citep{Davison1997}. \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets are simulated from RSM's \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} fit, keeping each person's total score and the items they responded to. The statistic compares the two fitted models: each item's thresholds are taken relative to that item's central location in the RSM fit and in the PCM fit, and the squared differences between the two are summed over items and thresholds. It is computed for the fits to the data and recomputed on each simulated dataset with both models refitted (PCM with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} when \textcolor{gray}{\texttt{self}} was calibrated with \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, which has no PCM version). The \textit{p}-value is (1 + the number of simulated statistics at least as large as the observed one) / (1 + the number of simulated datasets). Only \textcolor{gray}{\texttt{test=\textquotesingle LR\textquotesingle }} is available.
```

### model_selection_rsm (line 1030)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~method=\textquotesingle log-lik\textquotesingle , log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### model_selection_rsm (line 1057)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the fits and bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### fit_statistics_rsm (line 1118)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~q1\_\allowbreak overall\_\allowbreak exceedances=10, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~q1\_\allowbreak overall\_\allowbreak exceedances=10)}}\\
```

### fit_statistics_rsm (line 1171)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### global_fit_rsm (line 1726)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=None, mc\_\allowbreak exceedances=None, unused\_\allowbreak categories=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~p\_\allowbreak method=None, mc\_\allowbreak exceedances=None, unused\_\allowbreak categories=None)}}\\
```

### global_fit_rsm (line 1760)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and Monte Carlo draws. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### dif_test_rsm (line 2227)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~method=None, matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, seed=None)}}\\
```

### dif_test_rsm (line 2288)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### recovery_stats_rsm (line 2361)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~plot=True, **plot\_\allowbreak kwargs, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~plot=True, **plot\_\allowbreak kwargs)}}\\
```

### recovery_stats_rsm (line 2384)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the simulated datasets. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

## mfrm.tex

### calibrate_mfrm (line 960)

```latex
Produces central item location estimates, a single shared set of Rasch--Andrich threshold estimates, and facet element (e.g.\ rater) location estimates. There are four approaches to estimation available, selectable through the \textcolor{gray}{\texttt{estimation}} argument: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }}. With \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }}, each facet element is fitted on its own representation (see Mixed below).\\
```
*Was:*
```latex
Produces central item location estimates, a single shared set of Rasch--Andrich threshold estimates, and facet element (e.g.\ rater) location estimates. There are three approaches to estimation available, selectable through the \textcolor{gray}{\texttt{estimation}} argument: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle}}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle}} and \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle}}.\\
```

### calibrate_mfrm (line 963)

```latex
The default, \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, fits item, threshold and facet (rater henceforth for simplicity) parameters jointly by iterative regularised composite likelihood (RCL). It uses every pair of response patterns $ (k, l) $ and $ (k-1, l+1) $ with the same total: two responses by the same rater to two items, or by two raters to the same item \citep{Elliott2022a}. This generalises the Bradley--Terry likelihood \citep{Bradley1952, Zermelo1929} (which it reduces to for dichotomous items, as in the SLM's \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}) to polytomous items and many-facet designs.  Each representation has its own item locations and thresholds (\textcolor{gray}{\texttt{self.items\_\allowbreak \{model\}}}, \textcolor{gray}{\texttt{self.thresholds\_\allowbreak \{model\}}}). Facet parameters carry ridge penalties (\textcolor{gray}{\texttt{global\_\allowbreak ridge}}, \textcolor{gray}{\texttt{item\_\allowbreak ridge}}, \textcolor{gray}{\texttt{threshold\_\allowbreak ridge}}, \textcolor{gray}{\texttt{interaction\_\allowbreak ridge}} and \textcolor{gray}{\texttt{stretch\_\allowbreak ridge}}) and the counts of each response pattern carry additive smoothing (\textcolor{gray}{\texttt{constant}}). Category widths with fewer than 20 threshold patterns are shrunk toward the mean of the other widths; the widths next to an empty middle category have no response patterns of their own (every pattern that would inform them includes the empty category): they are left out of the joint fit and set by the RSM's CPAT threshold estimator (\textcolor{gray}{\texttt{threshold\_\allowbreak method=\textquotesingle cpat-cl\textquotesingle }}) on all raters' responses together, given the item locations from the joint fit, which smooths them heavily. \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}} are fitted directly in their shift and stretch parameters.\\
```
*Was:*
```latex
The default, \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, fits item, threshold and facet (rater henceforth for simplicity) parameters jointly by iterative regularised composite likelihood (RCL). It uses every pair of response patterns $ (k, l) $ and $ (k-1, l+1) $ with the same total: two responses by the same rater to two items, or by two raters to the same item \citep{Elliott2022a}. This generalises the Bradley--Terry likelihood \citep{Bradley1952, Zermelo1929} (which it reduces to for dichotomous items, as in the SLM's \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}) to polytomous items and many-facet designs.  Each representation has its own item locations and thresholds (\textcolor{gray}{\texttt{self.items\_\allowbreak \{model\}}}, \textcolor{gray}{\texttt{self.thresholds\_\allowbreak \{model\}}}). Facet parameters carry ridge penalties (\textcolor{gray}{\texttt{ridge}}, \textcolor{gray}{\texttt{interaction\_\allowbreak ridge}}, \textcolor{gray}{\texttt{stretch\_\allowbreak ridge}}) and the counts of each response pattern carry additive smoothing (\textcolor{gray}{\texttt{constant}}). Category widths with fewer than 20 threshold patterns are shrunk toward the mean of the other widths; the widths next to an empty middle category are left out of the joint fit and estimated separately by conditional maximum likelihood from their own threshold patterns (\textcolor{gray}{\texttt{\textquotesingle cpat-cml\textquotesingle }}), given the item locations from the joint fit. \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}} are fitted directly in their shift and stretch parameters.\\
```

### calibrate_mfrm (line 967)

```latex

\textbf{CML}\\
\textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }} is penalised conditional maximum likelihood \citep{Andersen1970}: each facet element $ \times $ item pair is treated as an item, and the likelihood conditions on each person's total over all their ratings. The facet element effects carry the same penalties as with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, with their own defaults.\\

\textbf{Mixed}\\
\textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }} fits all facet elements jointly, each on its own representation (\textcolor{gray}{\texttt{rater\_\allowbreak models}}), sharing the item locations and thresholds. The representations are read relative to \textcolor{gray}{\texttt{anchors}}, the facet elements whose average defines the instrument (\textcolor{gray}{\texttt{None}}: all facet elements). Without \textcolor{gray}{\texttt{rater\_\allowbreak models}}, each facet element's representation is first chosen by \textcolor{gray}{\texttt{per\_\allowbreak rater\_\allowbreak model\_\allowbreak selection}}.\\
```

### calibrate_mfrm (line 1002)

```latex
\textcolor{gray}{\texttt{self.calibrate(model=\textquotesingle global\textquotesingle , constant=None, threshold\_\allowbreak constant=None, method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~matrix\_\allowbreak power=None, log\_\allowbreak lik\_\allowbreak tol=None, matrix\_\allowbreak marginals=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~regression=\textquotesingle huber\textquotesingle , robust=False, estimation=None, global\_\allowbreak ridge=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~item\_\allowbreak ridge=None, threshold\_\allowbreak ridge=None, interaction\_\allowbreak ridge=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~stretch\_\allowbreak ridge=None, unused\_\allowbreak categories=None, solver=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~ridge=None, rater\_\allowbreak models=None, anchors=None, selection=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~progress=None)}}

There is also one method for each facet representation, taking the same arguments without \textcolor{gray}{\texttt{model}}:
```
*Was:*
```latex
Eight methods are defined, one for each facet representation; all methods share the same argument format:
```

### calibrate_mfrm (line 1014)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, ...)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None)}}
```

### calibrate_mfrm (line 1018)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, item\_\allowbreak ridge=None, ...)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None)}}
```

### calibrate_mfrm (line 1022)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, threshold\_\allowbreak ridge=None, ...)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None)}}
```

### calibrate_mfrm (line 1026)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, item\_\allowbreak ridge=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~threshold\_\allowbreak ridge=None, interaction\_\allowbreak ridge=None, ...)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None, interaction\_\allowbreak ridge=None)}}
```

### calibrate_mfrm (line 1032)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, item\_\allowbreak ridge=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~threshold\_\allowbreak ridge=None, ...)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None)}}
```

### calibrate_mfrm (line 1038)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, stretch\_\allowbreak ridge=None, ...)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, stretch\_\allowbreak ridge=None)}}
```

### calibrate_mfrm (line 1043)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, stretch\_\allowbreak ridge=None, ...)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, stretch\_\allowbreak ridge=None)}}
```

### calibrate_mfrm (line 1048)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, stretch\_\allowbreak ridge=None, ...)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, stretch\_\allowbreak ridge=None)}}\\
```

### calibrate_mfrm (line 1060)

```latex
\textcolor{gray}{\texttt{model}} & String: the representation, one of \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle mixed\textquotesingle }}. Default is \textcolor{gray}{\texttt{model=\textquotesingle global\textquotesingle }}.\\
\midrule
```

### calibrate_mfrm (line 1078)

```latex
\textcolor{gray}{\texttt{estimation}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} (see Description). Default \textcolor{gray}{\texttt{estimation=None}} resolves to the method of the object's last calibration, or \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} for the first one. \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}}, \textcolor{gray}{\texttt{matrix\_\allowbreak marginals}} and \textcolor{gray}{\texttt{robust}} apply to \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} only; other values passed with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} are ignored, with a \textcolor{gray}{\texttt{UserWarning}}. The method chosen is carried into the refits made by other methods (bootstrap standard errors, DIF and DRF). \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} is faster on very large designs, but its standard errors are bootstrapped.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{estimation}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} (see Description). Default \textcolor{gray}{\texttt{estimation=None}} resolves to the method of the object's last calibration, or \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} for the first one. \textcolor{gray}{\texttt{method}}, \textcolor{gray}{\texttt{matrix\_\allowbreak power}}, \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}}, \textcolor{gray}{\texttt{matrix\_\allowbreak marginals}} and \textcolor{gray}{\texttt{robust}} apply to \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} only; other values passed with \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} are ignored, with a \textcolor{gray}{\texttt{UserWarning}}. The method chosen is carried into the refits made by other methods (bootstrap standard errors, DIF and DRF). \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} is faster on very large designs, but its standard errors are bootstrapped.\\
```

### calibrate_mfrm (line 1080)

```latex
\textcolor{gray}{\texttt{ridge}} & Float or \textcolor{gray}{\texttt{None}}: deprecated (\textcolor{gray}{\texttt{FutureWarning}}). The v1.4 penalty; given, it reproduces v1.4's estimates and cannot be combined with \textcolor{gray}{\texttt{global\_\allowbreak ridge}}, \textcolor{gray}{\texttt{item\_\allowbreak ridge}} or \textcolor{gray}{\texttt{threshold\_\allowbreak ridge}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{ridge}} & Float or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} and \textcolor{gray}{\texttt{global}}, \textcolor{gray}{\texttt{items}}, \textcolor{gray}{\texttt{thresholds}}, \textcolor{gray}{\texttt{bivector}} and \textcolor{gray}{\texttt{matrix}} (its bivector parts) only. Ridge penalty \textcolor{gray}{\texttt{ridge}} $ \times \sum $ (deviation)$^2 $ on the facet element deviations, in logits. \textcolor{gray}{\texttt{None}} (default) resolves to $ 0.3 $.\\
```

### calibrate_mfrm (line 1082)

```latex
\textcolor{gray}{\texttt{global\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: every representation, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Penalty on each facet element's severity. \textcolor{gray}{\texttt{None}} (default) resolves to $ 0.75 $.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{interaction\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} and \textcolor{gray}{\texttt{matrix}} only. Penalty on the item $ \times $ threshold interactions beyond the bivector parts. \textcolor{gray}{\texttt{None}} (default) resolves to $ 3 $.\\
```

### calibrate_mfrm (line 1084)

```latex
\textcolor{gray}{\texttt{item\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{items}}, \textcolor{gray}{\texttt{bivector}} and \textcolor{gray}{\texttt{matrix}} only, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Penalty on each facet element's item effects. \textcolor{gray}{\texttt{None}} (default) resolves to $ 0.405 $, $ 0.288 $ and $ 0.264 $ respectively ($ 0.135 $, $ 0.096 $ and $ 0.088 $ with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}).\\
\midrule
\textcolor{gray}{\texttt{threshold\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{thresholds}}, \textcolor{gray}{\texttt{bivector}} and \textcolor{gray}{\texttt{matrix}} only, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Penalty on each facet element's threshold effects. \textcolor{gray}{\texttt{None}} (default) resolves to $ 0.192 $, $ 0.288 $ and $ 0.264 $ respectively ($ 0.064 $, $ 0.096 $ and $ 0.088 $ with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}).\\
\midrule
\textcolor{gray}{\texttt{interaction\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{matrix}} only, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Penalty on the item $ \times $ threshold interactions beyond the bivector parts. \textcolor{gray}{\texttt{None}} (default) resolves to $ 7 $ ($ 0.7 $ with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}).\\
\midrule
\textcolor{gray}{\texttt{stretch\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}} only, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Penalty on $ (\omega - 1)^2 $. \textcolor{gray}{\texttt{None}} (default) resolves to $ 2 $ ($ 0.1 $ with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}).\\
\midrule
\textcolor{gray}{\texttt{unused\_\allowbreak categories}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }} only, when a score category is unused by every person with a non-extreme total. \textcolor{gray}{\texttt{None}} or \textcolor{gray}{\texttt{\textquotesingle ridge\textquotesingle }} (default): a penalty on the category widths next to it gives finite estimates. \textcolor{gray}{\texttt{\textquotesingle null\textquotesingle }}: the category has probability zero \citep{Wilson1993}; the threshold below it is stored as \textcolor{gray}{\texttt{inf}} and the next threshold holds the sum of the two.\\
\midrule
\textcolor{gray}{\texttt{solver}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }} only, not the stretch representations. \textcolor{gray}{\texttt{\textquotesingle newton\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle lbfgs\textquotesingle }}: how the maximum is found (all reach the same estimates). \textcolor{gray}{\texttt{None}} (default) resolves to \textcolor{gray}{\texttt{\textquotesingle chord\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{rater\_\allowbreak models}} & Dictionary, pandas series or list: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }} only. Each facet element's representation, keyed by facet element or in facet element order. \textcolor{gray}{\texttt{None}} (default) chooses them with \textcolor{gray}{\texttt{per\_\allowbreak rater\_\allowbreak model\_\allowbreak selection}}.\\
\midrule
\textcolor{gray}{\texttt{anchors}} & List or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }} only. The facet elements whose average defines the instrument. \textcolor{gray}{\texttt{None}} (default) means all facet elements.\\
\midrule
\textcolor{gray}{\texttt{selection}} & Dictionary or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }} without \textcolor{gray}{\texttt{rater\_\allowbreak models}} only. Arguments passed to \textcolor{gray}{\texttt{per\_\allowbreak rater\_\allowbreak model\_\allowbreak selection}}.\\
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for slow fits. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{stretch\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} and \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}} only. Penalty on $ (\omega - 1)^2 $. \textcolor{gray}{\texttt{None}} (default) resolves to $ 1 $.\\
```

### calibrate_mfrm (line 1133)

```latex
\textcolor{gray}{\texttt{self.convergence}} & Dictionary or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}, the number of Newton--Raphson iterations (\textcolor{gray}{\texttt{\textquotesingle iterations\textquotesingle }}) and whether the fit converged (\textcolor{gray}{\texttt{\textquotesingle converged\textquotesingle }}), with a \textcolor{gray}{\texttt{UserWarning}} if it did not; \textcolor{gray}{\texttt{None}} with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Also stored per representation as \textcolor{gray}{\texttt{self.convergence\_\allowbreak \{model\}}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.convergence}} & Dictionary or \textcolor{gray}{\texttt{None}}: with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, the number of Newton--Raphson iterations (\textcolor{gray}{\texttt{\textquotesingle iterations\textquotesingle }}) and whether the fit converged (\textcolor{gray}{\texttt{\textquotesingle converged\textquotesingle }}), with a \textcolor{gray}{\texttt{UserWarning}} if it did not; \textcolor{gray}{\texttt{None}} with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Also stored per representation as \textcolor{gray}{\texttt{self.convergence\_\allowbreak \{model\}}}.\\
```

### calibrate_mfrm (line 1135)

```latex
\textcolor{gray}{\texttt{self.global\_\allowbreak ridge}}, \textcolor{gray}{\texttt{self.item\_\allowbreak ridge}}, \textcolor{gray}{\texttt{self.threshold\_\allowbreak ridge}}, \textcolor{gray}{\texttt{self.interaction\_\allowbreak ridge}}, \textcolor{gray}{\texttt{self.stretch\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: the penalties used (\textcolor{gray}{\texttt{None}} where they do not apply).\\
\midrule
\textcolor{gray}{\texttt{self.cml\_\allowbreak unused\_\allowbreak categories}} & Dictionary: with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}, the unused score categories (empty if none).\\
\midrule
\textcolor{gray}{\texttt{self.items\_\allowbreak mixed}}, \textcolor{gray}{\texttt{self.thresholds\_\allowbreak mixed}} & pandas series: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }}: item locations and thresholds.\\
\midrule
\textcolor{gray}{\texttt{self.facet\_\allowbreak effects\_\allowbreak mixed}} & pandas DataFrame: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }}: every facet element's effects in the matrix layout, (facet element, item) $ \times $ threshold.\\
\midrule
\textcolor{gray}{\texttt{self.facet\_\allowbreak effects\_\allowbreak mixed\_\allowbreak \{model\}}} & pandas series/DataFrame: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }}: one table per representation used, in that representation's layout.\\
\midrule
\textcolor{gray}{\texttt{self.rater\_\allowbreak params\_\allowbreak mixed}} & Dictionary: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }}: \textcolor{gray}{\texttt{\{facet element: (representation, parameters)\}}}.\\
\midrule
\textcolor{gray}{\texttt{self.rater\_\allowbreak models}}, \textcolor{gray}{\texttt{self.anchors\_\allowbreak mixed}} & pandas series, list or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{model=\textquotesingle mixed\textquotesingle }}: the representations and anchors of the fit.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.ridge}}, \textcolor{gray}{\texttt{self.interaction\_\allowbreak ridge}}, \textcolor{gray}{\texttt{self.stretch\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: the regularisation values used (\textcolor{gray}{\texttt{None}} where they do not apply).\\
```

### calibrate_anchor_mfrm (line 1573)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None)}}
```

### calibrate_anchor_mfrm (line 1577)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, item\_\allowbreak ridge=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None)}}
```

### calibrate_anchor_mfrm (line 1582)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, threshold\_\allowbreak ridge=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None)}}
```

### calibrate_anchor_mfrm (line 1586)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, item\_\allowbreak ridge=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~threshold\_\allowbreak ridge=None, interaction\_\allowbreak ridge=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, ridge=None, interaction\_\allowbreak ridge=None)}}
```

### calibrate_anchor_mfrm (line 1591)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ estimation=None, matrix\_\allowbreak marginals=None, global\_\allowbreak ridge=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ item\_\allowbreak ridge=None, threshold\_\allowbreak ridge=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ estimation=None, matrix\_\allowbreak marginals=None, ridge=None)}}
```

### calibrate_anchor_mfrm (line 1597)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, stretch\_\allowbreak ridge=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, stretch\_\allowbreak ridge=None)}}
```

### calibrate_anchor_mfrm (line 1603)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, global\_\allowbreak ridge=None, stretch\_\allowbreak ridge=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, stretch\_\allowbreak ridge=None)}}
```

### calibrate_anchor_mfrm (line 1608)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, matrix\_\allowbreak marginals=None, global\_\allowbreak ridge=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~stretch\_\allowbreak ridge=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, matrix\_\allowbreak marginals=None, stretch\_\allowbreak ridge=None)}}\\
```

### calibrate_anchor_mfrm (line 1625)

```latex
\textcolor{gray}{\texttt{estimation}} & String or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}. Estimation method: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} (regularised composite likelihood over all response-pattern pairs), \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} (the same, restricted to CPAT patterns), \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} (penalised conditional maximum likelihood) or \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }}. \textcolor{gray}{\texttt{None}} uses the method of the object's latest calibration, or \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} if none. Default is \textcolor{gray}{\texttt{estimation=None}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{estimation}} & String or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}. Estimation method: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} (regularised composite likelihood over all response-pattern pairs), \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} (the same, restricted to CPAT patterns) or \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }}. \textcolor{gray}{\texttt{None}} uses the method of the object's latest calibration, or \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} if none. Default is \textcolor{gray}{\texttt{estimation=None}}.\\
```

### calibrate_anchor_mfrm (line 1645)

```latex
\textcolor{gray}{\texttt{ridge}} & Float or \textcolor{gray}{\texttt{None}}: deprecated (\textcolor{gray}{\texttt{FutureWarning}}). The v1.4 penalty; cannot be combined with \textcolor{gray}{\texttt{global\_\allowbreak ridge}}, \textcolor{gray}{\texttt{item\_\allowbreak ridge}} or \textcolor{gray}{\texttt{threshold\_\allowbreak ridge}}. Default is \textcolor{gray}{\texttt{ridge=None}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}. Ridge penalty on the facet element deviations, in logits. \textcolor{gray}{\texttt{None}} resolves to 0.3. Default is \textcolor{gray}{\texttt{ridge=None}}.\\
```

### calibrate_anchor_mfrm (line 1647)

```latex
\textcolor{gray}{\texttt{global\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. Penalty on each facet element's severity. \textcolor{gray}{\texttt{None}} resolves to 0.75. Default is \textcolor{gray}{\texttt{global\_\allowbreak ridge=None}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{interaction\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}. Penalty on the item $ \times $ threshold interactions beyond the bivector parts. \textcolor{gray}{\texttt{None}} resolves to 3. Default is \textcolor{gray}{\texttt{interaction\_\allowbreak ridge=None}}.\\
```

### calibrate_anchor_mfrm (line 1649)

```latex
\textcolor{gray}{\texttt{item\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}. Penalty on each facet element's item effects. \textcolor{gray}{\texttt{None}} resolves to 0.405, 0.288 and 0.264 respectively (0.135, 0.096 and 0.088 with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}). Default is \textcolor{gray}{\texttt{item\_\allowbreak ridge=None}}.\\
\midrule
\textcolor{gray}{\texttt{threshold\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}. Penalty on each facet element's threshold effects. \textcolor{gray}{\texttt{None}} resolves to 0.192, 0.288 and 0.264 respectively (0.064, 0.096 and 0.088 with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}). Default is \textcolor{gray}{\texttt{threshold\_\allowbreak ridge=None}}.\\
\midrule
\textcolor{gray}{\texttt{interaction\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}. Penalty on the item $ \times $ threshold interactions beyond the bivector parts. \textcolor{gray}{\texttt{None}} resolves to 7 (0.7 with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}). Default is \textcolor{gray}{\texttt{interaction\_\allowbreak ridge=None}}.\\
\midrule
\textcolor{gray}{\texttt{stretch\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }}. Penalty on $ (\omega - 1)^2 $. \textcolor{gray}{\texttt{None}} resolves to 2 (0.1 with \textcolor{gray}{\texttt{estimation=\textquotesingle cml\textquotesingle }}). Default is \textcolor{gray}{\texttt{stretch\_\allowbreak ridge=None}}.\\
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for slow fits. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{stretch\_\allowbreak ridge}} & Float or \textcolor{gray}{\texttt{None}}: Only relevant when \textcolor{gray}{\texttt{calibrate=True}}, \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, and \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }}. Penalty on $ (\omega - 1)^2 $. \textcolor{gray}{\texttt{None}} resolves to 1. Default is \textcolor{gray}{\texttt{stretch\_\allowbreak ridge=None}}.\\
```

### std_errors_mfrm (line 1747)

```latex
Produces standard errors of central item locations, thresholds, category widths and facet element effects. \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }} (representations calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}, and \textcolor{gray}{\texttt{mixed}}) computes analytic Huber-White Sandwich Estimator standard errors \citep{Huber1967}; \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle hessian\textquotesingle }} (CML only) uses the inverse Hessian alone; \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle bootstrap\textquotesingle }} draws \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} bootstrap resamples of persons. \textcolor{gray}{\texttt{se\_\allowbreak method=None}} (default) uses the sandwich where it applies; if it does not apply (an unused score category, a fit that did not converge, estimates not identified by the data, or a design too large), the bootstrap is used, with a \textcolor{gray}{\texttt{UserWarning}}. An explicit \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }} gives NaN for estimates the data do not identify. Standard errors of facet elements that used only the lowest or only the highest score category are NaN.\\
```
*Was:*
```latex
Produces standard errors of central item locations, thresholds, category widths and facet element effects. \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }} (representations calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}) computes analytic Huber-White Sandwich Estimator standard errors \citep{Huber1967}; \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle bootstrap\textquotesingle }} draws \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} bootstrap resamples of persons. \textcolor{gray}{\texttt{se\_\allowbreak method=None}} (default) uses the sandwich where it applies; if it does not apply (an unused score category, a fit that did not converge, estimates not identified by the data, or a design too large), the bootstrap is used, with a \textcolor{gray}{\texttt{UserWarning}}. An explicit \textcolor{gray}{\texttt{se\_\allowbreak method=\textquotesingle sandwich\textquotesingle }} gives NaN for estimates the data do not identify. Standard errors of facet elements that used only the lowest or only the highest score category are NaN.\\
```

### std_errors_mfrm (line 1755)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1761)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1767)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1773)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1779)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1785)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1791)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1797)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle , progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, store\_\allowbreak bootstrap=False, seed=None, se\_\allowbreak method=None)}}
```

### std_errors_mfrm (line 1833)

```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle hessian\textquotesingle }} (CML only), \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }} or \textcolor{gray}{\texttt{None}} (see Description). \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }} raises a \textcolor{gray}{\texttt{ValueError}} for a representation calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}, or with \textcolor{gray}{\texttt{store\_\allowbreak bootstrap=True}}. With the sandwich, the \textcolor{gray}{\texttt{\_\allowbreak bootstrap}} attributes are \textcolor{gray}{\texttt{None}}. Default is \textcolor{gray}{\texttt{se\_\allowbreak method=None}}.\\
\midrule
\textcolor{gray}{\texttt{no\_\allowbreak data\_\allowbreak se}} & String: \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }} and \textcolor{gray}{\texttt{mixed}}: for the facet element effects the data do not inform (a facet element $ \times $ item cell with no responses, or a facet element's thresholds next to a score category it never used; listed in \textcolor{gray}{\texttt{self.no\_\allowbreak data\_\allowbreak parameters}}), \textcolor{gray}{\texttt{\textquotesingle posterior\textquotesingle }}, the standard error from the information including the ridge, or \textcolor{gray}{\texttt{\textquotesingle nan\textquotesingle }}, which leaves them out of confidence intervals and tests. Default is \textcolor{gray}{\texttt{no\_\allowbreak data\_\allowbreak se=\textquotesingle posterior\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{se\_\allowbreak method}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bootstrap\textquotesingle }} or \textcolor{gray}{\texttt{None}} (see Description). \textcolor{gray}{\texttt{\textquotesingle sandwich\textquotesingle }} raises a \textcolor{gray}{\texttt{ValueError}} for a representation not calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, for \textcolor{gray}{\texttt{mixed}}, or with \textcolor{gray}{\texttt{store\_\allowbreak bootstrap=True}}. With the sandwich, the \textcolor{gray}{\texttt{\_\allowbreak bootstrap}} attributes are \textcolor{gray}{\texttt{None}}. Default is \textcolor{gray}{\texttt{se\_\allowbreak method=None}}.\\
```

### anchor_std_errors_mfrm (line 2036)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, progress=None)}}
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}
```

### anchor_std_errors_mfrm (line 2060)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### model_selection_mfrm (line 2472)

```latex
\item \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}: the same composite likelihood ratio test, AIC and BIC, on CPAT's patterns only.
\item \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }}: this has no likelihood, so the test is a conditional parametric bootstrap \citep{Davison1997}, and only \textcolor{gray}{\texttt{test=\textquotesingle LR\textquotesingle }} is available. For each smaller representation in a tested pair, \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets are simulated from its \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} fit, keeping each person's total over all their ratings and their missing ratings. The statistic compares the two fitted representations: the sum, over facet elements, items and thresholds, of the squared difference between the larger and smaller representations' fitted locations, each relative to its own mean. It is computed for the fits to the data and recomputed on each simulated dataset with both representations refitted with the object's calibration settings. The \textit{p}-value is (1 + the number of simulated statistics at least as large as the observed one) / (1 + the number of simulated datasets).
```
*Was:*
```latex
\item \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }}: these have no likelihood, so the test is a conditional parametric bootstrap \citep{Davison1997}, and only \textcolor{gray}{\texttt{test=\textquotesingle LR\textquotesingle }} is available. For each smaller representation in a tested pair, \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} datasets are simulated from its \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} fit, keeping each person's total over all their ratings and their missing ratings. The statistic compares the two fitted representations: the sum, over facet elements, items and thresholds, of the squared difference between the larger and smaller representations' fitted locations, each relative to its own mean. It is computed for the fits to the data and recomputed on each simulated dataset with both representations refitted with the object's calibration settings. The \textit{p}-value is (1 + the number of simulated statistics at least as large as the observed one) / (1 + the number of simulated datasets).
```

### model_selection_mfrm (line 2482)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None)}}\\
```

### model_selection_mfrm (line 2494)

```latex
\textcolor{gray}{\texttt{test}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }}, or \textcolor{gray}{\texttt{\textquotesingle BIC\textquotesingle }}. With \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}, the larger representation of a pair is preferred if the test (see Description) rejects the smaller one at level \textcolor{gray}{\texttt{alpha}}; with \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle BIC\textquotesingle }}, the representation with the lowest value is preferred (see \textcolor{gray}{\texttt{aic\_\allowbreak sig\_\allowbreak test}}). For \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}, nested pairwise tests are computed rather than a single ranking, since the eight representations do not form one single nested chain: \textcolor{gray}{\texttt{global}} vs.\ each of \textcolor{gray}{\texttt{items}}, \textcolor{gray}{\texttt{thresholds}}, \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}}; \textcolor{gray}{\texttt{items}} vs.\ \textcolor{gray}{\texttt{bivector}} and \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}}; \textcolor{gray}{\texttt{thresholds}} vs.\ \textcolor{gray}{\texttt{bivector}} and \textcolor{gray}{\texttt{centrality}}; \textcolor{gray}{\texttt{bivector}} vs.\ \textcolor{gray}{\texttt{matrix}} and \textcolor{gray}{\texttt{bistretch}}; \textcolor{gray}{\texttt{centrality}} vs.\ \textcolor{gray}{\texttt{bistretch}}; \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} vs.\ \textcolor{gray}{\texttt{bistretch}}. Default is \textcolor{gray}{\texttt{test=None}}, which resolves to \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }} for representations calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} (which has no likelihood, so the bootstrap test is the only one available) and to \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }} for \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{test}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }}, or \textcolor{gray}{\texttt{\textquotesingle BIC\textquotesingle }}. With \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}, the larger representation of a pair is preferred if the test (see Description) rejects the smaller one at level \textcolor{gray}{\texttt{alpha}}; with \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle BIC\textquotesingle }}, the representation with the lowest value is preferred (see \textcolor{gray}{\texttt{aic\_\allowbreak sig\_\allowbreak test}}). For \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}, nested pairwise tests are computed rather than a single ranking, since the eight representations do not form one single nested chain: \textcolor{gray}{\texttt{global}} vs.\ each of \textcolor{gray}{\texttt{items}}, \textcolor{gray}{\texttt{thresholds}}, \textcolor{gray}{\texttt{centrality}}, \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} and \textcolor{gray}{\texttt{bistretch}}; \textcolor{gray}{\texttt{items}} vs.\ \textcolor{gray}{\texttt{bivector}} and \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}}; \textcolor{gray}{\texttt{thresholds}} vs.\ \textcolor{gray}{\texttt{bivector}} and \textcolor{gray}{\texttt{centrality}}; \textcolor{gray}{\texttt{bivector}} vs.\ \textcolor{gray}{\texttt{matrix}} and \textcolor{gray}{\texttt{bistretch}}; \textcolor{gray}{\texttt{centrality}} vs.\ \textcolor{gray}{\texttt{bistretch}}; \textcolor{gray}{\texttt{pseudo\_\allowbreak halo}} vs.\ \textcolor{gray}{\texttt{bistretch}}. Default is \textcolor{gray}{\texttt{test=None}}, which resolves to \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }} for representations calibrated with \textcolor{gray}{\texttt{estimation=\textquotesingle cpat-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} (which have no likelihood, so the bootstrap test is the only one available) and to \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }} for \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}.\\
```

### model_selection_mfrm (line 2511)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the fits and bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### model_selection_mfrm (line 2539)

```latex
To compare all eight representations with the default test (AIC with the significance test for \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}; the bootstrap likelihood ratio test for \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }}):
```
*Was:*
```latex
To compare all eight representations with the default test (AIC with the significance test for \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}; the bootstrap likelihood ratio test for \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }}):
```

### per_rater_model_selection_mfrm (line 2558)

```latex

With \textcolor{gray}{\texttt{estimation=\textquotesingle cp-full-rcl\textquotesingle }} (the default), \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }}, each candidate representation of a facet element is a joint fit (\textcolor{gray}{\texttt{calibrate(model=\textquotesingle mixed\textquotesingle )}}) in the anchors' frame, with the other facet elements held at their current representations, and the candidates are compared by AIC or BIC; the joint fit with the chosen representations is then stored. The rest of this description applies to \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}.
```

### per_rater_model_selection_mfrm (line 2586)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, scheme=\textquotesingle iterate\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~estimation=None, max\_\allowbreak passes=10, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### per_rater_model_selection_mfrm (line 2601)

```latex
\textcolor{gray}{\texttt{test}} & String, one of \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle BIC\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}: model-selection criterion (\textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}: \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} only). Default is \textcolor{gray}{\texttt{test=\textquotesingle AIC\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{test}} & String, one of \textcolor{gray}{\texttt{\textquotesingle AIC\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle BIC\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle LR\textquotesingle }}: model-selection criterion. Default is \textcolor{gray}{\texttt{test=\textquotesingle AIC\textquotesingle }}.\\
```

### per_rater_model_selection_mfrm (line 2618)

```latex
\midrule
\textcolor{gray}{\texttt{estimation}} & String or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cpat-rcl\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cml\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle pair-cpat\textquotesingle }} (see Description). \textcolor{gray}{\texttt{None}} uses the object's latest estimation, or \textcolor{gray}{\texttt{\textquotesingle cp-full-rcl\textquotesingle }}. Default is \textcolor{gray}{\texttt{estimation=None}}.\\
\midrule
\textcolor{gray}{\texttt{scheme}} & String: not with \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }}. \textcolor{gray}{\texttt{\textquotesingle iterate\textquotesingle }}: start with every facet element on the most general candidate and update one facet element at a time until a pass changes nothing; \textcolor{gray}{\texttt{\textquotesingle one-pass\textquotesingle }}: each facet element with the others on the most general candidate. Default is \textcolor{gray}{\texttt{scheme=\textquotesingle iterate\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{max\_\allowbreak passes}} & Integer: \textcolor{gray}{\texttt{scheme=\textquotesingle iterate\textquotesingle }} only. Maximum number of passes. Default is \textcolor{gray}{\texttt{max\_\allowbreak passes=10}}.\\
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the fits. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### fit_statistics_mfrm (line 2690)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~conditional=False, conditional\_\allowbreak samples=1000, bootstrap\_\allowbreak samples=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~conditional=False, conditional\_\allowbreak samples=1000, bootstrap\_\allowbreak samples=None)}}\\
```

### fit_statistics_mfrm (line 2743)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### global_fit_mfrm (line 4137)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, p\_\allowbreak method=None, no\_\allowbreak of\_\allowbreak samples=500, mc\_\allowbreak exceedances=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~seed=None, split\_\allowbreak prop=0.5, p\_\allowbreak method=None, no\_\allowbreak of\_\allowbreak samples=500, mc\_\allowbreak exceedances=None)}}\\
```

### global_fit_mfrm (line 4168)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and Monte Carlo draws. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### dif_test_mfrm (line 4258)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### dif_test_mfrm (line 4341)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the group fits and bootstrap. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

### drf_test_mfrm (line 4395)

```latex
Differential Rater Functioning (DRF) test by an exogenous person covariate: the facet element (e.g.\ rater) analogue of DIF. Each group is fitted with \textcolor{gray}{\texttt{model}} (for \textcolor{gray}{\texttt{\textquotesingle mixed\textquotesingle }}, each facet element on its own representation), and each facet element that rated at least \textcolor{gray}{\texttt{min\_\allowbreak persons\_\allowbreak per\_\allowbreak group}} persons in each group is tested in its representation's terms: severity (\textcolor{gray}{\texttt{self.drf\_\allowbreak table}}), item effects (\textcolor{gray}{\texttt{self.drf\_\allowbreak item\_\allowbreak table}}), threshold effects (\textcolor{gray}{\texttt{self.drf\_\allowbreak threshold\_\allowbreak table}}), the item $ \times $ threshold interaction for \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }} (\textcolor{gray}{\texttt{self.drf\_\allowbreak interaction\_\allowbreak table}}), and all of these together (\textcolor{gray}{\texttt{self.drf\_\allowbreak overall\_\allowbreak table}}); for the stretch representations, the item and threshold tests are of the change in $ \omega $. Each facet element's difference between the groups is compared with the average difference of the other facet elements, leaving out, one at a time, those whose own difference is significant. The omnibus test (\textcolor{gray}{\texttt{self.drf\_\allowbreak omnibus\_\allowbreak table}}) asks whether any facet element differs between the groups, with an $ F $ reference. Cells whose categories have fewer than \textcolor{gray}{\texttt{min\_\allowbreak category\_\allowbreak count}} ratings in either group are left out, with a \textcolor{gray}{\texttt{UserWarning}}. Each group is calibrated on the items its persons responded to, and only items shared by both groups are compared.\\
```
*Was:*
```latex
Differential Rater Functioning (DRF) test by an exogenous person covariate: the facet-element-location analogue of DIF. The test depends on the model:

\begin{enumerate}
    \item \textbf{\textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}}: one test per common facet element (see below).
    \item \textbf{\textcolor{gray}{\texttt{\textquotesingle items\textquotesingle /\textquotesingle thresholds\textquotesingle /\textquotesingle bivector\textquotesingle /\textquotesingle matrix\textquotesingle }}}: a separate test of each facet element's effect on each item (\textcolor{gray}{\texttt{items}}) or threshold (\textcolor{gray}{\texttt{thresholds}}). \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle /\textquotesingle matrix\textquotesingle }} test both, in two tables: \textcolor{gray}{\texttt{drf\_\allowbreak table}} (items) and \textcolor{gray}{\texttt{drf\_\allowbreak threshold\_\allowbreak table}} (thresholds). \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }} uses marginal item and threshold projections.
    \item \textbf{\textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle /\textquotesingle pseudo\_\allowbreak halo\textquotesingle /\textquotesingle bistretch\textquotesingle }}}: one joint test per common facet element of all its parameters together ($ \lambda $ and $ \omega $; for \textcolor{gray}{\texttt{bistretch}}, $ \lambda $, $ \omega_{\text{items}} $ and $ \omega_{\text{thresholds}} $). The statistic is $ T = \Delta^\top \Sigma^{-1} \Delta $, where $ \Delta $ is the vector of focal-minus-reference differences in the facet element's parameters, and $ \Sigma $ is the covariance matrix of those differences estimated from paired bootstraps. $ T $ is compared with a $ \chi^2 $ distribution with 2 degrees of freedom (3 for \textcolor{gray}{\texttt{bistretch}}). Separate z-tests for each parameter are also reported. $ \lambda $ is purified as for \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}; $ \omega $ is not purified.
\end{enumerate}

Only common facet elements are tested: those that rated at least \textcolor{gray}{\texttt{min\_\allowbreak persons\_\allowbreak per\_\allowbreak group}} persons in each group. Others are listed in \textcolor{gray}{\texttt{self.drf\_\allowbreak excluded\_\allowbreak raters}}. Purification uses the same anchor-selection method as \textcolor{gray}{\texttt{dif\_\allowbreak test}}, applied to facet element locations; all common facet elements are tested. For \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }} there is one translation constant per focal group; for \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle /\textquotesingle thresholds\textquotesingle /\textquotesingle bivector\textquotesingle /\textquotesingle matrix\textquotesingle }} there is one per item or threshold.

The omnibus test is available for \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }} only, and is run by default. It holds items and thresholds at their pooled values and lets the facet effects vary per group, with $ df = $ common facet elements $ - 1 $.

Each group is calibrated on the items its persons responded to, and only items shared by both groups are tested. Parameters that depend on which facet elements or items they are relative to (the \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }} threshold marginal, and the parameters of \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }}) are first anchored on the facet elements and items shared by both groups. A focal group that shares fewer than two facet elements, or (where the model needs them) no items, with the reference group (possible with more than two groups) is left out with a \textcolor{gray}{\texttt{UserWarning}} and recorded in \textcolor{gray}{\texttt{self.drf\_\allowbreak excluded\_\allowbreak groups}}.
```

### drf_test_mfrm (line 4398)

```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak test(covariate, model=\textquotesingle global\textquotesingle , reference=None, min\_\allowbreak anchors=2,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~wald\_\allowbreak alpha=0.05, min\_\allowbreak persons\_\allowbreak per\_\allowbreak group=5, min\_\allowbreak category\_\allowbreak count=5,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~omnibus=None, correction=\textquotesingle bh\textquotesingle , alpha=0.05, logit\_\allowbreak threshold=0.43, plot=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~plot\_\allowbreak kwargs=None, p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle , no\_\allowbreak of\_\allowbreak samples=500,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~seed=None, threshold\_\allowbreak constant=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak test(covariate, model=\textquotesingle global\textquotesingle , reference=None, selection\_\allowbreak method=\textquotesingle wald\textquotesingle ,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~corr\_\allowbreak tol=0.95, sd\_\allowbreak ratio\_\allowbreak tol=1.1, min\_\allowbreak anchors=2, wald\_\allowbreak alpha=0.05,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~min\_\allowbreak persons\_\allowbreak per\_\allowbreak group=5, omnibus=None, welch=False, correction=\textquotesingle bh\textquotesingle , alpha=0.05,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~logit\_\allowbreak threshold=0.43, no\_\allowbreak of\_\allowbreak samples=500, plot=False, plot\_\allowbreak kwargs=None,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~warm\_\allowbreak corr=True, tolerance=0.00001, max\_\allowbreak iters=100, ext\_\allowbreak score\_\allowbreak adjustment=0.5,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~constant=None, threshold\_\allowbreak constant=None, method=None, matrix\_\allowbreak power=None, robust=False,}}\\
\textcolor{gray}{\texttt{~~~~~~~~~~~~~log\_\allowbreak lik\_\allowbreak tol=0.000001, seed=None)}}\\
```

### drf_test_mfrm (line 4416)

```latex
\textcolor{gray}{\texttt{model}} & String, one of \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle mixed\textquotesingle }} (which needs a fit by \textcolor{gray}{\texttt{calibrate(model=\textquotesingle mixed\textquotesingle )}}). Default is \textcolor{gray}{\texttt{model=\textquotesingle global\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{model}} & String, one of \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }}: the facet element representation each group is calibrated under, which determines the tests (see above). Default is \textcolor{gray}{\texttt{model=\textquotesingle global\textquotesingle }}.\\
```

### drf_test_mfrm (line 4420)

```latex
\textcolor{gray}{\texttt{min\_\allowbreak anchors}} & Integer: the fewest facet elements kept in the comparison for each facet element's test. Default is \textcolor{gray}{\texttt{min\_\allowbreak anchors=2}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{selection\_\allowbreak method}} & String, one of \textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle none\textquotesingle }}: how the facet element location scale is purified before testing. \textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }}: sequential Wald tests of each common facet element's difference, using \textcolor{gray}{\texttt{wald\_\allowbreak alpha}} and \textcolor{gray}{\texttt{min\_\allowbreak anchors}}. \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }}: excludes outlying facet elements by the modified z-score procedure \citep{IglewiczHoaglin1993}, using \textcolor{gray}{\texttt{corr\_\allowbreak tol}}, \textcolor{gray}{\texttt{sd\_\allowbreak ratio\_\allowbreak tol}} and \textcolor{gray}{\texttt{min\_\allowbreak anchors}}. \textcolor{gray}{\texttt{\textquotesingle none\textquotesingle }}: mean shift over all common facet elements. Default is \textcolor{gray}{\texttt{selection\_\allowbreak method=\textquotesingle wald\textquotesingle }}.\\
```

### drf_test_mfrm (line 4422)

```latex
\textcolor{gray}{\texttt{wald\_\allowbreak alpha}} & Float: significance level at which facet elements are left out of the comparison. Default is \textcolor{gray}{\texttt{wald\_\allowbreak alpha=0.05}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{corr\_\allowbreak tol}} & Float: minimum correlation between the two groups' facet element locations on the retained facet elements (\textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }} only). Default is \textcolor{gray}{\texttt{corr\_\allowbreak tol=0.95}}.\\
```

### drf_test_mfrm (line 4424)

```latex
\textcolor{gray}{\texttt{min\_\allowbreak persons\_\allowbreak per\_\allowbreak group}} & Integer: minimum number of persons a facet element must have rated in each group to be tested. Others are listed in \textcolor{gray}{\texttt{self.drf\_\allowbreak excluded\_\allowbreak raters}}. Default is \textcolor{gray}{\texttt{min\_\allowbreak persons\_\allowbreak per\_\allowbreak group=5}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{sd\_\allowbreak ratio\_\allowbreak tol}} & Float: the ratio of the two groups' facet element location standard deviations on the retained facet elements must lie in $ [1/\textcolor{gray}{\texttt{sd\_\allowbreak ratio\_\allowbreak tol}}, \textcolor{gray}{\texttt{sd\_\allowbreak ratio\_\allowbreak tol}}] $ (\textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }} only). Default is \textcolor{gray}{\texttt{sd\_\allowbreak ratio\_\allowbreak tol=1.1}}.\\
```

### drf_test_mfrm (line 4426)

```latex
\textcolor{gray}{\texttt{min\_\allowbreak category\_\allowbreak count}} & Integer or \textcolor{gray}{\texttt{None}}: cells whose categories have fewer ratings than this in either group are left out of the tests; \textcolor{gray}{\texttt{None}} leaves nothing out. Default is \textcolor{gray}{\texttt{min\_\allowbreak category\_\allowbreak count=5}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{min\_\allowbreak anchors}} & Integer: minimum number of common facet elements retained by purification (\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }}). Default is \textcolor{gray}{\texttt{min\_\allowbreak anchors=2}}.\\
```

### drf_test_mfrm (line 4428)

```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean or \textcolor{gray}{\texttt{None}}: whether to run the omnibus test. \textcolor{gray}{\texttt{None}} runs it. Default is \textcolor{gray}{\texttt{omnibus=None}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{wald\_\allowbreak alpha}} & Float: significance level for each facet element's purification test (\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }} only). Default is \textcolor{gray}{\texttt{wald\_\allowbreak alpha=0.05}}.\\
```

### drf_test_mfrm (line 4430)

```latex
\textcolor{gray}{\texttt{correction}} & String or \textcolor{gray}{\texttt{None}}, one of \textcolor{gray}{\texttt{\textquotesingle bh\textquotesingle }} (Benjamini--Hochberg \citep{BenjaminiHochberg1995}), \textcolor{gray}{\texttt{\textquotesingle bonferroni\textquotesingle }} or \textcolor{gray}{\texttt{None}}: multiple-comparison correction across each table's tests. Default is \textcolor{gray}{\texttt{correction=\textquotesingle bh\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{min\_\allowbreak persons\_\allowbreak per\_\allowbreak group}} & Integer: minimum number of persons a facet element must have rated in each group to be a common facet element. Others are excluded and listed in \textcolor{gray}{\texttt{self.drf\_\allowbreak excluded\_\allowbreak raters}}. Default is \textcolor{gray}{\texttt{min\_\allowbreak persons\_\allowbreak per\_\allowbreak group=5}}.\\
```

### drf_test_mfrm (line 4432)

```latex
\textcolor{gray}{\texttt{alpha}} & Float: significance level for flagging. Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{omnibus}} & Boolean or \textcolor{gray}{\texttt{None}}: whether to run the omnibus test (\textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }} only; \textcolor{gray}{\texttt{True}} with any other model raises a \textcolor{gray}{\texttt{ValueError}}). \textcolor{gray}{\texttt{None}} runs it for \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}. Default is \textcolor{gray}{\texttt{omnibus=None}}.\\
```

### drf_test_mfrm (line 4434)

```latex
\textcolor{gray}{\texttt{logit\_\allowbreak threshold}} & Float: minimum absolute severity difference, in logits, for flagging in \textcolor{gray}{\texttt{self.drf\_\allowbreak table}}. Default is \textcolor{gray}{\texttt{logit\_\allowbreak threshold=0.43}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{welch}} & Boolean: if \textcolor{gray}{\texttt{True}}, the Wald tests use a $t$-distribution with Welch--Satterthwaite degrees of freedom instead of the standard normal, and a \textcolor{gray}{\texttt{df}} column is added. Does not affect the joint tests or the omnibus test. Default is \textcolor{gray}{\texttt{welch=False}}.\\
```

### drf_test_mfrm (line 4436)

```latex
\textcolor{gray}{\texttt{plot}} & Boolean: if \textcolor{gray}{\texttt{True}}, plots each facet element's severity in the two groups for each focal group with \textcolor{gray}{\texttt{plot\_\allowbreak anchor\_\allowbreak selection}}, stored in \textcolor{gray}{\texttt{self.drf\_\allowbreak plots}}. Default is \textcolor{gray}{\texttt{plot=False}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{correction}} & String or \textcolor{gray}{\texttt{None}}, one of \textcolor{gray}{\texttt{\textquotesingle bh\textquotesingle }} (Benjamini--Hochberg \citep{BenjaminiHochberg1995}), \textcolor{gray}{\texttt{\textquotesingle bonferroni\textquotesingle }} or \textcolor{gray}{\texttt{None}}: multiple-comparison correction across each table's tests. Not applied to the omnibus test. Default is \textcolor{gray}{\texttt{correction=\textquotesingle bh\textquotesingle }}.\\
\midrule
\textcolor{gray}{\texttt{alpha}} & Float: significance level for flagging (corrected \textit{p}-values if \textcolor{gray}{\texttt{correction}} is set). Default is \textcolor{gray}{\texttt{alpha=0.05}}.\\
\midrule
\textcolor{gray}{\texttt{logit\_\allowbreak threshold}} & Float: minimum absolute difference, in logits, for flagging. Default is \textcolor{gray}{\texttt{logit\_\allowbreak threshold=0.43}}.\\
\midrule
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: bootstrap resamples for each group's standard errors. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
\midrule
\textcolor{gray}{\texttt{plot}} & Boolean: if \textcolor{gray}{\texttt{True}}, plots the purification for each focal group with \textcolor{gray}{\texttt{plot\_\allowbreak anchor\_\allowbreak selection}} (\textcolor{gray}{\texttt{\textquotesingle wald\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle robust\_\allowbreak z\textquotesingle }} only), stored in \textcolor{gray}{\texttt{self.drf\_\allowbreak plots}}. Default is \textcolor{gray}{\texttt{plot=False}}.\\
```

### drf_test_mfrm (line 4440)

```latex
\textcolor{gray}{\texttt{p\_\allowbreak method}} & String: \textcolor{gray}{\texttt{\textquotesingle monte-carlo\textquotesingle }}, Monte Carlo \textit{p}-values with the choice of comparison facet elements repeated on each draw, or \textcolor{gray}{\texttt{\textquotesingle chi2\textquotesingle }}, chi-squared \textit{p}-values (faster, slightly liberal). Default is \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{warm\_\allowbreak corr}} & Boolean: if \textcolor{gray}{\texttt{True}}, person locations are Warm's weighted likelihood estimates; otherwise maximum likelihood estimates. Used only for the omnibus test. Default is \textcolor{gray}{\texttt{warm\_\allowbreak corr=True}}.\\
```

### drf_test_mfrm (line 4442)

```latex
\textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples}} & Integer: draws for \textcolor{gray}{\texttt{p\_\allowbreak method=\textquotesingle monte-carlo\textquotesingle }}. Default is \textcolor{gray}{\texttt{no\_\allowbreak of\_\allowbreak samples=500}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{tolerance}} & Float: convergence criterion for person estimation (omnibus test only). Default is \textcolor{gray}{\texttt{tolerance=0.00001}}.\\
```

### drf_test_mfrm (line 4444)

```latex
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for the draws. \textcolor{gray}{\texttt{None}} draws fresh entropy (Section \ref{reproducibility}). Default is \textcolor{gray}{\texttt{seed=None}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{max\_\allowbreak iters}} & Integer: maximum Newton--Raphson iterations for person estimation (omnibus test only). Default is \textcolor{gray}{\texttt{max\_\allowbreak iters=100}}.\\
```

### drf_test_mfrm (line 4446)

```latex
\textcolor{gray}{\texttt{threshold\_\allowbreak constant}} & Float or \textcolor{gray}{\texttt{None}}: passed to each group's calibration. Default is \textcolor{gray}{\texttt{threshold\_\allowbreak constant=None}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment}} & Float in $ (0, 1) $: adjustment to extreme scores for maximum likelihood person estimation (omnibus test only). Default is \textcolor{gray}{\texttt{ext\_\allowbreak score\_\allowbreak adjustment=0.5}}.\\
\midrule
\textcolor{gray}{\texttt{constant}} & Float or \textcolor{gray}{\texttt{None}}: additive smoothing constant for each group's calibration. \textcolor{gray}{\texttt{None}} uses the data-adaptive default. Default is \textcolor{gray}{\texttt{constant=None}}.\\
\midrule
\textcolor{gray}{\texttt{threshold\_\allowbreak constant}} & Float or \textcolor{gray}{\texttt{None}}: additive smoothing constant for the thresholds. \textcolor{gray}{\texttt{None}} follows \textcolor{gray}{\texttt{constant}}. Default is \textcolor{gray}{\texttt{threshold\_\allowbreak constant=None}}.\\
\midrule
\textcolor{gray}{\texttt{method}} & String or \textcolor{gray}{\texttt{None}}: pairwise estimation method (\textcolor{gray}{\texttt{\textquotesingle ls\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle evm\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle cos\textquotesingle }} or \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }}), \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} only. \textcolor{gray}{\texttt{None}} resolves by representation. Default is \textcolor{gray}{\texttt{method=None}}.\\
\midrule
\textcolor{gray}{\texttt{matrix\_\allowbreak power}} & Integer or \textcolor{gray}{\texttt{None}}: power to which the pairwise matrix is raised, \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} only. \textcolor{gray}{\texttt{None}} resolves to 0 for \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} and 5 otherwise. Default is \textcolor{gray}{\texttt{matrix\_\allowbreak power=None}}.\\
\midrule
\textcolor{gray}{\texttt{robust}} & Boolean: \textcolor{gray}{\texttt{estimation=\textquotesingle pair-cpat\textquotesingle }} only; for \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, estimates the item and threshold vectors directly rather than as marginal means of the matrix representation. Default is \textcolor{gray}{\texttt{robust=False}}.\\
\midrule
\textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol}} & Float: convergence criterion for the \textcolor{gray}{\texttt{\textquotesingle log-lik\textquotesingle }} method. Default is \textcolor{gray}{\texttt{log\_\allowbreak lik\_\allowbreak tol=0.000001}}.\\
\midrule
\textcolor{gray}{\texttt{seed}} & Integer or \textcolor{gray}{\texttt{None}}: seed for each group's bootstrap. \textcolor{gray}{\texttt{None}} draws fresh entropy (Section \ref{reproducibility}). Default is \textcolor{gray}{\texttt{seed=None}}.\\
```

### drf_test_mfrm (line 4469)

```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak group\_\allowbreak sizes}} & Dictionary: \textcolor{gray}{\texttt{\{group\_\allowbreak value: int\}}}, the number of persons in each group.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak group\_\allowbreak sizes}} & Dictionary: \textcolor{gray}{\texttt{\{group\_\allowbreak value: int\}}}, the number of persons (non-extreme, covariate not missing) in each group.\\
```

### drf_test_mfrm (line 4473)

```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak excluded\_\allowbreak groups}} & Dictionary: \textcolor{gray}{\texttt{\{focal\_\allowbreak value: str\}}}, focal groups that cannot be compared with the reference group, with the reason.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak table}} & pandas dataframe, common facet elements only. \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}: one row per (facet element, focal group), with columns \textcolor{gray}{\texttt{Group}}, \textcolor{gray}{\texttt{Reference}}, \textcolor{gray}{\texttt{Focal}}, \textcolor{gray}{\texttt{Focal (purified)}} (facet element locations), \textcolor{gray}{\texttt{Difference}}, \textcolor{gray}{\texttt{SE}}, \textcolor{gray}{\texttt{z}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{p (corrected)}}, \textcolor{gray}{\texttt{Selected}}, \textcolor{gray}{\texttt{Flagged}}, and \textcolor{gray}{\texttt{df}} if \textcolor{gray}{\texttt{welch=True}}. \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}: one row per (facet element, item or threshold, focal group), with the same columns plus an \textcolor{gray}{\texttt{Item}} or \textcolor{gray}{\texttt{Category}} column; one correction across the whole table; for \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}, items only. \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }}: one row per (facet element, focal group), with \textcolor{gray}{\texttt{Group}}; \textcolor{gray}{\texttt{Reference\_\{param\}}}, \textcolor{gray}{\texttt{Focal\_\{param\}}} and \textcolor{gray}{\texttt{Difference\_\{param\}}} for each parameter; \textcolor{gray}{\texttt{T}} (the joint statistic), \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{p (corrected)}}, \textcolor{gray}{\texttt{Selected}} ($ \lambda $ purification) and \textcolor{gray}{\texttt{Flagged}} (joint test); and \textcolor{gray}{\texttt{z\_\{param\}}}, \textcolor{gray}{\texttt{p\_\{param\}}}, \textcolor{gray}{\texttt{p\_\{param\} (corrected)}} and \textcolor{gray}{\texttt{Flagged\_\{param\}}} for each parameter, with a separate correction.\\
```

### drf_test_mfrm (line 4475)

```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak table}} & pandas dataframe: severity tests, one row per (facet element, focal group), with columns \textcolor{gray}{\texttt{Group}}, \textcolor{gray}{\texttt{Representation}}, \textcolor{gray}{\texttt{Reference}}, \textcolor{gray}{\texttt{Focal}}, \textcolor{gray}{\texttt{Difference}}, \textcolor{gray}{\texttt{SE}}, \textcolor{gray}{\texttt{T}}, \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{p (corrected)}}, \textcolor{gray}{\texttt{Flagged}}, \textcolor{gray}{\texttt{Pool}}, \textcolor{gray}{\texttt{Cells left out}}, \textcolor{gray}{\texttt{Note}}.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak threshold\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }} and \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }} only: the threshold tests, with the same columns as \textcolor{gray}{\texttt{self.drf\_\allowbreak table}} and a \textcolor{gray}{\texttt{Category}} column, corrected separately. \textcolor{gray}{\texttt{None}} for other models.\\
```

### drf_test_mfrm (line 4477)

```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak item\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: item tests, with columns \textcolor{gray}{\texttt{Group}}, \textcolor{gray}{\texttt{Representation}}, \textcolor{gray}{\texttt{T}}, \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{p (corrected)}}, \textcolor{gray}{\texttt{Flagged}}, \textcolor{gray}{\texttt{Pool}}, \textcolor{gray}{\texttt{Cells left out}}, \textcolor{gray}{\texttt{Note}}, and each item's difference, SE and Holm-corrected \textit{p}-value (\textcolor{gray}{\texttt{Difference omega}}, \textcolor{gray}{\texttt{SE omega}} for the stretch representations).\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak omnibus\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: one row per focal group, with \textcolor{gray}{\texttt{LR}}, \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{Flagged}}. \textcolor{gray}{\texttt{None}} unless the omnibus test is run.\\
```

### drf_test_mfrm (line 4479)

```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak threshold\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: threshold tests, with the same columns, and each category width's difference, SE and Holm-corrected \textit{p}-value.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak tc}} & Dictionary: the translation constants applied to each focal group, one per common facet element (each excluding that facet element). \textcolor{gray}{\texttt{\textquotesingle global\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle centrality\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle pseudo\_\allowbreak halo\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle bistretch\textquotesingle }} ($ \lambda $ only): \textcolor{gray}{\texttt{\{focal\_\allowbreak value: pandas.Series\}}}. \textcolor{gray}{\texttt{\textquotesingle items\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle thresholds\textquotesingle }}: \textcolor{gray}{\texttt{\{focal\_\allowbreak value: \{item or threshold: pandas.Series\}\}}}. \textcolor{gray}{\texttt{\textquotesingle bivector\textquotesingle }}, \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }}: \textcolor{gray}{\texttt{\{focal\_\allowbreak value: \{\textquotesingle items\textquotesingle : \{item: pandas.Series\}, \textquotesingle thresholds\textquotesingle : \{threshold: pandas.Series\}\}\}}}.\\
```

### drf_test_mfrm (line 4481)

```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak interaction\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: \textcolor{gray}{\texttt{\textquotesingle matrix\textquotesingle }} only: the item $ \times $ threshold interaction tests, with the same columns.\\
\midrule
\textcolor{gray}{\texttt{self.drf\_\allowbreak overall\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: the tests of all of a facet element's components together, with the same columns.\\
\midrule
\textcolor{gray}{\texttt{self.drf\_\allowbreak omnibus\_\allowbreak table}} & pandas dataframe or \textcolor{gray}{\texttt{None}}: one row per focal group, with \textcolor{gray}{\texttt{T}}, \textcolor{gray}{\texttt{df}}, \textcolor{gray}{\texttt{F}}, \textcolor{gray}{\texttt{Denominator df}}, \textcolor{gray}{\texttt{p}}, \textcolor{gray}{\texttt{Flagged}}. \textcolor{gray}{\texttt{None}} unless the omnibus test is run.\\
\midrule
\textcolor{gray}{\texttt{self.drf\_\allowbreak anchor\_\allowbreak selection}} & Dictionary: \textcolor{gray}{\texttt{\{focal\_\allowbreak value: dataframe\}}}, each facet element's severity in the reference and focal groups and whether it is among the comparison facet elements.\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{self.drf\_\allowbreak anchor\_\allowbreak selection}} & Dictionary: \textcolor{gray}{\texttt{\{focal\_\allowbreak value: dataframe or None\}}}, the purification table for each focal group (\textcolor{gray}{\texttt{None}} if \textcolor{gray}{\texttt{selection\_\allowbreak method=\textquotesingle none\textquotesingle }}).\\
```

### recovery_stats_mfrm (line 4516)

```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~divergence\_\allowbreak tol=0.1, plot=True, **plot\_\allowbreak kwargs, progress=None)}}\\
```
*Was:*
```latex
\textcolor{gray}{\texttt{~~~~~~~~~~~~~~~~~~~~~~~~divergence\_\allowbreak tol=0.1, plot=True, **plot\_\allowbreak kwargs)}}\\
```

### recovery_stats_mfrm (line 4547)

```latex
\midrule
\textcolor{gray}{\texttt{progress}} & Boolean or \textcolor{gray}{\texttt{None}}: progress line (done / total, time elapsed and left) for the simulated datasets. \textcolor{gray}{\texttt{None}} (default) shows it when the run is expected to take more than 10 seconds; \textcolor{gray}{\texttt{True}} always; \textcolor{gray}{\texttt{False}} never.\\
```

## 2026-11-06 (sections changed in the local files; not yet in Overleaf)

New references (`new_references.bib`, add to `references.bib` on Overleaf): Holm1979, Jin2026.

mfrm.tex
- `check_anchor_homogeneity`: rewritten (the joint-fit check as default for cp-full-rcl/cpat-rcl/cml; the Cochran/robust test as an option and the pair-cpat default).
- `calibrate`: `anchors` argument (anchor sets with cut cells); returns `anchor_cuts_mixed`, `anchor_set_mixed`.
- `per_rater_model_selection`: anchors held on 'global' with joint fits; `anchors` argument; the anchored variant and `anchor_*` attributes marked pair-cpat only; table note; `threshold_constant`, `robust` added.
- `rater_stats_df`: `divergence_test` (stacked-sandwich SEs, `divergence_se_method`, 't' degrees of freedom); `matrix_power`, `log_lik_tol` added.
- `threshold_stats_df`: `matrix_power`, `log_lik_tol` added.
- `anchor_std_errors`, `category_stats_df`, `save_stats`: `threshold_constant`, `robust` added.
- `item_stats_df`: `ext_scores` added.
- `drf_test`: description, one added sentence on a facet element that gave fewer than `min_category_count` ratings in a score category in either group (global effect still tested; threshold test not run; Note columns).

rsm.tex
- `category_stats_df`, `dif_test`, `save_stats`: `threshold_constant` added.
- `res_corr_analysis`: `no_of_samples`, `interval` added.
- `csem`, `person`: `person` renamed `persons` (rows as in slm.tex); `person`, `person_estimates`: `missing_as_incorrect` added.

pcm.tex
- `res_corr_analysis`: `no_of_samples`, `interval` added.

All chapters
- 'severity' replaced by 'global effect' throughout mfrm.tex (a facet element's overall level), including `check_anchor_homogeneity`'s 'global effect' row and Reason value; kept: ETS A/B/C DIF 'severity-category' (slm/rsm/pcm/mfrm) and Jin and Eckes's 'category-specific severity shift'.
- `check_anchor_homogeneity` (2026-11-07): paragraph on the F / t references for the Wald tests, with the small-sample simulation results; note that the split_alpha figures used chi-squared references.
- Cite keys renamed to first author + year (2026-11-07): VarinVidoni2005 -> Varin2005, JinEckes2026 -> Jin2026 (mfrm.tex, pcm.tex, rsm.tex, new_references.bib).
- slm.tex `calibrate` (2026-11-07): CML sentence rewritten (response pattern given the total; sufficiency; no distributional assumption; extreme totals contribute nothing).
- slm.tex `std_errors` (2026-11-07): `no_data_se` row rewritten (user's wording: how SEs are determined for items not estimable under ridge CML; 'posterior' and 'nan' as separate clauses; conservative-tests note moved to its own sentence).

DONE 2026-11-09 (was PENDING since 2026-11-07): applied to the downloaded slm.tex and ported to rsm/pcm/mfrm; status of every item in manual_port_table_2026-11-09.md
- Stale "bootstrap" standard-error wording: calibrate_anchor/dif_test call std_errors() with se_method=None, which gives the sandwich for estimation='cml' and method='log-lik', bootstrap only for 'cos'/'ls'/'evm'. Find and fix every such reference (grep -i bootstrap), e.g. slm.tex calibrate_anchor selection_method ("needs bootstrap standard errors"), no_of_samples, seed, the 'wald' example ("fully reproducible bootstrap"), dif_test no_of_samples. Same fix in docstrings: raschpy/slm.py:712, 730, 2670 (and the counterparts in rsm/pcm/mfrm).
- slm.tex `fit_statistics` (2026-11-07): conditional=True sentence on item-restscore, Q1 (per item and overall, Monte Carlo p) and person conditional p rewritten as a list of three statistics.
- slm.tex `fit_statistics` `q1_groups` (2026-11-08): default grouping described exactly (persons/60, 2 to 10 groups, groups under 30 merged while more than 2 remain; about 60 per group only at 120-600 persons); fewer than 2 groups gives NaN with a warning. Code fixed the same day (merging stops at 2 groups). User pasting the LaTeX on Overleaf; port to rsm/pcm.
- slm.tex `fit_statistics` attributes (2026-11-08, user pasting on Overleaf): `item_q1_groups` reworded (z; variance including item estimation error); `item_q1_overall` reworded (no 'item scores' row label in SLM; PCM/RSM keep row labels incl. 'thresholds').
- slm.tex `global_fit` (2026-11-08, user pasting on Overleaf): Andersen sentence "pairwise conditional likelihood those estimates maximise" -> "composite likelihood those estimates maximise"; same sentence in rsm.tex 1723, pcm.tex 1835.
- slm.tex `global_fit` (2026-11-08, user pasting on Overleaf): Andersen with 'cp-full-rcl' — added that the composite LR follows a weighted sum of chi-square variables (Kent1982); `p_method` row: 'chi2' is the exact weighted chi-square p-value, not Satterthwaite (stale); Pace/Kent as \citep. Port to rsm/pcm (same sentences). Docstring of _selection.composite_andersen fixed the same way.
- slm.tex `global_fit` description (2026-11-08, user editing on Overleaf): Andersen sentence shortened (composite likelihood, Kent1982 weighted chi-square; separate/pooled refits dropped); Wald sentence reworded (difference between groups' estimates, each group's covariance); dangling "the Wald test then uses the CML covariances" removed; split='exogenous' moved to a closing sentence as an omnibus DIF test with \ref{dif_test_slm} (more than two groups for 'andersen' only); p_method row shortened, details (Monte Carlo from pooled fit, adjusted LR Pace2011, exact weighted chi-square, Martin-Löf sparse warning) moved to the description. Port to rsm/pcm/mfrm with their labels.
- slm.tex `calibrate` `ridge` row (2026-11-08, user pasting on Overleaf): "not generally needed ... only pulls the item locations towards 0" qualified: true with complete data; with sparse data plain CML spreads locations out and ridge=0.1 removes that with lower error and keeps every item (sim_slm_edge, 1,920 datasets). Docstring raschpy/slm.py ridge changed the same way.
- slm.tex `global_fit` `mc_exceedances` row (2026-11-08, user pasting on Overleaf): reworded (None runs all no_of_samples; set to 10 when p_method=None chooses Monte Carlo).
- slm.tex `global_fit` (2026-11-08, user on Overleaf): `unused_categories` row removed from the SLM chapter (all three options equivalent for dichotomous items: an item without a CML estimate in a group is left out). Keep the row in rsm/pcm; rewrite it there too (current wording poor).
- slm.tex `global_fit` `global_fit_obsexp` row (2026-11-08, user pasting on Overleaf): reworded to match item_q1_groups (persons, observed and expected mean item scores under the pooled fit, z with variance including item estimation error).
- slm.tex `global_fit` `global_fit_item_table` row (2026-11-08, user pasting on Overleaf): reworded; difference and Wald z only with two groups (as the code does).

- slm.tex local_dependence_gamma (2026-11-08, on Overleaf): p_method row shortened to the option list; how each p-value is computed (permutation variance, Kendall1970; Monte Carlo proportion; Davis1967 asymptotic SE, liberal) moved to the description. Port to rsm/pcm/mfrm.

- slm.tex dif_test purification paragraph (2026-11-08, on Overleaf): removed the wrong "optional Welch correction and small-reference-group size adjustment (pass selection_welch='True')" (no such argument; welch/size_adjust affect only the final tests, not anchor selection); added a sentence on welch (small samples) and size_adjust (large samples); stale "bootstrap standard errors" replaced with std_errors default (sandwich for CML/log-lik, bootstrap for cos/ls/evm) -- one of the PENDING stale-bootstrap fixes. Port to rsm/pcm/mfrm.

## 2026-11-09 (slm.tex downloaded from Overleaf; ported to rsm/pcm/mfrm locally; not yet uploaded)

Per-change status (copied in full / part / not at all, with reasons): manual_port_table_2026-11-09.md. Backup before editing: ~/RaschPy-research/manual_backup_2026-11-09_pre_port.
Fixes to the downloaded slm.tex: global_fit p_method row still had the stale Satterthwaite wording (shortened row + Description sentences applied); dif_test paragraph 'min_diff' -> logit_threshold and 'below' -> 'at most'; fit_statistics description ('which introduces bias', Q1 groups by location, punctuation); Benjamini1995 -> BenjaminiHochberg1995; stale 'bootstrap' SE wording (14 places); progress rows made uniform.
Docstrings: stale 'bootstrap' SE wording fixed in slm.py, rsm.py, pcm.py.
- fit_statistics description (2026-11-09, user's wording): "without involving person locations, whose estimates would introduce bias" -> "which avoids the bias introduced when person estimates are used" (slm, rsm, pcm).
- Cite key BenjaminiHochberg1995 -> Benjamini1995 throughout (2026-11-09, user; first-author format). rsm 1, pcm 3, mfrm 3, slm 2 occurrences.

## DONE 2026-11-10 (was PENDING; applied when the user passed pcm.tex)
- `se` argument rows ("since those do not depend on them" / "fit statistics do not depend on them": those/them undefined; long rows). Replace in every chapter: fit_statistics-type rows (slm 983, rsm 1138, pcm 1254, mfrm 2720 in the local copies) and item/rater-stats rows (slm 1100, 1225; rsm 1255; pcm 1369; mfrm 2899, 3092) with the short rows agreed 2026-11-10.
- Redundant "own" sweep (user, 2026-11-10; standing rule since 2026-10-14, missed in the 2026-11-09 port): local chapters still have slm 5, rsm 6, pcm 19, mfrm 55, mfrm_sim 5, pcm_sim 1, about 1 instances of "own". On the next download, delete every "own" that follows a possessive or its/their unless it marks a contrast the sentence needs; keep idioms that carry meaning ("each table to its own file/sheet", "in its own right", "your own values"). Apply to every chapter, not only the downloaded one.
- When the user passes pcm.tex (2026-11-10 instruction): check the PCM edits, port them to rsm.tex, apply the se-row fix and the "own" sweep (all chapters), then regenerate the change table.

## 2026-11-10 (pcm.tex downloaded from Overleaf; checked, ported to rsm/mfrm locally; not yet uploaded)
Backup before editing: ~/RaschPy-research/manual_backup_2026-11-10_pcm_download. Per-change status: manual_port_table_2026-11-10.md. se rows and 'own' sweep applied in all chapters. CHANGELOG: removed the stale claim that RSM 'cpat-rcl' raises for test='AIC'/'BIC' (checked: it runs the composite test).
- 2026-11-10: 'onto supplied given values' restored (user: 'given' v 'observed' is standard Rasch terminology); my change to 'reference values' reverted in slm, rsm, pcm.

## 2026-11-11 (rsm.tex downloaded from Overleaf; checked, swept, ported locally; not yet uploaded)
Backups: ~/RaschPy-research/manual_backup_2026-11-11_rsm_download (as downloaded), ~/RaschPy-research/manual_backup_2026-11-11_post_port (after this round). Edit scripts and JSON logs: ~/RaschPy-research/manual_edits/cell_sweep_2026-11-11.py, rsm_port_2026-11-11.py. MFRM change table: manual_changes_v2.0.0_mfrm.md (shared-with column pools SLM, PCM and RSM).
- rsm.tex fixes: SE width row grammar ("to correctly reflects" -> ", so that it correctly reflects"); missing spaces in the global_fit and dif_test signatures; remaining "Stage 2" wording removed (8 rows), all threshold_constant rows now "for the CPAT threshold distances".
- "Stage 1/2" removed everywhere (user): rsm.tex; raschpy/rsm.py docstrings and one comment (18); CHANGELOG line 255. "threshold stage" kept (user).
- "cell" sweep (user): rsm 6, pcm 7 (incl. dif_test correction row shortened + Description sentence), mfrm 19 edits. Kept: anchor-check "cells" (API term, defined in step 3; now defined in-sentence at earlier uses), Cells left out column, self.cell_constants, self.global_fit_obsexp_cells.
- Ported from rsm.tex: missing_as_incorrect wording (slm 2, pcm 2, mfrm 2); anchor "line up with the given values" (slm, pcm); MFRM CP-Full-RCL (Varin2011, Newton-Raphson, no "iterative", SLM aside dropped, "threshold comparisons"); MFRM CPAT-RCL paragraph rewritten (also fixes its omission of the item comparisons and "iterative form of CPAT"); MFRM non-iterative cosine; threshold SE "(see Section"; category-width SE row; NaN formatting; MFRM dif_test correction row shortened + Description sentence.
- PAIR-CPAT threshold paragraph (rsm 427, mfrm 989; user-approved): `constant` -> `threshold_constant` in "This stage uses…" through "strongly recommended" (4 each).
- Cite key IglewiczHoaglin1993 -> Iglewicz1993 (user, 2026-11-11): 3 each in slm, pcm, rsm, mfrm.
- "closed-form" removed (user, 2026-11-11): slm 513/pcm 659 cosine example -> non-iterative; main.tex 81 -> non-iterative alternatives; mfrm 451 stretch models "closed-form restriction with genuinely fewer" -> "restriction with fewer"; mfrm 1549 centrality example parenthesis deleted (wrong for the default cp-full-rcl, which fits centrality directly).

## 2026-11-15 (F-reference audit; local edits at the user's request, not yet uploaded)

Backups: before ~/RaschPy-research/manual_backup_2026-11-15_pre_faudit, after ~/RaschPy-research/manual_backup_2026-11-15_post_faudit. Edit scripts: ~/RaschPy-research/manual_edits/e1–e7_*.py (applied with apply.py).

- std_errors (slm, pcm, rsm): Description paragraph on the Kauermann–Carroll correction and t-on-(N − 1) intervals; Usage adds se_correction; interval row shortened; new se_correction row; Returns adds self.se_correction. SLM: "tem names" typo; item_low/item_high typed pandas series.
- global_fit (slm, pcm, rsm): Wald referred to multivariate Welch F (Nel1986); sandwich covariances carry se_correction; welch deprecated (Usage/row welch=None); summary row adds denominator df.
- dif_test (slm, pcm, rsm): Welch t on Welch–Satterthwaite df, columns t and df; welch deprecated; min_anchors default 6 → 2 (Usage and row); size_adjust text no longer conditional on welch; dif_table/threshold_dif_table z → t, df unless welch=False. SLM: duplicated size_adjust paragraph after Usage removed; old "optional Welch correction" sentence replaced.
- calibrate_anchor (slm, pcm, rsm): 'wald' selection on t with N − 1 df; anchor_selection columns (SE (deviation), t, df). RSM: omnibus Wald/Hotelling F (Hotelling1931) replaces LR; per-category t on N − 1; alpha row shortened; deprecated warm_corr/tolerance/max_iters/ext_score_adjustment (Usage None, one row, Description sentence); Returns columns. PCM: Wald/Hotelling F replaces LR; threshold_check and alpha rows shortened; same deprecation; Returns columns.
- fit_statistics (slm, pcm, rsm, mfrm): restscore Monte Carlo paragraph; Usage and new restscore_p_method row; conditional_samples row covers restscore; restscore table columns.
- recovery_stats (slm, pcm, rsm, mfrm): t on N − 1 for item-side and facet element estimates, z for persons (Description); alpha row; Returns columns t/df (persons z); index/None notes moved to the Description. MFRM: "bootstrap standard errors" → std_errors; z → t.
- category_stats Prop disordered (pcm, rsm, mfrm): Description covers bootstrap proportion and t probability; row shortened, guidance moved to the Description.
- drf_test (mfrm): F reference and Welch t sentence; p_method 'chi2' → 'f'; Denominator df in drf_table and drf_item_table.
- local_dependence_gamma (slm, pcm, rsm): 'asymptotic' removed (Description sentence, p_method row). Davis1967 no longer cited.
- new_references.bib: Nel1986, Hotelling1931 appended.

## 2026-11-16 (aic_sig_test → aic_relative_likelihood; local edits, not yet uploaded)

Backup before: ~/RaschPy-research/manual_backup_2026-11-16_pre_aic. Edit script: ~/RaschPy-research/manual_edits/e8_aic_relative_likelihood.py (run on the backup copies).

- model_selection (rsm, pcm): Description AIC item: the formal-significance-test sentence (p = P(χ²_df ≥ ΔAIC + 2df)) replaced: PCM preferred only if its AIC is lower and RSM's relative likelihood (1 when RSM's AIC is the lower) is at most alpha; not a p-value; with a likelihood, an LR test at P(χ²_df > 2df + 2 ln(1/α)), levels 0.0063 / 0.00095 / 2.2e-6 at df 6 / 19 / 57; suits an unconditionally calculated likelihood, more conservative than test='LR' with conditional/composite likelihoods. Composite item, Usage, test/alpha rows: new name (alpha row: "threshold for the relative likelihood"). New aic_relative_likelihood row (short) + deprecated aic_sig_test row. Returns: aic_summary rows Relative likelihood/Test level; _aic_p row → _aic_relative_likelihood, _aic_level, _aic_p (v1.4.0 name); gate row "behind the decision".
- model_selection (mfrm): new Description paragraph (baseline, relative likelihood, LR-test level, rationale, cp-full-rcl nested/non-nested, moved from the old argument row); Usage (alpha=0.05 moved to the second line); argument row shortened + deprecated row; alpha row; Returns: _aic_relative_likelihood, _aic_level, _aic_p; gate row.
- per_rater_model_selection (mfrm): new Description paragraph (same rule; table columns Relative likelihood/Test level, or relative_likelihood_vs_baseline/test_level with pair-cpat and anchors); "significance test against the simplest active baseline" → "relative likelihood of the simplest active baseline"; Usage; argument row (also fixes the "rewuiring … p > 0.95" text) + deprecated row; alpha row; candidate_models cross-reference.
- NOT changed: per_rater_model_selection_table row still says the anchors' rows carry a Note that they are held on 'global' — out of date with the anchoring work in progress (Note now only with force_global_anchors=True).

## 2026-11-16 (per-rater LR ladder: stretch step; local edit, not yet uploaded)

Backup before: ~/Documents/RaschPy_manual/mfrm_backup_2026-11-16_ladder.tex.

- per_rater_model_selection (mfrm), Description, second-stage list, 'global winner' item: when both centrality and pseudo_halo beat global, global is ruled out and each is tested against bistretch (df 1 each), as items and thresholds against bivector; bistretch if both rejected, the one not rejected if only one is, else the one that beat global with the smaller p. Replaces the global v bistretch test.
- NOT yet done: rows for the new ridge arguments (ridge= in model_selection for rsm/pcm; global_ridge, item_ridge, threshold_ridge, interaction_ridge, stretch_ridge in model_selection and per_rater_model_selection for mfrm); the pair-cpat table columns LR_glb_vs_bis/p_glb_vs_bis are now LR_cen_vs_bis/p_cen_vs_bis and LR_pha_vs_bis/p_pha_vs_bis (check whether the manual lists them).

## 2026-11-16 (stretch_delta; code only, manual NOT yet edited)

- per_rater_model_selection (mfrm): new argument stretch_delta (float, default 0). On the rungs whose larger representation adds a single stretch factor (centrality > global, bistretch > pseudo_halo: threshold stretch; pseudo_halo > global, bistretch > centrality: item stretch), the departure is the change in that stretch factor between the two fits (|factor − 1|, relative to the frame), a proportion (0.1 = up to 10% stretch); delta (widths) applies to the other rungs. Those rungs' Delta-hat columns are in the same proportion. per_rater_model_selection_info gains stretch_delta. Needs test='LR' and cp-full-rcl/cpat-rcl/cml.
- The manual has no delta / delta_units rows yet either: add delta, delta_units and stretch_delta rows (and the Delta-hat table columns) together.

## 2026-11-17 (LR ladder; code + local mfrm.tex, backup mfrm_backup_2026-11-17_buildup.tex)

- per_rater_model_selection (mfrm): stretch_delta now also takes {'items': x, 'thresholds': y} (separate item- and threshold-stretch tolerances; a key left out is 0), because the same proportion is a much larger effect on the thresholds than on the item locations (Rome: item SD 0.56 logits, threshold RMS 4.46). per_rater_model_selection_info gains item_stretch_delta and threshold_stretch_delta. Manual row for stretch_delta: type float or dict.
- Top-down LR ladder: when it ends on a single stretch (centrality or pseudo_halo), it now also tests the other side's stretch (bistretch over it).
- Build-up LR ladder from 'global' is now the default for cp-full-rcl / cpat-rcl / cml (research switch mfrm._LR_ORDER = "up"; "down" = old top-down). Per side (items, thresholds): nothing / stretch / full set, closed-testing order (a side leaves nothing only if its stretch or full set beats global; then full set if it beats the stretch). Joint check global v bistretch when both sides stay at nothing. Two hidden representations (item effects + threshold stretch, threshold effects + item stretch) are fitted for the tests only and reported, fitted and used as bivector (until v2.3). Then matrix v bivector.
- Lone-anchor tests: an anchor that alone among the anchors lacks a part has an empty test of that part; once the passes settle it is retested against the anchors' average, with a one-line Note per facet element.
- Top-down bounce (other stretch beside a single stretch) kept for the "down" order only. Tried for pair-cpat (switch mfrm._PAIR_CPAT_OTHER_STRETCH): mean hit rate +0.02 but worst case 0.56 -> 0.52 and losses for centrality, items, pseudo_halo raters (changes reach other raters through the joint fits), so left OFF.
- mfrm.tex per_rater_model_selection: Description paragraph on the build-up ladder, the lone-anchor retest and the close-fit tests (delta, delta_units, stretch_delta); test row no longer says LR is pair-cpat only; Usage gains delta, delta_units, stretch_delta; three short Argument rows.
- NOT done, pre-existing gap: the Description still says anchors are held on 'global' and not selected; the code now selects them (force_global_anchors=False default). Needs a rewrite in the MFRM manual review.
- AIC relative-likelihood check steps down (next-lowest AIC checked against the baseline until one passes) in model_selection, per_rater_model_selection fitted and pair-cpat paths. Only changes results under cp-full-rcl (per-model scaling of the check); under a likelihood the lowest-AIC model has the lowest relative likelihood, so nothing changes. mfrm.tex: one sentence pair added to each of the two AIC paragraphs (model_selection, per_rater_model_selection).
- per_rater_model_selection_table no longer shows columns for the ladder-only representations (items_centrality, thresholds_pseudo_halo).
- per_rater_model_selection: new delta_measure ('rms' default, 'max', 'both') and delta_max (largest single deviation, purified, Holm); AIC/BIC accept delta / stretch_delta / delta_max as a step-down gate ('Stepped down from' column). mfrm.tex (backup mfrm_backup_2026-11-17_maxmeasure.tex): Usage line, delta/stretch_delta rows no longer LR-only, delta_units row mentions delta_max, two new short rows, Description paragraphs on the max measure and the AIC/BIC step-down.

## 2026-11-18 — delta_measure 'both' renamed 'either'
- per_rater_model_selection: delta_measure='both' renamed 'either' (the larger representation is kept if either the RMS or the largest-deviation test rejects; the old name read as both having to reject). Unshipped, so no alias.
- mfrm.tex: lines 2606 (max-measure paragraph), 2608 (AIC/BIC step-down paragraph), 2678 (delta_measure row). Backup: mfrm_backup_2026-11-18_either.tex.
- Also: mfrm.py docstrings/validation, tests/test_anchor_check.py, CHANGELOG.md.

## 2026-11-18 — DRF global-effect-only fallback
- mfrm.tex `drf_test` description (line ~4595, after "The Note columns say which categories and groups were short."): added the sentence on facet elements whose item and threshold effects cannot be estimated (global effect still tested as the mean over the remaining item-threshold pairs; other tests not run; left out of the others' comparisons and the omnibus test). Backup: mfrm_backup_2026-11-18_drf_level.tex.
- No manual text describes the unused-category warning itself, so the bottom/top-category extension needed no manual change.

## 2026-11-18 — per_rater_model_selection tolerance defaults
- mfrm.tex (backup ~/RaschPy-research/backups/mfrm_tex_backup_2026-11-18_defaults.tex): Usage line (delta, stretch_delta, delta_measure, delta_max default None); the four argument rows (type/default None).
- Description, max-measure paragraph: 'either' now tests both at alpha (step's p = the smaller of the two), with the simulation result (false rejections up by at most 0.013 over the RMS test alone).
- Description, new paragraph before the AIC/BIC step-down paragraph: defaults (LR: delta 0.1, delta_max 0.2 widths, stretch_delta 0.1, 'either'; AIC/BIC: off); defaults suit typical rater-severity work, set stricter/looser as needed; keep delta_max ≈ 2 × delta and why; no-width fallback to the plain tests with a warning.
- Also: mfrm.py (signature, defaults resolution, fallback, either at alpha, docstrings), base.py comment on _POWER_DELTA_DEFAULT, CHANGELOG.md.

## 2026-11-19 — per_rater_model_selection: tolerance defaults for AIC/BIC too (mfrm.tex)
- Defaults paragraph: the tolerance defaults now apply with every test ('cp-full-rcl', 'cpat-rcl', 'cml'); removed "with test='AIC' or 'BIC' they are 0 and 'rms' (no tolerances)".
- Fixed a literal "\n\n" (backslash-n characters) before the "With test='AIC' or 'BIC' and a tolerance above 0" paragraph: now a real paragraph break.
- AIC/BIC step-down paragraph: added that it compares estimated departures (not significance), the simulation rates, and that test='LR' stays closer to its nominal level.

## 2026-11-19 — RSM/PCM model_selection: per-item close-fit test, delta/sampling documented
- rsm.tex, pcm.tex (backups rsm_backup_2026-11-19_itemrms.tex, pcm_backup_2026-11-19_itemrms.tex), section model_selection:
  - Description: new "Close-fit test" paragraph (delta, delta_units, delta_measure 'overall'/'item'/'either', simulation rates) and "Sampling" paragraph (sampling='power', sampling=n, n_subsamples).
  - Usage: delta, delta_units, delta_measure, sampling, power_threshold, n_subsamples added.
  - Arguments: rows for delta, delta_units, delta_measure, sampling, power_threshold, n_subsamples; 'sampling' removed from the deprecated row (it was wrongly listed as deprecated).
  - Returns: close_fit_p / close_fit_p_item and item_close_fit rows.

## 2026-11-19 — MFRM excluded_models
- mfrm.tex (backup mfrm_backup_2026-11-19_excluded.tex): model_selection and per_rater_model_selection — Usage line `excluded_models=None` added after candidate_models; Arguments row `excluded_models` added after the candidate_models row in both.

## 2026-11-19 — MFRM usage lines rebuilt from the signatures
- mfrm.tex (backup mfrm_backup_2026-11-19_usage.tex): model_selection and per_rater_model_selection Usage blocks rewritten in signature order, deprecated names left out (models, aic_sig_test; and for model_selection sampling, likelihood, warm_corr, tolerance, max_iters, ext_score_adjustment). Added: estimation, the five ridges, excluded_models (both); force_global_anchors (per_rater). Removed per_rater's duplicated threshold_constant and robust; progress moved to the end.
- Not yet done: Arguments rows are missing for model_selection's estimation and ridges, and per_rater's force_global_anchors and ridges.

## 2026-11-19 — MFRM missing argument rows
- mfrm.tex (backup mfrm_backup_2026-11-19_rows.tex):
  - model_selection: Description sentence on estimation rewritten (it said candidates must share one estimation and are calibrated with the existing one; the code has `estimation=` and recalibrates mismatched candidates with a UserWarning); new Description paragraph on the five ridge arguments; Arguments rows for `estimation` and the ridges (before `progress`).
  - per_rater_model_selection: new Description paragraphs on the ridges and on `force_global_anchors`; Arguments rows for `force_global_anchors` and the ridges (before `progress`).

## 2026-11-19 — model selection build-up (mfrm.tex; backup mfrm_backup_2026-11-19_buildup.tex)
- `model_selection`: Description's AIC paragraph replaced by the build-up ladder, the per-step relative likelihood, the close-fit test (delta, stretch_delta, delta_units) and the default-0.2 paragraph with simulation rates; Usage adds delta, delta_units, stretch_delta; Arguments: test row (AIC/BIC build up), new rows delta, delta_units, stretch_delta, aic_relative_likelihood row (smaller representation at each step), alpha row (close-fit tests); Returns: LR row (departure, close-fit p), AIC row rebuilt (adds _aic_steps; drops _aic_baseline, _aic_relative_likelihood, _aic_level, _aic_p), _aic_gate row removed, BIC row (adds _bic_steps), info row (tolerances).
- `per_rater_model_selection`: AIC/BIC step-down paragraph replaced by the build-up rule with simulation rates; AIC relative-likelihood paragraph split into the ladder rule and the 'pair-cpat' baseline rule; candidate_models row (ladder skips steps; direct filter for 'pair-cpat'); aic_relative_likelihood row (simpler representation).

## 2026-11-19 — RSM/PCM model_selection default AIC + close fit (rsm.tex, pcm.tex; backups rsm_/pcm_backup_2026-11-19_aic_close.tex)
- Both `model_selection` sections, same edits:
  - Description: new paragraph after the criteria list (test=None resolves to 'AIC', or 'LR' for the bootstrap test or with sampling; model_comparison_rsm_pcm_test; every criterion needs the close-fit test).
  - Close-fit paragraph: applies with every test (LR decides by it; AIC/BIC must also reject); delta=None = 0.1 widths, 0 with sampling=n or bootstrap, warning + 0 when widths unusable.
  - New paragraph on the misspecification simulations (plain LR, AIC, AIC+RL, default; BIC never).
  - Sampling: test='LR' only, test=None resolves to it.
  - Usage: test=None. Arguments: test, delta, alpha rows rewritten.
  - Returns: new model_comparison_rsm_pcm_test row; close-fit p row "with any test"; AIC and BIC summary rows mention delta/close-fit rows.
  - Examples: default (AIC summary) and LR with delta=0.
- Not compiled locally (no LaTeX here): check on Overleaf.

## 2026-11-19 — delta_measure option names aligned: 'overall', 'largest', 'either' (backups *_backup_2026-11-19_measure_names.tex)
- rsm.tex, pcm.tex (model_selection): 'item' → 'largest' in the Close-fit list item (now "whether any one item departs: …"), the delta_measure Arguments row, and the item_close_fit Returns row.
- mfrm.tex (per_rater_model_selection): Description 'max' → 'largest', table columns p RMS / p max → p overall / p largest; delta_measure Arguments row 'rms', 'max' → 'overall', 'largest'.

## 2026-11-19 — min_effect documented as the size guard where there is no close-fit test (backups *_backup_2026-11-19_min_effect.tex)
- rsm.tex, pcm.tex (model_selection): sentence added to the min_effect Description paragraph (bootstrap routes; delta preferred elsewhere; why).
- mfrm.tex: new paragraph after model_selection's tolerance-defaults paragraph ('cml', 'pair-cpat'); sentence appended to per_rater_model_selection's "At every rung" paragraph ('pair-cpat'). Argument rows unchanged.

## 2026-11-21 — sweep of model_selection (RSM, PCM, MFRM), per_rater_model_selection and check_anchor_homogeneity (backups in ~/RaschPy-research/manual_backup_2026-11-21_selection_sweep/; scripts ~/RaschPy-research/manual_edits/sweep_1121_*.py)

Applied to the local .tex files; upload to Overleaf and check.

### rsm.tex / pcm.tex — model_selection
- New `ridge` argument row (after `n_subsamples`) and `ridge=None` in Usage (RSM Usage reflowed to three lines). RSM: None uses PCM's default; PCM: None uses the object's ridge.
- AIC bullet: df of the implied LR level now $(I - 1)(m - 1)$ (was $(I - 1)(K - 1)$, a symbol not defined there).
- BIC bullet: "there is no relative likelihood check for BIC as there is for AIC" (was "no significance test for BIC like the AIC significance test").
- min_effect paragraph rewritten to match the code (backups in ~/RaschPy-research/manual_backup_2026-11-21_min_effect/; script sweep_1121_min_effect.py): min_effect is compared with Delta-hat, the bias-corrected overall departure of PCM from RSM in delta_units; a ValueError with the bootstrap test (PAIR). Was: largest absolute threshold difference in logits, PAIR's size guard, ETS values.
- min_effect row: "Float: the smallest estimated departure of PCM from RSM, in delta_units, needed to prefer PCM (see Description). Default is min_effect=0 (no minimum)."
- effect attribute row: largest absolute threshold difference in logits, reported only. New row self.model_comparison_rsm_pcm_delta_hat.

### mfrm.tex — model_selection
- `test`, `candidate_models`, `min_effect` rows cut to type/options/default; the LR nested pairs, test=None resolution, nesting requirement and same-model leave-out rule moved to a new Description paragraph; min_effect's definition, LR condition, AIC/BIC walk-back and ETS values moved into the min_effect paragraph.
- Example: "AIC with the relative likelihood check" (was "with the significance test").

### mfrm.tex — per_rater_model_selection
- Anchors paragraph: anchors start on global and are selected like the other facet elements (force_global_anchors=True holds them); anchor sets can give different valid assignments (moved from the anchors row); check.suggested may drop anchors, free anchors' patterns (global-effects-only anchors) or cut cells. Was: anchors held on global and not selected.
- New candidate_models Description paragraph (moved from the row; same-model leave-out list repeated rather than referenced).
- Lone anchor: tested against the anchors' average whenever its part test would compare a model with itself, with every test (LR, AIC, BIC). Was: only after the passes settle.
- min_effect paragraph: ignored with a UserWarning under cp-full-rcl/cpat-rcl/cml (delta, delta_max, stretch_delta do the job); definition and ETS values for pair-cpat. Was: claimed min_effect applied at every LR rung.
- pair-cpat paragraph: all facet elements including anchors are tested; robust=True details (moved from row).
- force_global_anchors: anchors marked to anchor through their global effects only are still selected.
- Rows cut to type/options/default: anchors, candidate_models, min_effect ('pair-cpat' only; ignored with a UserWarning otherwise), robust. '(pair-cpat only)' added to sampling, seed, person-estimation and calibration argument rows.
- Returns table: anchors' rows on 'global' with a "held there" Note only with force_global_anchors=True; Note also marks anchors tested against the anchors' average.
- Example: LR ladder with delta=0.2 (was min_effect=0.3, which is ignored under the default estimation).

### mfrm.tex — calibrate
- `anchors` row cut to type/options/default; its cut-cell text moved to the model='mixed' Description paragraph, plus global-effects-only anchors (give them a representation with a pattern; UserWarning if put on 'global').

### mfrm.tex — check_anchor_homogeneity (rewritten for the two-step check)
- Description: the five-step check (pattern screen, cells, global effect screen, refinement, split_alpha) replaced by the two steps (global effect subset search; RMS pattern test within the suggested set), the overall test (reported, does not drive suggestions), Tolerance (close-fit tests, delta in widths), the docstring's simulations, the overall Wald test's Hotelling reference. The split_alpha simulation figures and the small-sample figures for the old check are removed (they describe the old check). test=None resolution and the TypeError/UserWarning rules moved from the row to the Description.
- Usage: split_alpha and max_share removed; delta, delta_units, report added.
- Arguments: split_alpha and max_share rows removed; rater_homogeneity_test, min_effect, action ('report'/'apply'), min_anchors rows rewritten; delta, delta_units, report rows added; anchors row names check.suggested.
- Returns: check.screen removed; check.global_effects, check.pattern, check.ties, check.pattern_ties, check.profile, check.subsets, check.free_pattern, check.hold_pattern added; suggestions' actions now 'drop'/'free pattern'; restore(anchor) without cell; apply_suggested(choice=0).
- Examples: drop + free_pattern by hand; delta=0.2 with action='apply' (was min_effect=0.2, action='cells').

## 2026-11-21 — check_anchor_homogeneity: information-weighted pattern, whole-panel widths (mfrm.tex; backup ~/RaschPy-research/manual_backup_2026-11-21_weighted_check/; script ~/RaschPy-research/manual_edits/weighted_check_1121.py)
- Description, Pattern step: each cell weighted by the inverse sampling variance of the anchor's location there (as per_rater_model_selection sizes a departure); weighted centring.
- Tolerance paragraph: widths from the joint fit of every facet element (anchors on global, others on matrix), anchored on the anchors given (global effects) and on the suggested set (pattern); pattern test's sum of squares now weighted.
- Simulation paragraph rewritten for the weighted check (10 datasets per departure scenario; adds middle- and top-threshold cell cases). Also fixes a broken ``wide''/``narrow'' quote construct.
- min_effect row: "weighted RMS"; delta_units row: widths of the joint fit of every facet element; check.pattern row: RMS information-weighted, check.pattern.attrs['width'].
- FIX of my own earlier edit today (check rewrite): the "Nothing is changed by the check..." paragraph had inverted code formatting (plain text inside \texttt, code names outside), caused by the broken quote construct above; rewritten with code names in grey typewriter.

## 2026-11-21 — check_anchor_homogeneity: pattern step = per_rater_model_selection's tests on the suggested set (mfrm.tex; backup ~/RaschPy-research/manual_backup_2026-11-21_select_pattern/; script ~/RaschPy-research/manual_edits/select_pattern_1121.py)
- Description, Pattern step: per_rater_model_selection(scheme='one-pass') on the suggested set's responses, the set as anchors, other members on 'global'; check's delta (in logits via the width of the fit anchored on the set), delta_max = 2 delta, stretch_delta default, check's alpha and min_effect; anchor selected off 'global' is pattern free. Anchors with cut cells: close-fit RMS test over the cells not cut (they stay on 'global' in the selection). The weighted RMS is now the reported size, not the test.
- Tolerance paragraph: pattern tests are the selection's; the weighted noncentral chi-square reference now applies to anchors with cut cells only.
- Simulation paragraph: new figures (null 4/400 pattern-free, 14/1199 no-departure anchors; wide: item, thresholds and middle cell 10/10 at both N, top cell 2/10 and 0/10; narrow 1–3/10 at N 300, 3–7/10 at N 800; 99.5% agreement with per_rater_model_selection on every rater's responses).
- min_effect row shortened to type/default ("the smallest effect to act on, in logits"); detail now in the Description steps.
- check.pattern row: p replaced by Representation (selected); p for anchors with cut cells only. check.suggestions row: Representation for a free pattern.

## 2026-11-21 — per_rater_model_selection: AIC relative likelihood rescaled under cpat-rcl too (mfrm.tex; backup ~/RaschPy-research/manual_backup_2026-11-21_cpat_rl/; script ~/RaschPy-research/manual_edits/cpat_rl_1121.py)
- Description, AIC relative likelihood paragraph: "with 'cp-full-rcl' and 'cpat-rcl' (composite likelihoods), Δ is first divided by ..." (was 'cp-full-rcl' only; code fixed to match).

## 2026-11-23 — model_selection fixes + sampling; per_rater_model_selection remodelled (mfrm.tex, user's upload of 2026-11-23; backup ~/RaschPy-research/manual_backup_2026-11-23_user_upload/mfrm.tex; scripts ~/RaschPy-research/manual_edits/ms_fixes_2026-11-23.py, per_rater_remodel_2026-11-23.py)
- calibrate (user's edits, lines ~973–978): checked, no change.
- model_selection: backtick 'pair-cpat' put in grey typewriter; 'cml' added to the min_effect paragraph's close-fit list; duplicate ridge paragraph before Usage removed; added "estimation=None uses the estimation of the object's latest calibration ('cp-full-rcl' if there is none)"; new sampling paragraph (refit on n persons per subsample, n_subsamples, 'dynamic' rule, not with delta/stretch_delta); Usage gains sampling=None, n_subsamples=1; sampling removed from the deprecated row; seed row covers subsamples; new sampling and n_subsamples rows; new Returns row self.model_comparison_mfrm_sampling.
- per_rater_model_selection Description rewritten on the user's model_selection style: what it does, estimation and ridge; anchors/force_global_anchors condensed; tests by estimation; compact ladder (bistretch re-examination, full-set-beside-stretch reported as bivector, anchor-against-anchors'-average Note kept); AIC relative likelihood; close-fit "To avoid oversensitivity..." paragraph; delta_measure/delta_max without simulation figures (one sentence on delta_max ≈ 2 delta); min_effect; candidate_models/equivalences; pair-cpat two-stage ladder; robust; downstream methods. Removed: simulation-figure sentences (0.013 rise; AIC 8%/BIC 7% v step-down) and the long ratio rationale.
- per_rater rows delta, delta_units, stretch_delta, delta_measure, delta_max: "Type: what. Default ..., which resolves to ...". sampling row untouched (per_rater pair-cpat level check pending).

## 2026-11-23 — RSM/PCM model_selection: sampling='dynamic' restored as a refit (rsm.tex, pcm.tex; backup ~/RaschPy-research/manual_backup_2026-11-23_rsm_dynamic/)
- Sampling paragraph: added "sampling='dynamic' does the same with n = min(20(I−1)(m−1), 1500) ...".
- Close-fit paragraph and delta row: delta=None is 0 with sampling=n, 'dynamic' or the bootstrap test.
- sampling row: options now None, 'power', 'dynamic' or integer; n_subsamples row: for sampling=n or 'dynamic'.

## 2026-11-23 — mfrm.tex, per_rater_model_selection Description (tests sentence)
Backup: ~/RaschPy-research/manual_backup_2026-11-23_tests_wording/mfrm.tex.before
"the tests are those of model_selection: [LR only]" → "each step is decided as in model_selection, by the likelihood ratio test (test='LR') or by AIC or BIC (test='AIC', the default, or 'BIC')", then the LR reference distributions, plus AIC/BIC use the composite likelihood's effective number of parameters (Varin2005) for cp-full-rcl / cpat-rcl.

## 2026-11-23 — mfrm.tex, per_rater_model_selection Description (restructured)
Backup: ~/RaschPy-research/manual_backup_2026-11-23_tests_wording/mfrm.tex.before_restructure
Third paragraph + AIC relative-likelihood paragraph rewritten as: (1) the ladder paragraph (+ anchor Note sentence); (2) "three ways to decide each step, set by test" itemize: AIC (default, with the relative-likelihood rule and its test level), BIC, LR (cml chi-square / composite Kent 1982); (3) a closing sentence: AIC/BIC likelihoods per estimation (Varin2005), and how a tolerance combines with each.

## 2026-11-23 evening — user returned mfrm.tex (Overleaf edits)
Copy: ~/RaschPy-research/manual_backup_2026-11-23_tests_wording/mfrm.tex.user_return_2026-11-23. No 'penalty minus' wording left in mfrm.tex.
Docstrings (mfrm.py, model_selection + per_rater_model_selection aic_relative_likelihood): "penalty minus the smaller's per extra free parameter" → "effective number of parameters minus the smaller's, divided by the difference in free parameters". Backup: ~/RaschPy-research/mfrm_kc/backup_1122/mfrm.py.before_penalty_wording

## 2026-11-23 — mfrm.tex, per_rater_model_selection Description fixes on the user's returned text
Backup: ~/RaschPy-research/manual_backup_2026-11-23_tests_wording/mfrm.tex.before_desc_fixes
1. Para 1: restored what the method does for cp-full-rcl/cpat-rcl/cml (each candidate assessed by fitting the full model, others kept at current representations; final model fitted and stored as mixed), in the user's preferred plain wording.
2. Ladder paragraph now opens "With 'cp-full-rcl', 'cpat-rcl' or 'cml', ..." (pair-cpat differs).
3. Anchors paragraph rewritten: wrong "unless the anchor is rejected" → global-effects-only anchors still selected; restored sole anchor stays on global and what check.suggested can do; fixed repetition and punctuation.
4. Lone-anchor paragraph: "). i.e.\ whose" → "), whose".
5. Para 1: "most parsimonious representation that explains its effects" → "a facet element with complex behaviour does not force the others into an overparameterised representation".

## 2026-11-23 late — user returned mfrm.tex, rsm.tex, pcm.tex + RaschPy_manual_2_0_0.pdf
Copies: ~/RaschPy-research/manual_backup_2026-11-23_tests_wording/*.user_return2_2026-11-23. Checked against pre-edit backups: all my edits present.
- mfrm.tex, user's changes: para 1 "a facet element with complex behaviour does not" → "one or two facet elements with complex behaviour do not"; anchors paragraph drops "the facet elements whose average defines it (None: all facet elements)"; layout only otherwise (\newpage, \\).
- rsm.tex, pcm.tex: one \newpage added each, no wording changes.
- PDF placed in GitHub/RaschPy (main checkout) at the repo root and raschpy/docs/ (identical); not in the RaschPy-dev worktree.
- PDF copied to RaschPy-dev/raschpy/docs/ (user approved).

## 2026-11-24 — mfrm.tex: se_correction documented (std_errors gap) + new se_correction= on model_selection, per_rater_model_selection, check_anchor_homogeneity
Backup: ~/RaschPy-research/mfrm_kc/backup_1124_se_correction_arg/mfrm.tex
- std_errors: Description paragraph on the Kauermann-Carroll correction (ported from slm.tex) + which methods follow it; se_correction added to all 8 Usage blocks; se_correction Arguments row; self.se_correction Returns row (per representation se_correction_{model}).
- model_selection, per_rater_model_selection: Description paragraph before Usage (tests use KC with cp-full-rcl/cpat-rcl; None faster, somewhat more liberal in small samples); Usage + Arguments row.
- check_anchor_homogeneity: Description paragraph before Usage (Wald/LR/close-fit, pattern step, check.cut refits); joint-fit Usage + Arguments row.

## 2026-11-24 later — se_correction default None for model_selection, per_rater_model_selection, check_anchor_homogeneity (mfrm.tex)
Backup: ~/RaschPy-research/mfrm_kc/backup_1124_se_correction_default_off/mfrm.tex
- The four KC paragraphs added earlier today replaced by one sentence each (user asked for minimal text with the time cost):
  std_errors: correction on by default; without it SEs understated in small samples.
  model_selection, per_rater_model_selection: "With 'cp-full-rcl' and 'cpat-rcl', se_correction='kauermann-carroll' applies the Kauermann--Carroll small-sample correction (Section std_errors) to the tests; it may help with very small sample sizes, where its time cost is also small, but with many persons and parameters it can take most of the run time."
  check_anchor_homogeneity: same without the estimation clause.
- Usage lines and Arguments rows of the three methods: default se_correction=None.

## 2026-11-24 — rater_stats_df marginal default None (mfrm.tex)
Backup: ~/RaschPy-research/backup_1124_mixed_marginal/mfrm.tex
- rater_stats_df Usage (line ~3433): marginal=True -> marginal=None.
- Arguments row `marginal`: "Boolean or None: If True or None, marginal mean facet effect vectors ... Default is marginal=None."
- Reason: the default call with model='mixed' warned; 'mixed' now warns only when marginal=True is passed (sentence at ~3566 unchanged, still accurate).
