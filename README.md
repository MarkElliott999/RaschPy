# RaschPy
_RaschPy_ is a Python package for Rasch analysis which can estimate parameters for a variety of Rasch models, generate a range of model fit statistics and output tables and graphical plots. \textit{RaschPy} also contains simulation functionality (used for the simulations in this work). \textit{RaschPy} is open source and free to download. Specifications are subject to change as the software is developed; the details listed here are correct at the time of writing (January 2024). The following is not intended to be a user manual, or even fully comprehensive at the time of writing, but rather to highlight the main functionality of _RaschPy_. A full navigable manual is available in the GitHub repository. An Excel spreadsheet demonstrating the PAIR algorithm for dichotomous data is also available, following the example of the Moulton JMLE dichotomous demo available via https://www.rasch.org/moulton.htm  (and using the same set of responses -- the final results are compared to the Moulton JMLE output). A polytomous example will be added later.

## Models
_RaschPy_ has a parent class `Rasch` for analysis, with the following child classes for different Rasch models:
- `SLM` for the simple logistic model (dichotomous Rasch model) (Rasch 1960)
- `PCM` for the partial credit model (Masters 1982)
- `RSM` for the rating scale model (Andrich 1978)
- `MFRM` for the many-facet Rasch model (rating scale model formulation) (Linacre 1994), including extended rater representations (Elliott and Buttery 2022a)

## Analysis
To analyse data, create an object of in the appropriate class, passing a pandas dataframe of response data as an argument along with other arguments relevant to the chosen Rasch model, such as the maximum score for `RSM` or `MFRM`, or a vector of maximum scores for \textcolor{gray}{\texttt{PCM}}. At the time of writing, the `RSM` and `MFRM` classes only support a single response group (i.e. all items must have the same threshold structure), and the `MFRM` class only supports one additional facet for rater severity. Parameter estimation uses variants of PAIR (Choppin 1968, 1985), the eigenvector method (Garner & Engelhard 2002, 2009) and CPAT (Elliott & Buttery 2022a, 2022b).

## Usage and citation
_RaschPy_ is provided as freeware under an Apache 2.0 Licence (see LICENSE file in this repository for details). Users are free to use or modify the code for their own purposes, but should cite using the following format:

Elliott, M (2024) _RaschPy_ Rasch analysis software. Downloaded from: https://github.com/MarkElliott999/RaschPy

## References
Andrich, D. (1978). A rating formulation for ordered response categories. _Psychometrika_, _43_(4), 561–573.

Choppin, B. (1968). Item bank using sample-free calibration. _Nature_, _219_(5156), 870–872.

Choppin, B. (1985). A fully conditional estimation procedure for Rasch model parameters. _Evaluation in Education_, _9_(1), 29–42.

Elliott, M., & Buttery, P. J. (2022a). Extended rater representations in the many-facet Rasch model. _Journal of Applied Measurement_, _22_(1), 133–160.

Elliott, M., & Buttery, P. J. (2022b). Non-iterative conditional pairwise estimation for the rating scale model. _Educational and Psychological Measurement_, _82_(5), 989–1019.

Garner, M., & Engelhard, G. (2002). An eigenvector method for estimating item parameters of the dichotomous and polytomous Rasch models. _Journal of Applied Measurement_, 3(2), 107–128.

Garner, M., & Engelhard, G. (2009). Using paired comparison matrices to estimate parameters of the partial credit Rasch measurement model for rater-mediated assessments. _Journal of Applied Measurement_, _10_(1), 30–41.

Linacre, J. M. (1994). _Many-Facet Rasch Measurement_. MESA Press.

Masters, G. N. (1982). A Rasch model for partial credit scoring. _Psychometrika_, _47_(2), 149–174.

Rasch, G. (1960). _Probabilistic models for some intelligence and attainment tests_. Danmarks Pædagogiske Institut.

## DISCLAIMER
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
