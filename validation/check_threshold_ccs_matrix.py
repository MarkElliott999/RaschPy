"""
Visual check: threshold CCS obs alignment for MFRM matrix model.

Generates threshold CCS plots for the matrix model with obs overlaid,
for a range of item/rater combinations. Save to PNG files for inspection.

Checks:
  - rater=None (neutral/mean curve)
  - rater specified
  - item specified vs item=None
"""

import sys
import warnings
import numpy as np

sys.path.insert(0, '/Users/markelliott/Documents/GitHub/RaschPy')
import raschpy as rp
from raschpy.simulation.mfrm_sim import MFRM_Sim

warnings.filterwarnings('ignore')
np.random.seed(42)

N_ITEMS   = 6
N_PERSONS = 500   # more persons → cleaner obs points
N_RATERS  = 4
MAX_SCORE = 3

sim = MFRM_Sim(N_ITEMS, N_PERSONS, N_RATERS, MAX_SCORE,
               manual_thresholds=[0, -1.5, 0.0, 1.5],  # wider spacing
               model='matrix', rater_range=2)
m = rp.MFRM(sim.scores, max_score=MAX_SCORE)
m.calibrate(model='matrix')

rater_1 = list(m.raters)[0]
rater_2 = list(m.raters)[1]
item_1  = list(m.items)[0]

plots = [
    # (description, kwargs)
    ('matrix_neutral_no_item',
     dict(model='matrix', obs='all', no_of_classes=5)),

    ('matrix_neutral_item_specified',
     dict(model='matrix', item=item_1, obs='all', no_of_classes=10)),

    ('matrix_rater1_no_item',
     dict(model='matrix', rater=rater_1, obs='all', no_of_classes=10)),

    ('matrix_rater1_item_specified',
     dict(model='matrix', rater=rater_1, item=item_1, obs='all', no_of_classes=10)),

    ('matrix_rater2_item_specified',
     dict(model='matrix', rater=rater_2, item=item_1, obs='all', no_of_classes=10)),
]

for name, kw in plots:
    filename = f'/Users/markelliott/Documents/GitHub/RaschPy/{name}'
    try:
        m.threshold_ccs(filename=filename, file_format='png', dpi=150, **kw)
        print(f'  Saved: {name}.png')
    except Exception as e:
        import traceback
        print(f'  FAILED: {name}')
        traceback.print_exc()

print('\nDone. Open the PNG files to inspect obs alignment.')
