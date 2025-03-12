import importlib
import main
importlib.reload(main)
from main import main
import itertools


TEST=True

if TEST: 
    LARGEST_TOTAL_COMPUTE_RATIO=[0.1,0.2]
    COMPUTE_GROWTH_RATE=[3.0,5.0] #we're going to set g_global_AI_compute to this
    FM_FIXED=[0.9,1.1]

if not TEST:
    LARGEST_TOTAL_COMPUTE_RATIO=[0.05,0.1,0.2,0.3,0.4]
    COMPUTE_GROWTH_RATE=[1.0,3.0,5.0,7.0] #we're going to set g_global_AI_compute to this
    FM_FIXED=[0.9,1.0,1.1]


param_combinations = list(itertools.product(LARGEST_TOTAL_COMPUTE_RATIO, COMPUTE_GROWTH_RATE, FM_FIXED))
for largest_total_compute_ratio, compute_growth_rate, fm_fixed in param_combinations:
    print(f"Running with LARGEST_TOTAL_COMPUTE_RATIO={largest_total_compute_ratio}, COMPUTE_GROWTH_RATE={compute_growth_rate}, FM_FIXED={fm_fixed}")
    main(g=compute_growth_rate, ratio=largest_total_compute_ratio, fm=fm_fixed)


