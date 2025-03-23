import importlib
import main
importlib.reload(main)
from main import main
import itertools


TEST=False

if TEST: 
    ALLOC=[30/70]
    G_GLOBAL_AI_COMPUTE_MEAN=[2.0]


if not TEST:
    ALLOC=[10/90,30/70,50/50]
    G_GLOBAL_AI_COMPUTE_MEAN=[2.5,3.5,4.5,5.5]


param_combinations = list(itertools.product(ALLOC, G_GLOBAL_AI_COMPUTE_MEAN))
for alloc, g_global_AI_compute_mean in param_combinations:
    print(f"Running with fixed_alloc={alloc}, g_global_AI_compute_mean={g_global_AI_compute_mean}")
    main(_alloc_=alloc, _g_=g_global_AI_compute_mean)


