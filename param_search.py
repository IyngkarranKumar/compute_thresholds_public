import importlib
import config
import main
importlib.reload(main)
importlib.reload(config)
from main import main
from config import Config
import itertools




if 1:
    Config.SAVE_RESULTS=True
    ALLOC=[30/70,50/50]
    G_GLOBAL_AI_COMPUTE_MEAN=[2.5,3.5]
    param_combinations = list(itertools.product(ALLOC, G_GLOBAL_AI_COMPUTE_MEAN))


    for alloc, g_global_AI_compute_mean in param_combinations:

        #set confing parameters
        Config.fixed_alloc=alloc #update config object
        Config.g_global_AI_compute_mean=g_global_AI_compute_mean #update config object

        main(Config) 