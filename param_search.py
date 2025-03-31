import importlib
import config
import main
importlib.reload(main)
importlib.reload(config)
from main import main
from config import Config  #load in config in file
import itertools




if 1:
    Config.SAVE_RESULTS=True
    Config.save_folder='results/alloc_search'
    ALLOC_=[30/70,40/60,50/50,60/40,70/30]
    param_combinations = list(itertools.product(ALLOC_))

    for alloc in param_combinations:

        #set confing parameters
        print('alloc:',alloc)
        Config.FIXED_ALLOCATION, Config.DYNAMIC_ALLOCATION = True, False
        Config.fixed_alloc=alloc[0]
        
        main(Config)