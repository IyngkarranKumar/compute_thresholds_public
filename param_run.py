import importlib
import config
import main
importlib.reload(main)
importlib.reload(config)
from main import main
from config import Config  #load in config in file
import itertools



#param search
if 0:
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

#scenario run
if 1: 
    Config.SAVE_RESULTS=False
    Config.save_folder='results/scenario_run'
    
    scenario_1 = {
        'description':'Fixed allocation 70/30, not setting 2024 LMS',
        'LMS_bounds':[0.1,0.5],
        'FIXED_ALLOCATION':True,
        'DYNAMIC_ALLOCATION':False,
        'fixed_alloc':70/30,
        'g_global_AI_compute_mean':2.25,
        'g_AI_workload_share_mean':1.4,
        'SET_2024_LMS':False,
    }

    scenario_2 = {
        'description':'Fixed allocation 70/30, setting 2024 LMS',
        'LMS_bounds':[0.1,0.5],
        'FIXED_ALLOCATION':True,
        'DYNAMIC_ALLOCATION':False,
        'fixed_alloc':70/30,
        'g_global_AI_compute_mean':2.25,
        'g_AI_workload_share_mean':1.4,
        'SET_2024_LMS':True,
    }

    scenario_3 = {
        'description':'Dynamic allocation following AI-2027, setting 2024 LMS',
        'LMS_bounds':[0.1,0.5],
        'FIXED_ALLOCATION':False,
        'DYNAMIC_ALLOCATION':True,
        'pred_alloc_dict':{
            2024: 40/60,
            2025: 30/70,
            2026: 30/70,
            2027: 30/70,
            2028: 20/80,
        },
        'g_global_AI_compute_mean':2.25,
        'g_AI_workload_share_mean':1.4,
        'SET_2024_LMS':True,
    }
    
    for scenario in [scenario_1, scenario_2]:
        print(scenario.get('description'))
        Config.min_lms,Config.max_lms = scenario.get('LMS_bounds', None)
        Config.SET_2024_LMS = scenario.get('SET_2024_LMS', None)
        Config.FIXED_ALLOCATION = scenario.get('FIXED_ALLOCATION', None)
        Config.DYNAMIC_ALLOCATION = scenario.get('DYNAMIC_ALLOCATION', None)
        Config.fixed_alloc = scenario.get('fixed_alloc', None)
        Config.pred_alloc_dict = scenario.get('pred_alloc_dict', None)
        Config.g_global_AI_compute_mean = scenario.get('g_global_AI_compute_mean', None)
        Config.g_AI_workload_share_mean = scenario.get('g_AI_workload_share_mean', None)
        Config.g_total = Config.g_global_AI_compute_mean + Config.g_AI_workload_share_mean
        
        
        main(Config)
        