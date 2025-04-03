import importlib
import config
import main
importlib.reload(main)
importlib.reload(config)
from main import main
from config import Config  #load in config in file
import itertools



#param search
if 1:
    Config.SAVE_RESULTS=False
    Config.save_folder='results/search_alpha'

    LMS_BOUNDS = [(0.01,0.2),()]
    GROWTH_RATES = [4.0,5.0]
    

    param_combinations = list(itertools.product(LMS_BOUNDS,GROWTH_RATES))

    for param in param_combinations:
        print(param)
        Config.min_lms,Config.max_lms = param[0]
        Config.g_global_AI_compute_mean = param[1]
        Config.g_AI_workload_share_mean = 0
        Config.g_total = Config.g_global_AI_compute_mean + Config.g_AI_workload_share_mean #must set this 
                
        main(Config)

#scenario run
if 0: 
    Config.SAVE_RESULTS=False
    Config.save_folder='results/scenario_run'
    Config.PLOT_TRAINING_PLOTS=True
    
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
        'description':'Dynamic allocation following Epoch gate, setting 2024 LMS',
        'LMS_bounds':[0.1,0.5],
        'FIXED_ALLOCATION':False,
        'DYNAMIC_ALLOCATION':True,
        'pred_alloc_dict':{
            2024: 90/10,
            2025: 90/10,
            2026: 70/30,
            2027: 70/30,
            2028: 70/30,
        },
        'g_global_AI_compute_mean':2.25,
        'g_AI_workload_share_mean':1.4,
        'SET_2024_LMS':True,
    }

    scenario_4 = {
        'description':'Dynamic allocation following Epoch gate, not setting 2024 LMS',
        'LMS_bounds':[0.1,0.5],
        'FIXED_ALLOCATION':False,
        'DYNAMIC_ALLOCATION':True,
        'pred_alloc_dict':{
            2024: 90/10,
            2025: 90/10,
            2026: 70/30,
            2027: 70/30,
            2028: 70/30,
        },
        'g_global_AI_compute_mean':2.25,
        'g_AI_workload_share_mean':1.4,
        'SET_2024_LMS':False,
    }
    
    for scenario in [scenario_3]:
        print(scenario.get('description'))
        Config.n_simulations = 1000 #to get 'true' CIs
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
        