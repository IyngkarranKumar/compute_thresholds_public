import importlib
import config
import main
import scenarios_config
import itertools

importlib.reload(main)
importlib.reload(config)
importlib.reload(scenarios_config)

from main import main
from config import Config  #load in initialised config_class
from scenarios_config import *


if 1:

    for scenario in paper_scenarios:
        #reset to baseline config 
        importlib.reload(config); from config import Config
        Config.name = scenario.get('name', 'no name found'); print('\n\n',Config.name)
        scenario = {k:v for k,v in scenario.items() if k != 'name'}

        param_combinations = list(itertools.product(*scenario.values()))

        ###REMOVE FOR NON-GRADIENT SCENARIOS
        param_combinations = [list(combo)for combo in param_combinations if combo[1]==combo[2]] #for the allocation gradient



        print(f"Running {len(param_combinations)} combinations for {Config.name}:")
        for i, combo in enumerate(param_combinations, 1):
            param_str = ", ".join(f"{k}={v}" for k,v in zip(scenario.keys(), combo)) if combo else "baseline"
            print(f"{i}. {param_str}\n")
        
        
        for combination in param_combinations:
            if len(combination) == 0: #run baseline
                print('running baseline')
                main(Config)
            else: 
                for param_name,param_value in zip(scenario.keys(),combination):
                    setattr(Config,param_name,param_value) #set the config attributes
                print(f"Running {param_str}")
                main(Config)
        


    
#param search
if 0:
    Config.SAVE_RESULTS=True
    Config.save_folder='results/AI_2027_ALLOCATIONS'


    search_config=None
    
    for key in search_config.keys():
        if not hasattr(Config,key):
            raise ValueError(f'Parameter {key} not found in Config class')

    param_combinations = list(itertools.product(*search_config.values()))

    for combination in param_combinations:
        print(combination)
        # Set config attributes for this parameter combination
        for param_name,param_value in zip(search_config.keys(),combination):
            setattr(Config,param_name,param_value) #set the config attributes

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
        