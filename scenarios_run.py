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



for scenario in paper_scenarios:
    #reset to baseline config 
    importlib.reload(config); from config import Config
    Config.name = scenario.get('name', 'no name found'); print('\n\n',Config.name)
    scenario = {k:v for k,v in scenario.items() if k != 'name'}

    param_combinations = list(itertools.product(*scenario.values()))



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
    

