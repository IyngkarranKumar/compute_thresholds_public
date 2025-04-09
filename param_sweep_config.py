import config, importlib, numpy as np
importlib.reload(config)
from config import Config

baseline_scenario_config={
    "name":"baseline"
}

standard_lms_2024_sampling = {
    "name":"standard_lms_2024_sampling",
    "SET_2024_LMS":[False],
}

allocations_search_config={
    "name":"allocations_search",
    "DYNAMIC_ALLOCATION":[True],
    "FIXED_ALLOCATION":[False],
    "COMPUTE_FRONTIER_COUNTS":[False],
    "pred_alloc_dict":[{
            2024: 40/60,
            2025: 40/60, 
            2026: 40/60,
            2027: 30/70,
            2028: 20/80,
        },
        {
            2024: 90/10,
            2025: 90/10, 
            2026: 70/30,
            2027: 70/30,
            2028: 70/30,
        },]
    }

growth_weightings_config={
    "name":"growth_weightings",
    "COMPUTE_FRONTIER_COUNTS":[False],
    "g_weights":[(0.1,0.9), (1/3,2/3), (0.5,0.5)]
}

allocation_gradient_config={
    "name":"allocation_gradient",
    "COMPUTE_FRONTIER_COUNTS":[False],
    "grad_cum_alloc_min":[0.5],
    "grad_cum_alloc_max":[1.0]
}


paper_scenarios = [baseline_scenario_config, standard_lms_2024_sampling, allocations_search_config, growth_weightings_config, allocation_gradient_config]
#paper_scenarios= [baseline_scenario_config]
paper_scenarios = [standard_lms_2024_sampling]

for scenario in paper_scenarios:
    assert np.all(list((hasattr(Config,key) for key in scenario.keys()))), f"Scenario {scenario} has a key that is not in Config"

