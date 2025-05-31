import config, importlib, numpy as np
importlib.reload(config)
from config import Config

#baseline scenario
baseline_scenario_config={
    "name":"baseline",
    "SAVE_RESULTS":[False],
    "save_folder":["results/paper_scenarios"],
    "n_simulations":[100],
    "SET_2024_COUNTS":[True],
    "COMPUTE_FRONTIER_COUNTS":[True],
    "LMS_SAMPLING":["log_normal"],
    "min_lms":[0.05],
    "max_lms":[0.50],
    "SET_2024_LMS":[True],
    "DYNAMIC_ALLOCATION":[True],
    "FIXED_ALLOCATION":[False],
    "grad_cum_alloc_min":[0.9],
    "grad_cum_alloc_max":[1.1]
}

#2024 without setting 2024 largest model to gpt-4o
uniform_lms_sampling_projections = {
    "name":"uniform_lms_projections",
    "LMS_SAMPLING":["uniform"],
}

#GATE allocations
allocations_search_config={
    "name":"allocations_search",
    "DYNAMIC_ALLOCATION":[True],
    "FIXED_ALLOCATION":[False],
    "COMPUTE_FRONTIER_COUNTS":[False],
    "pred_alloc_dict":[{
            2024: 90/10,
            2025: 90/10, 
            2026: 70/30,
            2027: 70/30,
            2028: 70/30,
        }]
    }

#different growth weightings
growth_weightings_config={
    "name":"growth_weightings",
    "COMPUTE_FRONTIER_COUNTS":[False],
    "g_weights":[(0.1,0.9), (1/3,2/3), (0.5,0.5)]
}

#lms smaller
alternate_lms_config={
    "name":"alternate_lms",
    "COMPUTE_FRONTIER_COUNTS":[True],
    "LMS_SAMPLING":["uniform"],
}

#allocation gradient a
allocation_gradient_config_a={
    "name":"allocation_gradient_a",
    "COMPUTE_FRONTIER_COUNTS":[False],
    "grad_cum_alloc_min":[0.7],
    "grad_cum_alloc_max":[0.9]
}

#allocation gradient b
allocation_gradient_config_b={
    "name":"allocation_gradient_b",
    "COMPUTE_FRONTIER_COUNTS":[False],
    "grad_cum_alloc_min":[0.5],
    "grad_cum_alloc_max":[0.7]
}



paper_scenarios = [baseline_scenario_config, 
                   uniform_lms_sampling_projections, 
                   allocations_search_config, 
                   growth_weightings_config, 
                   alternate_lms_config,
                   allocation_gradient_config_a,
                   allocation_gradient_config_b]


for scenario in paper_scenarios:
    assert np.all(list((hasattr(Config,key) for key in scenario.keys()))), f"Scenario {scenario} has a key that is not in Config"

