import numpy as np

class main_config:
    
    name='baseline'
    PLOT_SCHEMATIC_SCATTER=False
    TRAINING_COMPUTE_PLOTS=False
    FIT_ALLOCATION_PLOTS=False
    GENERATED_SAMPLE_PLOTS=False
    COMPUTE_FRONTIER_COUNTS=True
    SAVE_RESULTS,save_folder = True, 'results_2'
    WANDB_LOGGING, wandb_project=False, 'test-space'
    SET_2024_COUNTS=True

    #KEY PARAMETERS
    n_simulations = 1000 #for generating CIs
    g_weights = [0.25,0.75] #weighting of growth weights
    pred_alloc_dict = { #training-inference allocations
        2024: 40/60,
        2025: 40/60,
        2026: 40/60,
        2027: 30/70,
        2028: 30/70,
    }
    LMS_SAMPLING="log_normal" #largest model share sampling distribution['log_normal', 'uniform']
    min_lms,max_lms=0.05,0.50 #largest model share absolute bounds




    #training compute extrapolation config 
    AI2027_EXTRAP=True
    method_choice="method 2027"
    hist_alloc=40/60
    hist_alloc_multiplier=1+(1/hist_alloc)
    FIXED_ALLOCATION,DYNAMIC_ALLOCATION=False,True
    assert(FIXED_ALLOCATION+DYNAMIC_ALLOCATION)==1
    fixed_alloc=60/40

    g_historical=6.3 #from fit years
    g_global_AI_compute_mean=2.25
    g_AI_workload_share_mean=1.4 
    g_stdev=0.5


    #allocation fit parameters
    fit_years=np.arange(2017,2024)
    pred_years = np.arange(2024,2029)
    constraint_point=(1,1)
    filter_thresholds=1e-20 #ignore models smaller than this

    ##SAMPLING PARAMETERS
    ALLOC_FIT_TYPE='cumulative' #[cumulative, categorical]
    POINT_CUM_ALLOC_PARAMS=False #takes mean of historical datas
    DISTRIBUTION_CUM_ALLOC_PARAMS=True
    grad_cum_alloc_min, grad_cum_alloc_max = 0.9,1.1
    assert(POINT_CUM_ALLOC_PARAMS+DISTRIBUTION_CUM_ALLOC_PARAMS)==1, "Only one of DEFAULT_CUM_ALLOC_PARAMS or CUSTOM_CUM_ALLOC_PARAMS can be True"

    #IMPORTANT PARAMETER - largest model share
    assert LMS_SAMPLING in ['log_normal', 'uniform']
    SET_2024_LMS=True

    #min m sampling
    min_norm_m_min,min_norm_m_max = 1e-8, 1e-6 #wacky variable names


    n_catgs = 20


    #threshold counting PARAMETERS
    thresholds=[25, 26, 27, 28, 29]
    retrodict_thresholds=[23, 24, 25]
    threshold_widths = [0.5, 1, 1.5]  # List of threshold widths to analyze
    period_freq = '3M'  # frequency for doing frontier counts
    CI_percentiles=[5,50,95]

Config=main_config()