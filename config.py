import numpy as np

class Config:
    
    #workflow config
    PLOT_SCHEMATIC_SCATTER=False
    TRAINING_COMPUTE_PLOTS=False
    FIT_ALLOCATION_PLOTS=False
    GENERATED_SAMPLE_PLOTS=False
    SAVE_RESULTS,save_folder=False, 'results/test_save_folder'

    #sampling parameters
    n_simulations = 100 #for bootstrappng, sampling parameters etc. n_simulations = 10 #for bootstrappng, sampling parameters etc. 

    #training compute extrapolation config 
    AI2027_EXTRAP=True
    method_choice="method 2027" #['linear extrapolation', 'method 2027']
    hist_alloc=70/30
    hist_alloc_multiplier=1+(1/hist_alloc)
    FIXED_ALLOCATION=False
    fixed_alloc=70/30
    DYNAMIC_ALLOCATION=True #inference scaling continues improving
    assert(FIXED_ALLOCATION+DYNAMIC_ALLOCATION)==1
    pred_alloc_dict = {
            2024: 90/10,
            2025: 90/10,
            2026: 70/30,
            2027: 70/30,
            2028: 70/30,
        }
    g_historical=6.3 #from fit years
    g_global_AI_compute_mean=2.25
    g_AI_workload_share_mean=1.4 
    g_total_AI_2027 = g_global_AI_compute_mean*g_AI_workload_share_mean
    g_weights = [0.25,0.75] #historical, AI_2027
    g_total = g_weights[0]*g_historical + g_weights[1]*g_total_AI_2027
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
    LMS_SAMPLING="uniform"
    assert LMS_SAMPLING in ['gaussian', 'uniform']
    largest_model_share_mean,lms_stddev,min_lms,max_lms=0.3, 0.1,0.05,0.50
    SET_2024_LMS=False


    #min m sampling
    min_norm_m_min,min_norm_m_max = 1e-8, 1e-6 #wacky variable names

    #n_catg setting (higher the better, up to the point where delta M gets too small)
    n_catgs = 20


    #threshold counting PARAMETERS
    thresholds=[25, 26, 27, 28, 29]
    retrodict_thresholds=[23, 24, 25]
    threshold_widths = [0.5, 1, 1.5]  # List of threshold widths to analyze
    period_freq = '3M'  # frequency for doing frontier counts
    CI_percentiles=[10,50,90]