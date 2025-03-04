### File for running large number of scenarios


if 1: #===CONFIG===


    #training compute extrapolation
    LINEAR_EXTRAP=True
    AI2027_EXTRAP=True
    method_choice="linear extrapolation" #['linear extrapolation', 'method 2027']
    g_global_AI_compute=2.25
    g_AI_workload_share=2.0 #assuming AI_compute_usage/AI_compute_capacity = const - 3.0 gets the two superposed!


    #allocations
    FIXED_ALLOCATION=False
    fixed_tau=0.8 #all copute to training 
    DECREASING_TAU=True #inference scaling continues improving
    assert(FIXED_ALLOCATION+DECREASING_TAU)==1
    tau_dict = {
        2024: 1.0,
        2025: 0.9,
        2026: 0.8,
        2027: 0.7,
        2028: 0.6,
    }

    #generate samples
    CONST_FM=True
    LIN_EXTRAP_FM=False
    CUSTOM_FM=False
    if CUSTOM_FM:
        fm_grad_dict={
            2024:1.1,
            2025:1.1,
            2026:1.1,
            2027:1.1,
            2028:1.1,
            2029:1.1
        }
        fm_int_dict={
            2024:0.9,
            2025:0.8,
            2026:0.7,
            2027:0.6,
            2028:0.5,
            2029:0.4
        }

    #individual model size parameters
    log_min_norm_m = np.log10(1e-8) #the smallest model to allocate compute to is ~1e-8 the size of total compute spending that year
    log_max_norm_m = np.log10(1e-1) #free param - assume that largest model that year is no larger than 10% of total training compute (can find this from historic data and so sensitivity analysis)


    #bin sampling parameters
    bin_sampling_method='random'
    k=-100 #for exponential dist sampling


    thresholds=[25, 26, 27, 28, 29, 30]

    #frontier-connected threshold counts for samples
    threshold_widths = [0.5, 1, 1.5]  # List of threshold widths to analyze
    period_freq = '6M'  # Can be changed to any frequency like '1Y', '3M', '30D'

    retrodict_thresholds=[23, 24, 25]

    PLOT_SCHEMATIC_SCATTER=False
    TOTAL_COMPUTE_PLOT=False
    PLOT_SAMPLE_KDES=False
    PLOT_SAMPLE_SCATTERS=False



if 1: #===IMPORTS===
        
    import logging, time 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    start_time = time.time()

    try:
        logging.info("Starting imports...")
        import numpy as np
        from scipy import stats, optimize
        from sklearn.linear_model import LinearRegression
        from IPython.display import display
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import itertools
        import copy,re, pdb, warnings
        
        end_time = time.time()
        logging.info(f"Imports completed in {end_time - start_time:.2f} seconds")
        
        
    except Exception as e:
        logging.error(f"Error during imports: {str(e)}")

    np.random.seed(42)
    warnings.filterwarnings("ignore")

if 1: #UTILS

    def norm_exp_func(x,a,b,k):
        norm_factor=(1/k)*(np.exp(k*b)-np.exp(k*a))
        return (1/norm_factor)*np.exp(k*x)

    def sample_from_exp_dist(a,b,k,spacing='linear'):
        x=np.linspace(a,b,10000) #might need to change this to logspace
        dx=x[1]-x[0] #differnt if logspace
        pdf=norm_exp_func(x,a,b,k=k)
        assert(round(sum(pdf*dx),2)==1) #sanity check on probability dist
        prob_dist=pdf*dx
        prob_dist=prob_dist/np.sum(prob_dist) #ensure that sums exactly to 1 for use with np.random.choice

        return np.random.choice(x,p=prob_dist)

    def decimal_year_to_date(decimal_year):
        if isinstance(decimal_year, pd.Series):
            return decimal_year.apply(lambda x: decimal_year_to_date(x))
        if isinstance(decimal_year, (list, np.ndarray)):
            return [decimal_year_to_date(x) for x in decimal_year]
        year = int(decimal_year)
        remainder = decimal_year-year
        days_in_year = 366 if pd.Timestamp(year,1,1).is_leap_year else 365
        days = int(remainder*days_in_year)
        return pd.Timestamp(year,1,1)+pd.Timedelta(days=days)


    def compute_allocations(tau):
        tau = np.array(tau)
        train_alloc = tau/(tau+1)
        inference_alloc = 1/(tau+1)
        return train_alloc, inference_alloc


if 1: #===DATA LOADING===
    #Feb 2025 dataset

    #path 
    path="/Users/iyngkarrankumar/Documents/GovAI WF/EUAIA_thresholds_project/data/notable_ai_models_24_02_2025.csv"

    df = pd.read_csv(path)
    df = df[~df["Notability criteria"].isna()]

    df["compute"] = df["Training compute (FLOP)"]
    df["date"] = pd.to_datetime(df["Publication date"])
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["model"] = df["Model"]
    df["cost"] = df["Training compute cost (2023 USD)"]
    df["cost"] = df["cost"].fillna("$0")  # Handle NaN values
    df["cost"] = df["cost"].astype(str)  # Convert to string
    df["cost"] = df["cost"].str.replace(",", "").str.replace("$", "").astype(float)
    df = df[["model", "compute", "date", "cost","year"]]

    # Models to remove
    to_remove = ["AlphaGo Zero", "AlphaZero"]
    df = df[~df["model"].isin(to_remove)]



    # Print stats for full dataset
    print("=== Full Dataset ===")
    print("Most recent date:", df["date"].max())
    print("\nDatapoints per year:")
    for year in range(2017, 2025):
        count = len(df[df["year"] == year])
        print(f"{year}: {count}")

    max_compute_idx = df['compute'].idxmax()
    print(f"\nLargest compute value: {df.loc[max_compute_idx, 'compute']:.2e} ({df.loc[max_compute_idx, 'model']})")
    # Create dataset without specified years
    years_to_exclude = [2025, 2024]  # List of years to exclude
    df_filtered = df[~df["year"].isin(years_to_exclude)].copy()

    print(f"\n=== Dataset excluding years {years_to_exclude} ===")
    print("Most recent date:", df_filtered["date"].max())
    print("\nDatapoints per year:")
    for year in range(2017, 2024):
        count = len(df_filtered[df_filtered["year"] == year])
        print(f"{year}: {count}")

    max_compute_idx = df_filtered['compute'].idxmax()
    print(f"\nLargest compute value: {df_filtered.loc[max_compute_idx, 'compute']:.2e} ({df_filtered.loc[max_compute_idx, 'model']})")

    df = df_filtered

    # Report number of entries before removing NaN
    print(f"\n\n Number of entries before removing rows with compute=NaN: {len(df)}")

    # Remove rows with NaN in compute column
    df = df.dropna(subset=['compute'])

    # Report number of entries after removing rows with compute=NaN
    print(f"Number of entries after removing rows with compute=NaN: {len(df)}")

    
    if PLOT_SCHEMATIC_SCATTER: #generate basic scatterplot
        fig = sns.scatterplot(data=df[df['date']>'2010-01-01'], x='date',y='compute')
        fig.set(yscale='log')
        plt.grid(alpha=0.5)

        # Add line of best fit for historical data
        historical_data = df[df['date']>'2010-01-01']
        x = historical_data['date'].astype(np.int64) // 10**9  # Convert to unix timestamp
        y = historical_data['compute']
        z = np.polyfit(x, np.log(y), 1)
        p = np.poly1d(z)
        plt.plot(historical_data['date'], np.exp(p(x)), 'b--', alpha=0.8)

        future_dates = pd.date_range(start=f'{df.year.max()+1}-01-01', end='2029-12-31', periods=200)
        base = 1e23  # Starting point based on 2024 level
        noise = np.random.normal(0, 10, len(future_dates))
        years_from_2025 = (future_dates.year - (df.year.max()+1))

        growth_rate = 3.0  # Exponential growth rate
        future_compute = base * np.exp(growth_rate * years_from_2025) * (1 + noise)
        plt.scatter(future_dates, future_compute, alpha=0.3, color='red', label='Scenario A')

        growth_rate = 0.4
        future_compute = base * np.exp(growth_rate * years_from_2025) * (1 + noise)
        plt.scatter(future_dates, future_compute, alpha=0.3, color='green', label='Scenario B')

        growth_rate = 5.0  # Higher growth rate than Scenario A
        future_compute = base * np.exp(growth_rate * years_from_2025) * (1 + noise)
        plt.scatter(future_dates, future_compute, alpha=0.3, color='blue', label='Scenario C')

        plt.legend()
        plt.xlim([pd.Timestamp('2020-01-01'),pd.Timestamp('2030-01-01')])

        for exp in range(25,31):
            plt.axhline(y=10**exp,color='gray',linestyle='--',alpha=0.6)

if 1: #Training compute extrapolation
    #total AI relevant compute extrapolations

    assert method_choice in ['linear extrapolation','method 2027']

    #plot
    PLOT=True

    LOG_AGGREGATE_COMPUTE_DATA={}


    year_grouped_df=df.groupby(df['date'][df['date']>'2010-01-01'].dt.year)
    aggregate_compute=year_grouped_df['compute'].sum()
    log_aggregate_compute=np.log10(aggregate_compute)

    recent_years = log_aggregate_compute[log_aggregate_compute.index.isin(range(2020,df.year.max()+1))]
    recent_log_compute_dict = {int(k): v for k, v in recent_years.items()}


    if 1: #do historical data
        LOG_AGGREGATE_COMPUTE_DATA['historical data'] = {int(k): v for k, v in log_aggregate_compute.items()}


    if AI2027_EXTRAP:
        training_usage_2023 = 10**log_aggregate_compute.get(2023)
        total_usage_2023 = 2 * training_usage_2023
        
        AI_compute_usage={}
        for idx,year in enumerate(range(2024, 2029)):
            AI_compute_usage[year] = total_usage_2023*(g_global_AI_compute+g_AI_workload_share)**(idx+1)
        
        log_aggregate_compute_predictions_dict = {year: np.log10(compute) for year, compute in AI_compute_usage.items()}
        LOG_AGGREGATE_COMPUTE_DATA['Total-method 2027'] = log_aggregate_compute_predictions_dict
        
        '''
        n_H100es_2023=4e6 #from compute forecast - 4million H100 equivalents
        h100_util_rate=0.3 
        h100_fp16_peak_flop_s=1e15
        h100_fp16_flop_year=h100_fp16_peak_flop_s * (60*60*24*365) * h100_util_rate
        global_compute_2023=n_H100es_2023 * h100_fp16_flop_year


        ftm_share_ai_workloads=2*(2.5*10**(-4)) #need to multiply by two assuming equal allocation 
        '''


    if LINEAR_EXTRAP:
        # Fit exponential for extrapolation
        # Linear regression
        x = np.array(list(year_grouped_df.groups.keys())).reshape(-1, 1)
        y = log_aggregate_compute.values
        reg = LinearRegression().fit(x, y)

        # Generate future years for extrapolation
        pred_years = np.arange(df.year.max()+1, 2029)
        # Get predictions
        log_aggregate_compute_predictions = reg.predict(pred_years.reshape(-1, 1))
        log_aggregate_compute_predictions_dict = {int(year): pred for year, pred in zip(pred_years.flatten(), log_aggregate_compute_predictions)}

        # Combine historical and predicted data
        #combined_log_aggregate_compute_dict = dict(sorted({**recent_log_compute_dict, **log_aggregate_compute_predictions_dict}.items()))

        LOG_AGGREGATE_COMPUTE_DATA['Total-linear extrapolation']=log_aggregate_compute_predictions_dict


    #do allocations
    if 1: 
        if FIXED_ALLOCATION:
            train_alloc,inference_alloc=compute_allocations(tau=fixed_tau)
            LOG_AGGREGATE_COMPUTE_DATA['aggregate training compute'] = {year: val + np.log(train_alloc) for year, val in LOG_AGGREGATE_COMPUTE_DATA[f"Total-{method_choice}"].items()}
            LOG_AGGREGATE_COMPUTE_DATA['aggregate inference compute'] = {year: val + np.log(inference_alloc) for year, val in LOG_AGGREGATE_COMPUTE_DATA[f"Total-{method_choice}"].items()}
        
        if DECREASING_TAU:
            train_alloc_dict = {}
            inference_alloc_dict = {}
            
            for year, val in LOG_AGGREGATE_COMPUTE_DATA[f'Total-{method_choice}'].items():
                tau = tau_dict.get(year, 1.0) #gets key; if key not found, default to 1
                train_alloc, inference_alloc = compute_allocations(tau=tau)
                train_alloc_dict[year] = val + np.log10(train_alloc)
                inference_alloc_dict[year] = val + np.log10(inference_alloc)
                
            LOG_AGGREGATE_COMPUTE_DATA['aggregate training compute'] = train_alloc_dict
            LOG_AGGREGATE_COMPUTE_DATA['aggregate inference compute'] = inference_alloc_dict


    if TOTAL_COMPUTE_PLOT:
        plt.figure(figsize=(10,6))
        

        # Plot extrapolations for each method
        colors = {
            'historical data': 'blue',
            'Total-linear extrapolation': 'orange',
            'Total-method 2027': 'purple', 
            'aggregate training compute': 'green',
            'aggregate inference compute': 'red'
        }
        markers = {
            'historical data': 'o',
            'Total-linear extrapolation': 'o',
            'Total-method 2027': 's',
            'aggregate training compute': '.',
            'aggregate inference compute': 'x'
        }
        for method, predictions in LOG_AGGREGATE_COMPUTE_DATA.items():
            years = [y for y in predictions.keys()]
            values = [predictions[y] for y in years]
            plt.scatter(years, values, label=f'{method} (Projected)' if method!='historical data' else f'{method}',
                    color=colors[method], marker=markers[method])
        
        plt.xlabel('Year')
        plt.ylabel('Log10(Compute) [FLOP]')
        plt.title(f'Compute Usage Over Time')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(min(log_aggregate_compute.index), 2030, 2))


        # Plot compute allocations for different tau values
        plt.figure(figsize=(10,6))

        years = sorted(tau_dict.keys())
        tau_values = [tau_dict[y] for y in years]

        train_allocs = []
        inference_allocs = []
        if FIXED_ALLOCATION:
            train_allocs,inference_allocs = compute_allocations(tau=fixed_tau*np.ones(len(pred_years)))
        if DECREASING_TAU:
            train_allocs, inference_allocs = compute_allocations(tau=np.array(list(tau_dict.values())))


        plt.plot(years, train_allocs, 'g-', label='Training Allocation')
        plt.plot(years, inference_allocs, 'r-', label='Inference Allocation')
        plt.scatter(years, train_allocs, color='green', marker='o')
        plt.scatter(years, inference_allocs, color='red', marker='o')
        plt.ylim(0,1)

        plt.xlabel('Year')
        plt.ylabel('Allocation Fraction') 
        plt.title('Compute Allocations Over Time')
        plt.legend()
        plt.grid(True)
        plt.xticks(years)
         
if 1: #Generate compute samples
    #get compute_alloc fits
    fit_years=np.arange(2020,df.year.max()+1)
    FIT_DATA={year:None for year in fit_years}

    logging.info('Fitting f_M coefficients')
    for idx,year in enumerate(fit_years):
        total_compute=aggregate_compute[aggregate_compute.index==year].values
        datapoints_year=df[df['date'].dt.year==year]['compute']
        mean_log_compute=np.log10(datapoints_year).mean()

        sorted_computes=np.sort(datapoints_year)
        norm_factor=total_compute[0]
        norm_sorted_computes=sorted_computes/norm_factor
        cumsum=np.cumsum(sorted_computes)
        norm_cumsum=cumsum/norm_factor

        #store data 
        FIT_DATA[year]={
        'compute':sorted_computes,
        'cumulative_sum':cumsum,
        'norm_factor':norm_factor,
        'f_m_coeffs':None,
                }
        
        #fit data
        X = np.log10(norm_sorted_computes).reshape(-1, 1)
        y = np.log10(norm_cumsum)
        reg = LinearRegression().fit(X, y)
        FIT_DATA[year]['fit data'] = (X.ravel(),y.ravel())
        FIT_DATA[year]['f_m_coeffs'] = [reg.coef_[0], reg.intercept_]



    ##generate compute samples
    ##compute allocation parameters


    default_fm_grad,default_fm_int=np.mean([FIT_DATA[year]['f_m_coeffs'][0] for year in FIT_DATA]),np.mean([FIT_DATA[year]['f_m_coeffs'][1] for year in FIT_DATA])
    assert(CONST_FM+LIN_EXTRAP_FM+CUSTOM_FM)==1, "Only one of CONST_FM, LIN_EXTRAP_FM, or CUSTOM_FM can be True"


    #compute allocation parameters
    if CONST_FM:
        fm_grad,fm_int = np.mean([FIT_DATA[year]['f_m_coeffs'][0] for year in FIT_DATA]),np.mean([FIT_DATA[year]['f_m_coeffs'][1] for year in FIT_DATA])
    if LIN_EXTRAP_FM:
        pass


    all_years=np.concatenate([fit_years, pred_years.astype(int).ravel()])

    COMPUTE_SAMPLE_DATA={int(year):{} for year in all_years} #all years because we're also retrodicting

    for year in all_years:

        if year in fit_years:
            log_agg_training_compute=LOG_AGGREGATE_COMPUTE_DATA["historical data"][year]
        if year in pred_years:
            log_agg_training_compute=LOG_AGGREGATE_COMPUTE_DATA[f"Total-{method_choice}"][year]
            
        agg_training_compute=10**log_agg_training_compute #total compute used over the year

        #model sizes (as fraction of T_tot)
        norm_ms = np.logspace(log_min_norm_m,log_max_norm_m,2*(int(log_max_norm_m)-int(log_min_norm_m))+1)
        log_norm_ms = np.log10(norm_ms)

        if CONST_FM: 
            fm_grad,fm_int=default_fm_grad,default_fm_int
        elif LIN_EXTRAP_FM:
            raise NotImplementedError("Linear extrapolation of fm_grad and fm_int not implemented")
        elif CUSTOM_FM:
            fm_grad,fm_int=fm_grad_dict.get(year,1.1),fm_int_dict.get(year,0.92)
        if year in FIT_DATA.keys():
            fm_grad,fm_int=FIT_DATA[year]['f_m_coeffs']

        log_frac_cum_compute = fm_grad*log_norm_ms + fm_int
        cum_fm=10**log_frac_cum_compute

        model_ctgs = [f'{norm_ms[i]:.2e}--{norm_ms[i+1]:.2e}' for i in range(len(norm_ms)-1)]
        f_m = np.diff(cum_fm) #we don't include compute alloc to models 1e-8 smaller than total compute
        bin_compute_allocs=f_m*agg_training_compute #array of how much compute allocated to each bin
        DATA_alloc={model_ctgs[i]:
                    {'compute alloc':bin_compute_allocs[i]} for i in range(len(model_ctgs))}
        
        compute_samples_rand=[]

        for idx,(ctg,alloc) in enumerate(list(zip(model_ctgs,bin_compute_allocs))):
            #here alloc is the amount of alloc given to each individual bin

            bounds = ctg.split('--')
            norm_model_bin_lb,norm_model_bin_ub = float(bounds[0]),float(bounds[1])
            model_bin_lb,model_bin_ub = agg_training_compute*norm_model_bin_lb, agg_training_compute*norm_model_bin_ub #normalising factor is total training compute
            allocnorm_model_bin_lb,allocnorm_model_bin_ub=model_bin_lb/alloc, model_bin_ub/alloc

            #not generating multiple samples yet for CIs
            running_tot=0
            allocnormed_samples=[] 
            while running_tot<1:
                #SAMPLE
                if bin_sampling_method=='random':
                    sample = np.random.uniform(allocnorm_model_bin_lb, allocnorm_model_bin_ub)
                elif bin_sampling_method=='exp':
                    sample  = sample_from_exp_dist(a=allocnorm_model_bin_lb,b=allocnorm_model_bin_ub,k=k)

                #SUM CHECK
                if running_tot + sample > 1:
                    allocnormed_samples.append(1 - running_tot)
                    running_tot = 1
                else:
                    allocnormed_samples.append(sample)
                    running_tot += sample

            #print(f"Model category {ctg} adds {len(allocnormed_samples)} models")
            compute_samples_rand = compute_samples_rand + (list(alloc*np.array(allocnormed_samples)))


        compute_samples_rand = [x for x in compute_samples_rand if x!=0]

        COMPUTE_SAMPLE_DATA[year]['samples']=compute_samples_rand
        COMPUTE_SAMPLE_DATA[year]['date']=[decimal_year_to_date(year+np.random.random()) for _ in compute_samples_rand] #conver to stand pd datetime format



    logging.info("Number of samples per year:")
    for year in pred_years.ravel():
        print(f"{year}: {len(COMPUTE_SAMPLE_DATA[year]['samples'])} samples")

            

    if PLOT_SAMPLE_KDES: 
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        axes = axes.ravel()

        for idx, (year, value) in enumerate((y, s) for y, s in COMPUTE_SAMPLE_DATA.items() if y in pred_years):
            sns.kdeplot(data=np.log10(value['samples']), ax=axes[idx])
            axes[idx].set_title(f'Year {year}')
            axes[idx].set_xlabel('log compute (FLOPs)')
            axes[idx].set_ylabel('Density')
            axes[idx].grid(alpha=0.5)
            axes[idx].set_xlim([15,30])

        plt.tight_layout()
        plt.show()

    if PLOT_SAMPLE_SCATTERS:
        
        # Create scatter plot
        plt.figure(figsize=(12,6))
        plt.scatter(df[df['year'].isin(fit_years)]['date'], np.log10(df[df['year'].isin(fit_years)]['compute']), alpha=0.5, label='Historical')
        for year in pred_years:
            plt.scatter(COMPUTE_SAMPLE_DATA[year]['date'], np.log10(COMPUTE_SAMPLE_DATA[year]['samples']), alpha=0.5, label='Projected' if year==pred_years[0] else "", color='red')
        plt.xlabel('Year')
        plt.ylabel('Log Compute (FLOPs)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

if 1: # threshold counting

    ## regular counts 

    threshold_counts = {year: [] for year in pred_years.astype(int).ravel()}

    for year, samples in COMPUTE_SAMPLE_DATA.items():
        if year in pred_years:
            for threshold in thresholds:
                count = sum(x >= 10**threshold for x in samples['samples'])
                threshold_counts[year].append(count)

    df_counts = pd.DataFrame(threshold_counts,
                            index=[f'>1e{t}' for t in thresholds])


    # Make cumulative across years
    df_counts_cumulative = df_counts.copy()
    for idx in df_counts.index:
        df_counts_cumulative.loc[idx] = df_counts.loc[idx].cumsum()
    display(df_counts_cumulative)



    

    #frontier counts

    bins = pd.date_range(start=f"{pred_years.ravel().min()}-01-01", end=f"{pred_years.ravel().max()+1}-01-01", freq=period_freq).astype(f'period[{period_freq}]')
    period_data=pd.Series(bins[bins.searchsorted(df.date.dt.to_period(period_freq)) - 1], index=df.index)

    # Initialize results dictionary
    frontier_counts = {width: {period: 0 for period in bins} for width in threshold_widths}

    # For each period
    for period in bins:
        period_samples = []
        
        # Collect all samples from that period
        for year in COMPUTE_SAMPLE_DATA:
            #print(COMPUTE_SAMPLE_DATA[year]['date'])
            pd_dt_dates = pd.to_datetime(COMPUTE_SAMPLE_DATA[year]['date'])
            if period.year == year:
                # Filter samples that fall within the period
                period_start = period.start_time
                period_end = period.end_time
                period_mask = (pd_dt_dates >= period_start) & (pd_dt_dates < period_end)
                period_samples.extend(np.array(COMPUTE_SAMPLE_DATA[year]['samples'])[period_mask])
                
        if period_samples:
            # Find largest model in period
            frontier = max(period_samples)
            
            # Count models within each threshold width
            for width in threshold_widths:
                threshold = 10**width
                count = sum(abs(np.log10(model) - np.log10(frontier)) <= width for model in period_samples)
                frontier_counts[width][period] = count

    # Convert to DataFrame
    df_frontier = pd.DataFrame(frontier_counts)
    df_frontier.columns = [f'Within {w} OOM' for w in threshold_widths]

    # Sum up counts for each year
    yearly_counts = {}
    for width in threshold_widths:
        col = f'Within {width} OOM'
        yearly_counts[col] = df_frontier.groupby(df_frontier.index.year)[col].sum()


    df_frontier_yearly = pd.DataFrame(yearly_counts)
    df_frontier_yearly = df_frontier_yearly.transpose()


    display(df_frontier_yearly)

if 1: #backtesting 
    #backtesting the absolute thresholds

    retrodict_years=fit_years

    #observed
    # Create DataFrame from observed counts
    df_observed = pd.DataFrame.from_dict({threshold: {year: sum(df[df['year'] == year]['compute'] > 10**threshold)
                                                    for year in retrodict_years}
                                        for threshold in retrodict_thresholds}, 
                                        orient='index')
    df_observed.index = [f'>1e{t}' for t in retrodict_thresholds]
    df_observed.index.name = 'Threshold'

    # Create retrodict counts dictionary
    retrodict_counts = {year: [] for year in retrodict_years}

    for year, data in COMPUTE_SAMPLE_DATA.items():
        samples = data['samples']
        if year in retrodict_years:
            for threshold in retrodict_thresholds:
                count = sum(x >= 10**threshold for x in samples)
                retrodict_counts[year].append(count)

    df_retrodict = pd.DataFrame(retrodict_counts,
                            index=[f'{t:.2e}' for t in retrodict_thresholds])
    df_retrodict.index.name = 'Threshold'

    # Take cumulative sum across years for both dataframes
    df_observed_cumulative = df_observed.cumsum(axis=1)
    df_retrodict_cumulative = df_retrodict.cumsum(axis=1)


    # Create dataframe with observed and retrodicted values
    combined_df = pd.DataFrame(index=df_observed_cumulative.index)

    # Fill in the values as tuples of (observed, retrodicted)
    for year in df_observed_cumulative.columns:
        combined_df[year] = list(zip(df_observed_cumulative[year], df_retrodict_cumulative[year]))

    display(combined_df)

    ## frontier counts

    # Group data into 6-month periods
    bins = pd.date_range(start=df.date.min(), end=df.date.max(), freq=period_freq).astype(f'period[{period_freq}]')
    df['period'] = pd.Series(bins[bins.searchsorted(df.date.dt.to_period(period_freq)) - 1], index=df.index)
    df['log_compute'] = np.log10(df['compute'])

    frontier_counts = []

    for period in df['period'].unique():
        period_data = df[df['period'] == period]
        if len(period_data) > 0:
            largest_model = period_data['compute'].max()
            
            for width in threshold_widths:
                count = (np.abs(np.log10(largest_model)-period_data['log_compute']) <= width).sum()
                
                frontier_counts.append({
                    'period': period.to_timestamp(),
                    'threshold_width': width,
                    'count': count,
                    'largest_model': largest_model
                })

    frontier_df = pd.DataFrame(frontier_counts)

    # Filter for 2020-2023 and pivot to create summary dataframe
    summary_df = frontier_df[
        (frontier_df['period'].dt.year >= 2020) & 
        (frontier_df['period'].dt.year <= 2023)
    ].pivot(
        index='period',
        columns='threshold_width',
        values='count'
    ) #basically a reshaping operation
    summary_df.columns = [f'width: {w}' for w in threshold_widths]


    # Create similar table for COMPUTE_SAMPLE_DATA
    sample_frontier_counts = {width: {} for width in threshold_widths}

    # For each period
    for period in pd.date_range(start='2020', end='2024', freq=period_freq).astype(f'period[{period_freq}]'):
        period_samples = []
        
        # Collect all samples from that period
        for year in COMPUTE_SAMPLE_DATA:
            pd_dt_dates = pd.to_datetime((COMPUTE_SAMPLE_DATA[year]['date']))
            if period.year == year:
                period_start = period.start_time
                period_end = period.end_time
                period_mask = (pd_dt_dates >= period_start) & (pd_dt_dates < period_end)
                period_samples.extend(np.array(COMPUTE_SAMPLE_DATA[year]['samples'])[period_mask])
                
        if period_samples:
            # Find largest model in period
            frontier = max(period_samples)
            
            # Count models within each threshold width
            for width in threshold_widths:
                count = sum(abs(np.log10(model) - np.log10(frontier)) <= width for model in period_samples)
                sample_frontier_counts[width][period] = count

    sample_summary_df = pd.DataFrame(sample_frontier_counts)
    sample_summary_df.columns = [f'width: {w}' for w in threshold_widths]


    # Group by year and sum
    summary_df = summary_df.groupby(summary_df.index.year).sum()
    sample_summary_df = sample_summary_df.groupby(sample_summary_df.index.year).sum()
    # Combine observed and retrodicted counts
    combined_df = pd.DataFrame()
    for col in summary_df.columns:
        combined_df[col] = list(zip(summary_df[col], sample_summary_df[col]))
    combined_df.index = range(2020, 2024)

    print("\nFrontier counts (observed, retrodicted):")
    display(combined_df)


