

def main():

    if 1: #===IMPORTS===
            
        import logging, time, os
        from datetime import datetime

        logging.getLogger().handlers.clear()

        # Generate log file name based on current date and time
        log_filename = datetime.now().strftime('logs/%Y-%m-%d_%H-%M-%S.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_filename),
                #logging.StreamHandler()
            ]

        )

        import time
        start_time = time.time()
        logging.info("Starting imports...")
        import numpy as np
        from scipy import stats, optimize
        import matplotlib.pyplot as plt
        import pandas as pd #taking long to load here
        import seaborn as sns
        import itertools
        import copy,re, pdb, logging
        from sklearn import linear_model
        from collections import defaultdict
        import warnings
        
        end_time = time.time()
        logging.info(f"Imports completed in {end_time - start_time:.2f} seconds")
            

        np.random.seed(None)
        warnings.filterwarnings("ignore")

    if 1: #UTILS

        #util funcs cell
        def norm_exp_func(x,a,b,k):
            norm_factor=(1/k)*(np.exp(k*b)-np.exp(k*a))
            return (1/norm_factor)*np.exp(k*x)


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


        def alloc_ratio_to_alloc(alloc_ratio):
            #note - assumes alloc_rati = train/inf
            alloc_ratio=np.array(alloc_ratio)
            train_alloc=alloc_ratio/(1+alloc_ratio)
            inference_alloc=1-train_alloc
            return train_alloc, inference_alloc

        def round_dates(dates, freq):
            #from Claude; unsure how this works
            if freq == '6M':
                return dates.map(lambda d: d.replace(day=1) + pd.offsets.MonthEnd(6 - (d.month - 1) % 6))
            elif freq == '3M':
                return dates.map(lambda d: d.replace(day=1) + pd.offsets.MonthEnd(3 - (d.month - 1) % 3))
            elif freq == '1M':
                return dates.map(lambda d: d.replace(day=1) + pd.offsets.MonthEnd(1))
            elif freq == '1Y':
                return dates.map(lambda d: d.replace(month=1, day=1) + pd.offsets.YearEnd())
            else:
                raise ValueError("Unsupported frequency")

        def truncated_normal(mean,std_dev,min=None,max=None,size=1):
            if min is None: min = mean-3*std_dev
            if max is None: max = mean+3*std_dev
            samples = np.random.normal(mean, std_dev, size)
            return np.clip(samples, min, max) 


    if 1: #===CONFIG===

        #workflow config
        PLOT_SCHEMATIC_SCATTER=False
        TRAINING_COMPUTE_PLOTS=False
        FIT_ALLOCATION_PLOTS=False
        GENERATED_SAMPLE_PLOTS=False
        SAVE_RESULTS=False

        #sampling parameters
        n_simulations = 10 #for bootstrappng, sampling parameters etc. n_simulations = 10 #for bootstrappng, sampling parameters etc. 

        #training compute extrapolation config 
        AI2027_EXTRAP=True
        method_choice="method 2027" #['linear extrapolation', 'method 2027']
        hist_alloc=1/1
        hist_alloc_multiplier=1+(1/hist_alloc)
        FIXED_ALLOCATION=True
        fixed_alloc=40/60
        DYNAMIC_ALLOCATION=False #inference scaling continues improving
        assert(FIXED_ALLOCATION+DYNAMIC_ALLOCATION)==1
        pred_alloc_dict = {
                2024: 40/60,
                2025: 30/70,
                2026: 30/70,
                2027: 20/80,
                2028: 20/80,
            }
        g_global_AI_compute_mean=2.5
        g_AI_workload_share_mean=1.5 #assuming AI_compute_usage/AI_compute_capacity = const - 3.0 gets the two superposed!
        g_total = g_global_AI_compute_mean + g_AI_workload_share_mean
        g_stdev=0.0 #get more reasonable values by fixing rather than computing from historical data


        #allocation fit parameters
        fit_years=np.arange(2020,2024)
        pred_years = np.arange(2024,2029)
        constraint_point=(1,1)
        filter_thresholds=1e-20 #ignore models smaller than this

        ##generate sample parameters
        CONST_FM=False
        LIN_EXTRAP_FM=False
        CUSTOM_FM=True
        if CUSTOM_FM:
            custom_fm_grad=1.0 #it would be nicer to set c
        assert(CONST_FM+LIN_EXTRAP_FM+CUSTOM_FM)==1, "Only one of CONST_FM, LIN_EXTRAP_FM, or CUSTOM_FM can be True"

        #IMPORTANT PARAMETER - largest model share
        FRONTIER_MODEL_GROWTH="coupled"
        min_norm_m = 10**-7
        largest_model_share_mean,lms_stddev,min_lms,max_lms=0.3, 0.2/3,None,None

        n_catgs = 50


        #threshold counting PARAMETERS
        thresholds=[25, 26, 27, 28, 29]
        threshold_widths = [0.5, 1, 1.5]  # List of threshold widths to analyze
        period_freq = '3M'  # frequency for doing frontier counts
        retrodict_thresholds=[23, 24, 25]


        #SAVE CONFIG
        SAVE_CONFIG={
            "historical allocation":hist_alloc,
            "(g_global_AI_compute, g_AI_workload_share)":(g_global_AI_compute_mean, g_AI_workload_share_mean) if method_choice=='method 2027' else None,
            "compute allocation config": "dynamic inference allocation" if DYNAMIC_ALLOCATION else "fixed inference allocation",
            "allocations": pred_alloc_dict if DYNAMIC_ALLOCATION else fixed_alloc,
            "frontier model config": f"lms_mean={largest_model_share_mean}, lms_stddev={lms_stddev}",
            "n_catgs":n_catgs,
            "allocation":CONST_FM if CONST_FM else LIN_EXTRAP_FM if LIN_EXTRAP_FM else CUSTOM_FM,
        }

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
        logging.info("=== Full Dataset ===")
        logging.info("Most recent date: %s", df["date"].max())
        logging.debug("\nDatapoints per year:")
        for year in range(2017, 2025):
            count = len(df[df["year"] == year])
            logging.debug("%d: %d", year, count)

        max_compute_idx = df['compute'].idxmax()
        logging.info("\nLargest compute value: %.2e (%s)", df.loc[max_compute_idx, 'compute'], df.loc[max_compute_idx, 'model'])
        # Create dataset without specified years
        years_to_exclude = [2025, 2024]  # List of years to exclude
        df_filtered = df[~df["year"].isin(years_to_exclude)].copy()

        logging.info("\n=== Dataset excluding years %s ===", years_to_exclude)
        logging.info("Most recent date: %s", df_filtered["date"].max())

        max_compute_idx = df_filtered['compute'].idxmax()
        logging.info("\nLargest compute value: %.2e (%s)", df_filtered.loc[max_compute_idx, 'compute'], df_filtered.loc[max_compute_idx, 'model'])

        df = df_filtered

        # Report number of entries before removing NaN
        logging.info("\n\n Number of entries before removing rows with compute=NaN: %d", len(df))
        # Remove rows with NaN in compute column
        df = df.dropna(subset=['compute'])
        # Report number of entries after removing rows with compute=NaN
        logging.info("Number of entries after removing rows with compute=NaN: %d", len(df))

        # Log datapoints per year in the dataframe after removing NaN compute values
        logging.info("\nDatapoints per year after removing rows with compute=NaN:")
        for year in range(2017, 2025):
            count = len(df[df["year"] == year])
            logging.info("%d: %d", year, count)

        
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
                

        ###DATA STRUCTURE INIT
        LOG_AGGREGATE_COMPUTE_DATA={}



        for sim in range(n_simulations):
            LOG_AGGREGATE_COMPUTE_DATA[sim] = {}

            year_grouped_df=df.groupby(df['date'][df['date']>'2010-01-01'].dt.year)
            aggregate_compute=year_grouped_df['compute'].sum()
            log_aggregate_compute=np.log10(aggregate_compute)

            recent_years = log_aggregate_compute[log_aggregate_compute.index.isin(range(2020,df.year.max()+1))]
            recent_log_compute_dict = {int(k): v for k, v in recent_years.items()}


            if 1: #do historical data
                LOG_AGGREGATE_COMPUTE_DATA[sim]['historical aggregate training compute'] = {int(k): v for k, v in log_aggregate_compute.items()}
                LOG_AGGREGATE_COMPUTE_DATA[sim]['historical aggregate total compute'] = {int(k): v+np.log10(hist_alloc_multiplier) for k, v in log_aggregate_compute.items()}

            if AI2027_EXTRAP:
                training_usage_2023 = 10**log_aggregate_compute.get(2023)
                total_usage_2023 = 2 * training_usage_2023

                AI_compute_usage={}
                for idx,year in enumerate(range(2024, 2029)):
                    AI_compute_usage[year] = total_usage_2023 * (g_total+np.random.normal(0,g_stdev)) ** (idx + 1)

                log_aggregate_compute_predictions_dict = {year: np.log10(compute) for year, compute in AI_compute_usage.items()}
                LOG_AGGREGATE_COMPUTE_DATA[sim]['Total-method 2027'] = log_aggregate_compute_predictions_dict


            #do allocations
            if 1: 
                if FIXED_ALLOCATION:
                    train_alloc,inference_alloc=alloc_ratio_to_alloc(alloc_ratio=fixed_alloc)
                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate training compute'] = {year: val + np.log10(train_alloc) for year, val in LOG_AGGREGATE_COMPUTE_DATA[sim][f"Total-{method_choice}"].items()}
                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate inference compute'] = {year: val + np.log10(inference_alloc) for year, val in LOG_AGGREGATE_COMPUTE_DATA[sim][f"Total-{method_choice}"].items()}

                if DYNAMIC_ALLOCATION:
                    train_alloc_dict = {}
                    inference_alloc_dict = {}

                    for year, val in LOG_AGGREGATE_COMPUTE_DATA[sim][f'Total-{method_choice}'].items():
                        alloc_ratio=pred_alloc_dict.get(year,1.0)
                        train_alloc, inference_alloc = alloc_ratio_to_alloc(alloc_ratio=alloc_ratio)
                        train_alloc_dict[year] = val + np.log10(train_alloc)
                        inference_alloc_dict[year] = val + np.log10(inference_alloc)

                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate training compute'] = train_alloc_dict
                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate inference compute'] = inference_alloc_dict


        if TRAINING_COMPUTE_PLOTS:
            plt.figure(figsize=(10, 6))

            # Plot extrapolations for each method
            colors = {
                'historical aggregate training compute': 'blue',
                'historical aggregate total compute': 'cyan',
                'Total-method 2027': 'purple',
                'aggregate training compute': 'green',
                'aggregate inference compute': 'red',
            }
            markers = {
                'historical aggregate training compute': 'o',
                'historical aggregate total compute': 'v',
                'Total-method 2027': 's',
                'aggregate training compute': '.',
                'aggregate inference compute': 'x',
            }

            for method in colors.keys():
                all_sim_values = defaultdict(list)
                
                for sim in range(n_simulations):
                    predictions = LOG_AGGREGATE_COMPUTE_DATA[sim].get(method, {})
                    for year, value in predictions.items():
                        all_sim_values[year].append(value)
                
                years = sorted(all_sim_values.keys())
                medians = [np.median(all_sim_values[year]) for year in years]
                lower_bounds = [np.percentile(all_sim_values[year], 5) for year in years]
                upper_bounds = [np.percentile(all_sim_values[year], 95) for year in years]

                plt.plot(years, medians, label=f'{method} (Median)', color=colors[method], marker=markers[method])
                if "historical" not in method:
                    plt.fill_between(years, lower_bounds, upper_bounds, color=colors[method], alpha=0.2, label=f'{method} (90% CI)')

            plt.xlabel('Year')
            plt.ylabel('Log10(Compute) [FLOP]')
            plt.title(f'Compute Usage Over Time')
            plt.legend()
            plt.grid(True)
            plt.xticks(np.arange(min(log_aggregate_compute.index), 2030, 2))

            # Plot compute allocations for different tau values
            plt.figure(figsize=(10, 6))

            years = sorted(pred_alloc_dict.keys())
            alloc_ratios = [pred_alloc_dict[y] for y in years]

            train_allocs = []
            inference_allocs = []
            if FIXED_ALLOCATION:
                train_allocs, inference_allocs = alloc_ratio_to_alloc(np.ones(years.__len__()) * fixed_alloc)
            if DYNAMIC_ALLOCATION:
                train_allocs, inference_allocs = alloc_ratio_to_alloc(np.array(list(pred_alloc_dict.values())))

            plt.plot(years, train_allocs, 'g-', label='Training Allocation')
            plt.plot(years, inference_allocs, 'r-', label='Inference Allocation')
            plt.scatter(years, train_allocs, color='green', marker='o')
            plt.scatter(years, inference_allocs, color='red', marker='o')
            plt.ylim(0, 1)

            plt.xlabel('Year')
            plt.ylabel('Allocation Fraction')
            plt.title('Compute Allocations Over Time')
            plt.legend()
            plt.grid(True)
            plt.xticks(years)



    if 1: #fit allocations 
        FIT_DATA={year:None for year in fit_years}


        print('Fitting f_M coefficients')

        for idx,year in enumerate(fit_years):
            total_compute=aggregate_compute[aggregate_compute.index==year].values
            datapoints_year=df[df['date'].dt.year==year]['compute']
            mean_log_compute=np.log10(datapoints_year).mean()
            largest_model=datapoints_year.max()
            smallest_model=datapoints_year.min()
            norm_factor_total=total_compute[0]

            sorted_computes=np.sort(datapoints_year)
            norm_sorted_computes=sorted_computes/largest_model
            
            cumsum=np.cumsum(sorted_computes)
            norm_cum_alloc=cumsum/norm_factor_total
            _norm_catg_alloc_ = np.diff(norm_cum_alloc)
            norm_catg_alloc = np.concatenate((np.array([1-np.sum(_norm_catg_alloc_)]) , _norm_catg_alloc_))

            #store data 
            FIT_DATA[year]={
            'compute':sorted_computes,
            'cumulative_sum':cumsum,
            'norm_factor_total':norm_factor_total,
            'largest_model':largest_model,
            'norm_smallest_model':smallest_model/largest_model,
            'norm_cum_alloc fits':None,
            'norm_catg_alloc fits':None,
                    }
            
            #fit data
            X = np.log10(norm_sorted_computes).reshape(-1, 1)
            y = np.log10(norm_cum_alloc)
            X_trans,y_trans=X-constraint_point[0],y-constraint_point[1]
            reg_cum_alloc = linear_model.LinearRegression(fit_intercept=False).fit(X_trans, y_trans) #forcing X-a,y-b to go through (0,0) means X,y goes through (a,b)
            FIT_DATA[year]['cum_alloc_fits'] = [reg_cum_alloc.coef_[0], reg_cum_alloc.intercept_]


            X,y = np.log10(norm_sorted_computes).reshape(-1,1),np.log10(norm_catg_alloc).reshape(-1,1)
            reg_catg_alloc = linear_model.LinearRegression(fit_intercept=True).fit(X,y) #we claim that there is a linear relationship between log(norm_computes) and log(catg_alloc)
            FIT_DATA[year]['catg_alloc_fits'] = [reg_catg_alloc.coef_[0][0], reg_catg_alloc.intercept_[0]]
            FIT_DATA[year]['norm_catg_alloc']=norm_catg_alloc

        if FIT_ALLOCATION_PLOTS:
            fig, ax = plt.subplots(figsize=(10, 6))
            for year in fit_years:
                if year in FIT_DATA:
                    data = FIT_DATA[year]
                    norm_sorted_computes = data['compute'] / data['largest_model']
                    norm_catg_alloc = data['norm_catg_alloc']
                    m, b = data['catg_alloc_fits']
                    
                    ax.scatter(norm_sorted_computes, norm_catg_alloc, marker='o', s=50, label=f'Year {year} Data points')
                    log_catg_alloc = m * np.log10(norm_sorted_computes) + b #the relationship in log space
                    y_fit = 10**(log_catg_alloc)
                    ax.plot(norm_sorted_computes, y_fit, label=f'Year {year} Fit (m={m:.2f}, b={b:.2f})')
            
            ax.set_xlabel('Normalized Sorted Computes')
            ax.set_ylabel('Normalized Category Allocations')
            ax.set_title('Category Allocations and Fits')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True)
            plt.show()
            

        # Log debug - Print cum_alloc_fits for all years
        logging.info("cum_alloc_fits for each year:")
        for year in fit_years:
            cum_alloc_coeffs = FIT_DATA[year]['cum_alloc_fits']
            catg_alloc_coeffs = FIT_DATA[year]['catg_alloc_fits']
            logging.info(f"Year {year}: cum_alloc_fits - slope={cum_alloc_coeffs[0]:.4f}, intercept={cum_alloc_coeffs[1]:.4f}")
            logging.info(f"Year {year}: catg_alloc_fits - slope={catg_alloc_coeffs[0]:.4f}, intercept={catg_alloc_coeffs[1]:.4f}")

            
    if 1: #Generate compute samples


        all_years=np.concatenate([fit_years, pred_years.astype(int).ravel()])
        COMPUTE_SAMPLE_DATA = {sim: {int(year): {} for year in all_years} for sim in range(n_simulations)} #init data structure 

        for sim in range(n_simulations):

            #build in sampling
            if FRONTIER_MODEL_GROWTH=='coupled':
                norm_largest_model = truncated_normal(mean=largest_model_share_mean,std_dev=lms_stddev,min=min_lms,max=max_lms, size=1)[0]


            for year in all_years:

                #get total compute
                if year in fit_years:
                    log_agg_training_compute = LOG_AGGREGATE_COMPUTE_DATA[sim]["historical aggregate training compute"][year]
                if year in pred_years:
                    log_agg_training_compute = LOG_AGGREGATE_COMPUTE_DATA[sim]["aggregate training compute"][year]
                agg_training_compute = 10**log_agg_training_compute  # total compute used over the year

                #set largest model that year 
                if FRONTIER_MODEL_GROWTH=='coupled':
                    if year in fit_years: #remove this eventually
                        largest_model = FIT_DATA[year]['largest_model']
                        largest_model = norm_largest_model * agg_training_compute

                    else:
                        largest_model = norm_largest_model * agg_training_compute



                logging.info(f"Ratio of largest model to aggregate training compute for year {year}: {largest_model / agg_training_compute}")

                # model sizes (as fraction of largest_model)
                norm_ms = np.logspace(np.log10(min_norm_m), np.log10(1.0), num=n_catgs)
                log_norm_ms = np.log10(norm_ms)

                if CONST_FM:
                    fm_grad, fm_int = np.mean([FIT_DATA[year]['cum_alloc_fits'][0] for year in FIT_DATA]), np.mean([FIT_DATA[year]['cum_alloc_fits'][1] for year in FIT_DATA])
                elif LIN_EXTRAP_FM:
                    raise NotImplementedError("Linear extrapolation of fm_grad and fm_int not implemented")
                elif CUSTOM_FM:
                    fm_grad = custom_fm_grad
                if year in FIT_DATA.keys():
                    fm_grad, fm_int = FIT_DATA[year]['cum_alloc_fits']

                log_frac_cum_compute = fm_grad * log_norm_ms + fm_int
                frac_cum_compute = 10**log_frac_cum_compute
                assert np.round(np.sum(np.diff(frac_cum_compute)), 5) == 1.0  # allocations should sum to 1

                if year == 2024:
                    for idx, frac_alloc in enumerate(np.diff(frac_cum_compute)):
                        logging.debug(f"models {norm_ms[idx]} - {norm_ms[idx+1]} alloc: {frac_alloc}")

                model_ctgs = [f'{norm_ms[i]:.2e}--{norm_ms[i+1]:.2e}' for i in range(len(norm_ms) - 1)]
                f_m = np.diff(frac_cum_compute)  # we don't include compute alloc to models 1e-8 smaller than total compute
                bin_compute_allocs = f_m * agg_training_compute  # array of how much compute allocated to each bin
                DATA_alloc = {model_ctgs[i]: {'compute alloc': bin_compute_allocs[i]} for i in range(len(model_ctgs))}

                compute_samples_rand = []

                for idx, (ctg, alloc) in enumerate(list(zip(model_ctgs, bin_compute_allocs))):
                    # here alloc is the amount of alloc given to each individual bin
                    bounds = ctg.split('--')
                    norm_model_bin_lb, norm_model_bin_ub = float(bounds[0]), float(bounds[1])
                    model_bin_lb, model_bin_ub = largest_model * norm_model_bin_lb, largest_model * norm_model_bin_ub  # normalising factor is total training compute

                    if alloc > model_bin_ub:
                        pass  # do nothing; we're good
                    else:
                        capped_model_bin_ub = alloc  # cap the ub at the alloc
                        model_bin_ub = capped_model_bin_ub
                        logging.info(f"alloc: {alloc} < model_bin_ub: {model_bin_ub}. Capping model_bin_ub at alloc.")

                    # not generating multiple samples yet for CIs
                    allocnorm_model_bin_lb, allocnorm_model_bin_ub = model_bin_lb / alloc, model_bin_ub / alloc  # this is purely just for sampling; no physical meaning
                    running_tot = 0
                    allocnormed_samples = []
                    while running_tot < 1:
                        # SAMPLE
                        sample = np.random.uniform(allocnorm_model_bin_lb, allocnorm_model_bin_ub)
                        sample = float(sample) if isinstance(sample, np.ndarray) else sample

                        # SUM CHECK
                        if running_tot + sample > 1:
                            allocnormed_samples.append(1 - running_tot)
                            running_tot = 1
                        else:
                            allocnormed_samples.append(sample)
                            running_tot += sample

                    compute_samples_rand = compute_samples_rand + (list(alloc * np.array(allocnormed_samples)))

                compute_samples_rand = [x for x in compute_samples_rand if x != 0]

                COMPUTE_SAMPLE_DATA[sim][year]['samples'] = compute_samples_rand
                COMPUTE_SAMPLE_DATA[sim][year]['date'] = [decimal_year_to_date(year + np.random.random()) for _ in compute_samples_rand]  # convert to standard pd datetime format
                COMPUTE_SAMPLE_DATA[sim][year]['largest model'] = largest_model


        logging.debug("\nNumber of samples per year:")
        for year in pred_years.ravel():
            logging.debug(f"{year}: {len(COMPUTE_SAMPLE_DATA[0][year]['samples'])} samples") #take first sim

                

        if GENERATED_SAMPLE_PLOTS:
            fig, axes = plt.subplots(3, 2, figsize=(12, 8))
            axes = axes.ravel()

            for idx, year in enumerate(pred_years):
                all_samples = [np.log10(COMPUTE_SAMPLE_DATA[sim][year]['samples']) for sim in range(n_simulations)]
                
                # Define a common set of x-points for KDE evaluation
                x_points = np.linspace(15, 30, 1000)
                
                # Evaluate KDE for each simulation
                from scipy.stats import gaussian_kde
                kde_values = [gaussian_kde(samples)(x_points) for samples in all_samples]
                
                # Calculate median and 90th percentile KDE values at each x-point
                median_kde = np.median(kde_values, axis=0)
                lower_bound_kde = np.percentile(kde_values, 5, axis=0)
                upper_bound_kde = np.percentile(kde_values, 95, axis=0)
                
                # Plot the median KDE
                axes[idx].plot(x_points, median_kde, label='Median KDE')
                
                # Fill between the 5th and 95th percentile KDE values
                axes[idx].fill_between(x_points, lower_bound_kde, upper_bound_kde, alpha=0.3, label='90% CI')
                
                axes[idx].set_title(f'Year {year}')
                axes[idx].set_xlabel('log compute (FLOPs)')
                axes[idx].set_ylabel('Density')
                axes[idx].grid(alpha=0.5)
                axes[idx].set_xlim([15, 30])
                axes[idx].legend()

            plt.tight_layout()
            plt.show()

        if GENERATED_SAMPLE_PLOTS:
            ylims = (14, 30)
            plt.figure(figsize=(12, 6))
            plt.scatter(df[df['year'].isin(fit_years)]['date'], np.log10(df[df['year'].isin(fit_years)]['compute']), alpha=0.5, label='Historical')
            
            for year in pred_years:
                sample_data = COMPUTE_SAMPLE_DATA[0][year]
                plt.scatter(sample_data['date'], np.log10(sample_data['samples']), alpha=0.5, label='Projected Samples' if year == pred_years[0] else "", color='red')

            plt.xlabel('Year')
            plt.ylabel('Log Compute (FLOPs)')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.ylim(ylims)
            plt.yticks(np.arange(ylims[0], ylims[1], 0.5))
            plt.show()


    if 1: # verification retrodiction

        retrodict_years=fit_years

        #observed
        # Create DataFrame from observed counts
        df_observed = pd.DataFrame.from_dict({threshold: {year: sum(df[df['year'] == year]['compute'] > 10**threshold)
                                                        for year in retrodict_years}
                                            for threshold in retrodict_thresholds}, 
                                            orient='index')
        df_observed.index = [f'{10**threshold:.2e}' for threshold in retrodict_thresholds]
        df_observed.index.name = 'Threshold'

        # Create retrodict counts dictionary
        retrodict_counts = {year: {threshold: [] for threshold in retrodict_thresholds} for year in retrodict_years}

        for sim, sim_data in COMPUTE_SAMPLE_DATA.items():
            for year, year_data in sim_data.items():
                if year in retrodict_years:
                    for threshold in retrodict_thresholds:
                        count = (sum(x >= 10**threshold for x in year_data['samples'])).astype(int)
                        retrodict_counts[year][threshold].append(count)

        # Calculate median for each year and threshold
        retrodict_median_counts = {year: [] for year in retrodict_years}
        for year in retrodict_years:
            for threshold in retrodict_thresholds:
                median_count = (np.median(retrodict_counts[year][threshold])).astype(int)
                retrodict_median_counts[year].append(median_count)

        df_retrodict = pd.DataFrame(retrodict_median_counts,
                                index=[f'{10**t:.2e}' for t in retrodict_thresholds])
        df_retrodict.index.name = 'Threshold'

        # Take cumulative sum across years for both dataframes
        df_observed_cumulative = df_observed.cumsum(axis=1)
        df_retrodict_cumulative = df_retrodict.cumsum(axis=1)


        # Create dataframe with observed and retrodicted values
        combined_df = pd.DataFrame(index=df_observed_cumulative.index)

        # Fill in the values as tuples of (observed, retrodicted)
        for year in df_observed_cumulative.columns:
            combined_df[year] = list(zip(df_observed_cumulative[year], df_retrodict_cumulative[year]))


        # Calculate the difference between observed and retrodicted values
        difference_df = df_observed_cumulative - df_retrodict_cumulative

        absolute_threshold_retrodicted = combined_df
        absolute_threshold_retrodicted_difference = difference_df



        #retrodiction for frontier counts
        # Group data into specified periods
        df['period'] = round_dates(df['date'], period_freq)
        df['log_compute'] = np.log10(df['compute'])

        frontier_counts = {}

        for year in fit_years:
            year_filtered_df = df[df['date'].dt.year == year]
            frontier_counts[year] = {}
            for width in threshold_widths:
                width_year_counts = 0
                for idx, period in enumerate(sorted(year_filtered_df['period'].unique())):
                    largest_model = df[df['period'] < period]['compute'].max()  # get largest model before this period
                    period_data = df[df.period == period]
                    within_threshold_condition = ((np.log10(largest_model) - np.log10(period_data['compute'])) <= width) & ((np.log10(largest_model) - np.log10(period_data['compute'])) > 0)
                    above_frontier_condition = period_data['compute'] > largest_model
                    count = within_threshold_condition.sum() + above_frontier_condition.sum()
                    width_year_counts += count
                frontier_counts[year][width] = width_year_counts

        # Calculate median projection for retrodicted counts
        sample_frontier_counts = {year: {width: [] for width in threshold_widths} for year in fit_years}

        for sim, sim_data in COMPUTE_SAMPLE_DATA.items():
            for year in fit_years:
                year_data = sim_data[year]
                year_data['period'] = round_dates(pd.to_datetime(year_data['date']), period_freq)
                year_data['log_compute'] = np.log10(year_data['samples'])
                
                for width in threshold_widths:
                    width_year_counts = 0
                    for period in sorted(year_data['period'].unique()):
                        largest_model = max(np.concatenate([np.array(data['samples'])[np.array(data['date']) < period] for data in sim_data.values()]))
                        period_sample_data = np.array(year_data['samples'])[year_data['period'] == period]
                        within_threshold_condition = (np.log10(largest_model) - np.log10(period_sample_data) <= width) & (np.log10(largest_model) - np.log10(period_sample_data) > 0)
                        above_frontier_condition = period_sample_data > largest_model
                        width_year_counts += within_threshold_condition.sum() + above_frontier_condition.sum()
                    sample_frontier_counts[year][width].append(width_year_counts)

        # Calculate median for each year and width
        median_sample_frontier_counts = {year: {width: (np.median(sample_frontier_counts[year][width])).astype(int) for width in threshold_widths} for year in fit_years}

        combined_counts = {}

        for width in threshold_widths:
            combined_counts[width] = {}
            for year in fit_years:
                a = frontier_counts[year][width]
                b = median_sample_frontier_counts[year][width]
                combined_counts[width][year] = (a, b)

        combined_df = pd.DataFrame(combined_counts).T
        combined_df.index.name = 'width'
        combined_df.columns.name = 'year'


        difference_counts = {}

        for width in threshold_widths:
            difference_counts[width] = {}
            for year in fit_years:
                observed = frontier_counts[year][width]
                predicted = median_sample_frontier_counts[year][width]
                difference_counts[width][year] = observed - predicted

        difference_df = pd.DataFrame(difference_counts).T
        difference_df.index.name = 'width'
        difference_df.columns.name = 'year'

        frontier_threshold_retrodicted = combined_df
        frontier_threshold_retrodicted_difference = difference_df



    if 1: # predictions

        #predictions for absolute thresholds
        threshold_counts_all_simulations = {year: {threshold: [] for threshold in thresholds} for year in pred_years.astype(int).ravel()}

        # Iterate over each simulation
        for sim in range(len(COMPUTE_SAMPLE_DATA)):
            for year, samples in COMPUTE_SAMPLE_DATA[sim].items():
                if year in pred_years:
                    for threshold in thresholds:
                        count = sum(x >= 10**threshold for x in samples['samples'])
                        threshold_counts_all_simulations[year][threshold].append(count)

        # Calculate median and 90% CI for each year and threshold
        threshold_counts_summary = {year: [] for year in pred_years.astype(int).ravel()}
        for year in pred_years.astype(int).ravel():
            for threshold in thresholds:
                counts = threshold_counts_all_simulations[year][threshold]
                median_count = np.median(counts)
                lower_bound = np.percentile(counts, 5)
                upper_bound = np.percentile(counts, 95)
                threshold_counts_summary[year].append(f"{median_count:.0f} ({lower_bound:.0f}-{upper_bound:.0f})")

        df_median_counts = pd.DataFrame({year: [int(round(np.median(threshold_counts_all_simulations[year][threshold]))) for threshold in thresholds] for year in pred_years.astype(int).ravel()}, index=[f'>1e{t}' for t in thresholds])
        df_5th_percentile_counts = pd.DataFrame({year: [int(round(np.percentile(threshold_counts_all_simulations[year][threshold], 5))) for threshold in thresholds] for year in pred_years.astype(int).ravel()}, index=[f'>1e{t}' for t in thresholds])
        df_95th_percentile_counts = pd.DataFrame({year: [int(round(np.percentile(threshold_counts_all_simulations[year][threshold], 95))) for threshold in thresholds] for year in pred_years.astype(int).ravel()}, index=[f'>1e{t}' for t in thresholds])

        # Make cumulative across years
        df_median_cumulative = df_median_counts.cumsum(axis=1)
        df_5th_percentile_cumulative = df_5th_percentile_counts.cumsum(axis=1)
        df_95th_percentile_cumulative = df_95th_percentile_counts.cumsum(axis=1)

        # Combine into a single DataFrame
        df_combined_cumulative = df_median_cumulative.astype(str) + " (" + df_5th_percentile_cumulative.astype(str) + "-" + df_95th_percentile_cumulative.astype(str) + ")"

        if 0: 
            for sim in range(len(COMPUTE_SAMPLE_DATA)):
                for year, samples in COMPUTE_SAMPLE_DATA[sim].items():
                    if year in pred_years:
                        print(f"Year {year}: {len(samples['samples'])} samples")


        absolute_threshold_predicted = df_combined_cumulative


        #predictions for frontier counts
        # Generate period data for years 2024-2029 (2029 not inclusive)
        period_data = pd.date_range(start='2024-01-01', end='2029-01-01', freq='6M').strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        frontier_counts_all_simulations = {year: {width: [] for width in threshold_widths} for year in pred_years}

        for sim in range(len(COMPUTE_SAMPLE_DATA)):
            for year in pred_years:
                year_data = COMPUTE_SAMPLE_DATA[sim][year]
                year_data['period'] = round_dates(pd.to_datetime(year_data['date']), period_freq)
                year_data['log_compute'] = np.log10(year_data['samples'])
                
                for width in threshold_widths:
                    width_year_counts = 0
                    for period in sorted(year_data['period'].unique()):
                        largest_model = max(np.concatenate([np.array(data['samples'])[np.array(data['date']) < period] for data in COMPUTE_SAMPLE_DATA[sim].values()])) #get largest model until this period
                        period_sample_data = np.array(year_data['samples'])[year_data['period'] == period] #get models released in this period
                        within_threshold_condition = (np.log10(largest_model) - np.log10(period_sample_data) <= width) & (np.log10(largest_model) - np.log10(period_sample_data) > 0) #0 condition makes sure we don't catch models larger than frontier
                        above_frontier_condition  = period_sample_data > largest_model
                        count = within_threshold_condition.sum() + above_frontier_condition.sum() #how many models released this period within thresholds of largest model seen so far.
                        width_year_counts += count
                    frontier_counts_all_simulations[year][width].append(width_year_counts)
        
        # Calculate median and 90% CI for each year and width
        frontier_counts_summary = {year: [] for year in pred_years}
        for year in pred_years:
            for width in threshold_widths:
                counts = frontier_counts_all_simulations[year][width]
                median_count = np.median(counts)
                lower_bound = np.percentile(counts, 5)
                upper_bound = np.percentile(counts, 95)
                frontier_counts_summary[year].append(f"{median_count:.0f} ({lower_bound:.0f}-{upper_bound:.0f})")

        # Convert to DataFrame
        df_frontier_counts = pd.DataFrame(frontier_counts_summary).T
        df_frontier_counts.index.name = 'Year'
        df_frontier_counts.columns = [f'Within {width} OOM' for width in threshold_widths]
        df_frontier_counts = df_frontier_counts.transpose()

        frontier_threshold_predicted = df_frontier_counts


    if 1: #display and save

        if not SAVE_RESULTS:
            logging.info("Displaying results...\n")
            logging.info("=== Retrodicted Thresholds ===")
            logging.info("=== Absolute Threshold Retrodicted ===")
            display(absolute_threshold_retrodicted)
            display(absolute_threshold_retrodicted_difference)
            logging.info("=== Frontier Threshold Retrodicted ===")
            display(frontier_threshold_retrodicted)
            display(frontier_threshold_retrodicted_difference)
            logging.info("=== Predicted Thresholds ===")
            logging.info("=== Absolute Threshold Predicted ===")
            display(absolute_threshold_predicted)
            logging.info("=== Frontier Threshold Predicted ===")
            display(frontier_threshold_predicted)
        

        if SAVE_RESULTS:
        # Create results directory if it doesn't exist
            if not os.path.exists('results'):
                os.makedirs('results')

            # Get current date
            time.sleep(1) #just to get different file names
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Save tables to results file
            with open(f'results/{current_date}_thresholds.csv', 'w') as f:
                for key, value in SAVE_CONFIG.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                f.write("Absolute Threshold Retrodicted:\n")
                absolute_threshold_retrodicted.to_csv(f,sep='\t')
                f.write('\n\n')
                f.write("Frontier Threshold Retrodicted:\n")
                frontier_threshold_retrodicted.to_csv(f,sep='\t')
                f.write('\n\n')
                f.write("Absolute Threshold Predicted:\n")
                absolute_threshold_predicted.to_csv(f,sep='\t')
                f.write('\n\n')
                f.write("Frontier Threshold Predicted:\n")
                frontier_threshold_predicted.to_csv(f,sep='\t')


main()