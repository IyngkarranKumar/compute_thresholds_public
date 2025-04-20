import config, importlib
importlib.reload(config)
from config import Config

def main(Config):

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
        from IPython.display import display
        import wandb 
        
        end_time = time.time()
        logging.info(f"Imports completed in {end_time - start_time:.2f} seconds")
            

        np.random.seed(None)
        warnings.filterwarnings("ignore")

    if 1: #UTILS

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


    if 1: #===Config===

        #SAVE Config
        SAVE_Config={
            "workflow config": {
                "name": Config.name,
                "PLOT_SCHEMATIC_SCATTER": Config.PLOT_SCHEMATIC_SCATTER,
                "TRAINING_COMPUTE_PLOTS": Config.TRAINING_COMPUTE_PLOTS,
                "FIT_ALLOCATION_PLOTS": Config.FIT_ALLOCATION_PLOTS,
                "GENERATED_SAMPLE_PLOTS": Config.GENERATED_SAMPLE_PLOTS,
                "SAVE_RESULTS": Config.SAVE_RESULTS,
                "save_folder": Config.save_folder
            },
            "sampling parameters": {
                "n_simulations": Config.n_simulations
            },
            "training compute extrapolation": {
                "method_choice": Config.method_choice,
                "historical allocation": Config.hist_alloc,
                "hist_alloc_multiplier": Config.hist_alloc_multiplier,
                "allocation type": "fixed" if Config.FIXED_ALLOCATION else "dynamic",
                "fixed allocation": Config.fixed_alloc if Config.FIXED_ALLOCATION else None,
                "predicted allocations": Config.pred_alloc_dict if Config.DYNAMIC_ALLOCATION else None,
                "growth parameters": {
                    "g_historical": Config.g_historical,
                    "g_global_AI_compute_mean": Config.g_global_AI_compute_mean,
                    "g_AI_workload_share_mean": Config.g_AI_workload_share_mean,
                    "g_weights": Config.g_weights,
                    "g_stdev": Config.g_stdev
                }
            },
            "allocation fit parameters": {
                "fit_years": Config.fit_years.tolist(),
                "pred_years": Config.pred_years.tolist(),
                "constraint_point": Config.constraint_point,
                "filter_thresholds": Config.filter_thresholds
            },
            "sampling parameters": {
                "ALLOC_FIT_TYPE": Config.ALLOC_FIT_TYPE,
                "POINT_CUM_ALLOC_PARAMS": Config.POINT_CUM_ALLOC_PARAMS,
                "DISTRIBUTION_CUM_ALLOC_PARAMS": Config.DISTRIBUTION_CUM_ALLOC_PARAMS,
                "grad_cum_alloc_range": [Config.grad_cum_alloc_min, Config.grad_cum_alloc_max],
                "LMS_SAMPLING": Config.LMS_SAMPLING,
                "largest_model_share": {
                    "min": Config.min_lms,
                    "max": Config.max_lms
                },
                "SET_2024_LMS": Config.SET_2024_LMS,
                "min_norm_m_range": [Config.min_norm_m_min, Config.min_norm_m_max],
                "n_catgs": Config.n_catgs
            },
            "threshold parameters": {
                "thresholds": Config.thresholds,
                "retrodict_thresholds": Config.retrodict_thresholds,
                "threshold_widths": Config.threshold_widths,
                "period_freq": Config.period_freq,
                "CI_percentiles": Config.CI_percentiles
            }
        }


    if 1: #===DATA LOADING===
        #Feb 2025 dataset

        #path 
        path="data/notable_ai_models_24_02_2025.csv"

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
        to_remove = ["AlphaGo Zero", "AlphaZero","AlphaGo Master"] #historial outliers
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

        
        if Config.PLOT_SCHEMATIC_SCATTER: #generate basic scatterplot
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

        g_AI_2027 = Config.g_global_AI_compute_mean*Config.g_AI_workload_share_mean
        g_total = Config.g_weights[0]*Config.g_historical + Config.g_weights[1]*g_AI_2027
        SAVE_Config['training compute extrapolation']['g_total'] = g_total #for saving 
        SAVE_Config['training compute extrapolation']['g_AI_2027'] = g_AI_2027 #for saving
                
        ###DATA STRUCTURE INIT
        LOG_AGGREGATE_COMPUTE_DATA={}


        for sim in range(Config.n_simulations):
            LOG_AGGREGATE_COMPUTE_DATA[sim] = {}

            year_grouped_df=df.groupby(df['date'][df['date']>'2010-01-01'].dt.year)
            aggregate_compute=year_grouped_df['compute'].sum()
            log_aggregate_compute=np.log10(aggregate_compute)

            recent_years = log_aggregate_compute[log_aggregate_compute.index.isin(Config.fit_years)]

            if 1: #do historical data
                LOG_AGGREGATE_COMPUTE_DATA[sim]['historical aggregate training compute'] = {int(k): v for k, v in log_aggregate_compute.items()}
                LOG_AGGREGATE_COMPUTE_DATA[sim]['historical aggregate total compute'] = {int(k): v+np.log10(Config.hist_alloc_multiplier) for k, v in log_aggregate_compute.items()}

            if Config.AI2027_EXTRAP:
                previous_year_training_usage = 10**log_aggregate_compute.get(Config.fit_years[-1])
                total_usage_previous_year = Config.hist_alloc_multiplier * previous_year_training_usage

                AI_compute_usage={}
                sim_noise_term=np.random.normal(0,Config.g_stdev) #set noise term for each sim 
                for idx,year in enumerate(Config.pred_years):
                    AI_compute_usage[year] = total_usage_previous_year * (g_total+sim_noise_term) ** (idx + 1)

                log_aggregate_compute_predictions_dict = {year: np.log10(compute) for year, compute in AI_compute_usage.items()}
                LOG_AGGREGATE_COMPUTE_DATA[sim]['Total-method 2027'] = log_aggregate_compute_predictions_dict


            #do allocations
            if 1: 
                if Config.FIXED_ALLOCATION:
                    train_alloc,inference_alloc=alloc_ratio_to_alloc(alloc_ratio=Config.fixed_alloc)
                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate training compute'] = {year: val + np.log10(train_alloc) for year, val in LOG_AGGREGATE_COMPUTE_DATA[sim][f"Total-{Config.method_choice}"].items()}
                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate inference compute'] = {year: val + np.log10(inference_alloc) for year, val in LOG_AGGREGATE_COMPUTE_DATA[sim][f"Total-{Config.method_choice}"].items()}

                if Config.DYNAMIC_ALLOCATION:
                    train_alloc_dict = {}
                    inference_alloc_dict = {}

                    for year, val in LOG_AGGREGATE_COMPUTE_DATA[sim][f'Total-{Config.method_choice}'].items():
                        alloc_ratio=Config.pred_alloc_dict.get(year,1.0)
                        train_alloc, inference_alloc = alloc_ratio_to_alloc(alloc_ratio=alloc_ratio)
                        train_alloc_dict[year] = val + np.log10(train_alloc)
                        inference_alloc_dict[year] = val + np.log10(inference_alloc)

                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate training compute'] = train_alloc_dict
                    LOG_AGGREGATE_COMPUTE_DATA[sim]['aggregate inference compute'] = inference_alloc_dict


        if Config.TRAINING_COMPUTE_PLOTS:
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
                
                for sim in range(Config.n_simulations):
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

            years = sorted(Config.pred_alloc_dict.keys())

            train_allocs = []
            inference_allocs = []
            if Config.FIXED_ALLOCATION:
                train_allocs, inference_allocs = alloc_ratio_to_alloc(np.ones(years.__len__()) * Config.fixed_alloc)
            if Config.DYNAMIC_ALLOCATION:
                train_allocs, inference_allocs = alloc_ratio_to_alloc(np.array(list(Config.pred_alloc_dict.values())))

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
        FIT_DATA={year:None for year in Config.fit_years}


        logging.info('Fitting f_M coefficients')

        for idx,year in enumerate(Config.fit_years):
            total_compute=aggregate_compute[aggregate_compute.index==year].values
            datapoints_year=df[df['date'].dt.year==year]['compute']
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
            X_trans,y_trans=X-Config.constraint_point[0],y-Config.constraint_point[1]
            reg_cum_alloc = linear_model.LinearRegression(fit_intercept=False).fit(X_trans, y_trans) #forcing X-a,y-b to go through (0,0) means X,y goes through (a,b)
            FIT_DATA[year]['cum_alloc_fits'] = [reg_cum_alloc.coef_[0], reg_cum_alloc.intercept_]


            X,y = np.log10(norm_sorted_computes).reshape(-1,1),np.log10(norm_catg_alloc).reshape(-1,1)
            reg_catg_alloc = linear_model.LinearRegression(fit_intercept=True).fit(X,y) #we claim that there is a linear relationship between log(norm_computes) and log(catg_alloc)
            FIT_DATA[year]['catg_alloc_fits'] = [reg_catg_alloc.coef_[0][0], reg_catg_alloc.intercept_[0]]
            FIT_DATA[year]['norm_catg_alloc']=norm_catg_alloc

        if Config.FIT_ALLOCATION_PLOTS:
            fig, ax = plt.subplots(figsize=(10, 6))
            for year in Config.fit_years:
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
        for year in Config.fit_years:
            cum_alloc_coeffs = FIT_DATA[year]['cum_alloc_fits']
            catg_alloc_coeffs = FIT_DATA[year]['catg_alloc_fits']
            logging.info(f"Year {year}: cum_alloc_fits - slope={cum_alloc_coeffs[0]:.4f}, intercept={cum_alloc_coeffs[1]:.4f}")
            logging.info(f"Year {year}: catg_alloc_fits - slope={catg_alloc_coeffs[0]:.4f}, intercept={catg_alloc_coeffs[1]:.4f}")

            
    if 1: #Generate compute samples

        all_years=np.concatenate([Config.fit_years, Config.pred_years.astype(int).ravel()])
        COMPUTE_SAMPLE_DATA = {sim: {int(year): {} for year in all_years} for sim in range(Config.n_simulations)} #init data structure 

        for sim in range(Config.n_simulations):
            #print(f"Sim {sim} of {Config.n_simulations}")

            for year in all_years:
                #get total compute
                if year in Config.fit_years:
                    log_agg_training_compute = LOG_AGGREGATE_COMPUTE_DATA[sim]["historical aggregate training compute"][year]
                if year in Config.pred_years:
                    log_agg_training_compute = LOG_AGGREGATE_COMPUTE_DATA[sim]["aggregate training compute"][year]
                agg_training_compute = 10**log_agg_training_compute 


                #generate compute bin allocations (catg_alloc)
                VALID_ALLOCATION=False
                while not VALID_ALLOCATION:

                    #LMS sampling
                    if year in Config.fit_years: #uniform sampling for historical  data 
                        norm_largest_model = np.random.uniform(Config.min_lms, Config.max_lms)
                    else: 
                        if year==2024 and Config.SET_2024_LMS:
                            gpt_4o_size = 3.80*10**25
                            norm_largest_model = gpt_4o_size/agg_training_compute
                        else: 
                            if Config.LMS_SAMPLING=='uniform':
                                norm_largest_model = np.random.uniform(Config.min_lms, Config.max_lms)
                            elif Config.LMS_SAMPLING=='log_normal':
                                log_mean = np.mean([np.log(Config.min_lms), np.log(Config.max_lms)])
                                log_stdev = np.abs((np.log(Config.max_lms) - np.log(Config.min_lms))/4)
                                norm_largest_model = np.random.lognormal(mean=log_mean, sigma=log_stdev) #initial sample
                                while norm_largest_model < Config.min_lms or norm_largest_model > Config.max_lms: #resample until within bounds
                                    norm_largest_model = np.random.lognormal(mean=log_mean, sigma=log_stdev)


                    largest_model = norm_largest_model * agg_training_compute
                    #assert largest_model <= 0.5*agg_training_compute, print(f"Year: {year}, Largest Model: {largest_model}, Total Training Compute: {agg_training_compute}")

                    #sample smallest model that year
                    min_norm_m = 10**(np.random.uniform(np.log10(Config.min_norm_m_min),np.log10(Config.min_norm_m_max)))

                    # model sizes (as fraction of largest_model)
                    norm_ms = np.logspace(np.log10(min_norm_m), np.log10(1.0), num=Config.n_catgs)
                    log_norm_ms = np.log10(norm_ms)

                    assert Config.ALLOC_FIT_TYPE in ['cumulative','categorical']

                    if Config.POINT_CUM_ALLOC_PARAMS:
                        grad_cum_alloc = np.mean([FIT_DATA[year]['cum_alloc_fits'][0] for year in FIT_DATA.keys()])
                        int_cum_alloc = np.mean([FIT_DATA[year]['cum_alloc_fits'][1] for year in FIT_DATA.keys()])
                    elif Config.DISTRIBUTION_CUM_ALLOC_PARAMS:
                        grad_cum_alloc, int_cum_alloc = np.random.uniform(Config.grad_cum_alloc_min,Config.grad_cum_alloc_max), 0
                    else:
                        raise ValueError("Invalid choice of cumulative alloc params")

                    log_cum_alloc = grad_cum_alloc*log_norm_ms + int_cum_alloc
                    cum_alloc = 10**log_cum_alloc
                    catg_alloc = np.diff(cum_alloc)
                    sum_condition = abs(np.sum(catg_alloc) - 1) < 1e-2
                    assert sum_condition, f"Sum of category allocations {np.sum(catg_alloc)} not equal to 1" #stop code if not equal to 1

                    residual_catg_alloc = 1-np.sum(catg_alloc)
                    catg_alloc = np.concatenate(([residual_catg_alloc],catg_alloc))

                    absl_ms = norm_ms*largest_model
                    absl_model_catgs = [(absl_ms[i], absl_ms[i+1]) for i in range(len(absl_ms) - 1)]
                    absl_model_catgs = [(min_norm_m*largest_model,min_norm_m*largest_model)] + absl_model_catgs
                    absl_allocs = catg_alloc * agg_training_compute
                    alloc_ub_check_var = [ctg[-1] for ctg in absl_model_catgs] < absl_allocs
                    alloc_ub_condition = np.all(alloc_ub_check_var) #ensure all allocs are above the ctg_ub

                    if alloc_ub_condition: 
                        VALID_ALLOCATION=True
                    else:
                        VALID_ALLOCATION=False


                model_ctgs = [(norm_ms[i], norm_ms[i+1]) for i in range(len(norm_ms) - 1)]
                model_ctgs = [(min_norm_m,min_norm_m)] + model_ctgs #for first ctg bin - we sample just the smallest model
                ctgs_lbs, ctgs_ubs =[ctg[0] for ctg in model_ctgs], [ctg[-1] for ctg in model_ctgs] #useful vars

                bin_compute_allocs = catg_alloc * agg_training_compute  # array of how much compute allocated to each bin

                compute_samples_rand = []

                #draw samples
                for idx, (ctg, alloc) in enumerate(list(zip(model_ctgs, bin_compute_allocs))):
                    if idx==0: continue
                    

                    # set initial bounds
                    bounds = ctg
                    norm_model_bin_lb, norm_model_bin_ub = float(bounds[0]), float(bounds[1])
                    model_bin_lb, model_bin_ub = largest_model * norm_model_bin_lb, largest_model * norm_model_bin_ub  # normalising factor is total training compute
                    assert alloc > model_bin_ub
                    if alloc==0:  # skip bins which have no compute allocated to them - occurs when allocation gradient large 
                        continue 

                    
                    #perform sampling 
                    allocnorm_model_bin_lb, allocnorm_model_bin_ub = model_bin_lb / alloc, model_bin_ub / alloc  # this is purely just for sampling; no physical meaning
                    running_tot = 0
                    allocnormed_samples = []
                    while running_tot < 1:
                        # SAMPLE
                        sample = np.random.uniform(allocnorm_model_bin_lb, allocnorm_model_bin_ub)
                        sample = float(sample) if isinstance(sample, np.ndarray) else sample
                        assert sample <= 1 #sample should be smaller than alloc OR equal to it


                        # SUM CHECK
                        if running_tot + sample > 1:
                            allocnormed_samples.append(1 - running_tot)
                            running_tot = 1
                        else:
                            allocnormed_samples.append(sample)
                            running_tot += sample

                    bin_samples = alloc*np.array(allocnormed_samples) # un-normalise
                    compute_samples_rand = compute_samples_rand + (list(bin_samples)) #add to sample list


                compute_samples_rand = [x for x in compute_samples_rand if x != 0]

                COMPUTE_SAMPLE_DATA[sim][year]['samples'] = compute_samples_rand
                COMPUTE_SAMPLE_DATA[sim][year]['date'] = [decimal_year_to_date(year + np.random.random()) for _ in compute_samples_rand]  # convert to standard pd datetime format
                COMPUTE_SAMPLE_DATA[sim][year]['largest model'] = largest_model


        logging.debug("\nNumber of samples per year:")
        for year in Config.pred_years.ravel():
            logging.debug(f"{year}: {len(COMPUTE_SAMPLE_DATA[0][year]['samples'])} samples") #take first sim


        if Config.GENERATED_SAMPLE_PLOTS:
            fig, axes = plt.subplots(3, 2, figsize=(12, 8))
            axes = axes.ravel()

            for idx, year in enumerate(Config.pred_years):
                all_samples = [np.log10(COMPUTE_SAMPLE_DATA[sim][year]['samples']) for sim in range(Config.n_simulations)]
                
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

        if Config.GENERATED_SAMPLE_PLOTS:
            ylims = (14, 30)
            plt.figure(figsize=(12, 6))
            
            for year in all_years:
                if year in Config.fit_years:
                    sample_data = COMPUTE_SAMPLE_DATA[0][year]
                    plt.scatter(sample_data['date'], np.log10(sample_data['samples']), alpha=0.2, color='blue', label='Retrodicted Samples' if year == Config.fit_years[0] else "",marker='x')
                if year in Config.pred_years:
                    sample_data = COMPUTE_SAMPLE_DATA[0][year]
                    plt.scatter(sample_data['date'], np.log10(sample_data['samples']), alpha=0.5, label='Projected Samples' if year == Config.pred_years[0] else "", color='red')

            plt.scatter(df[df['year'].isin(Config.fit_years)]['date'], np.log10(df[df['year'].isin(Config.fit_years)]['compute']), alpha=1.0, label='Historical',marker='x')

            plt.xlabel('Year')
            plt.ylabel('Log Compute (FLOPs)')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.ylim(ylims)
            plt.yticks(np.arange(ylims[0], ylims[1], 0.5))
            plt.show()


    if 1: # verification retrodiction

        #backtesting the absolute Config.thresholds

        retrodict_years=Config.fit_years

        #observed
        # Create DataFrame from observed counts
        print('Computing retrodicted absolute counts')
        df_observed = pd.DataFrame.from_dict({threshold: {year: sum(df[df['year'] == year]['compute'] > 10**threshold)
                                                        for year in retrodict_years}
                                            for threshold in Config.retrodict_thresholds}, 
                                            orient='index')
        df_observed.index = [f'{10**threshold:.2e}' for threshold in Config.retrodict_thresholds]
        df_observed.index.name = 'Threshold'

        # Create retrodict counts dictionary
        retrodict_counts = {year: {threshold: [] for threshold in Config.retrodict_thresholds} for year in retrodict_years}

        for sim, sim_data in COMPUTE_SAMPLE_DATA.items():
            for year, year_data in sim_data.items():
                if year in retrodict_years:
                    for threshold in Config.retrodict_thresholds:
                        count = (sum(x >= 10**threshold for x in year_data['samples'])).astype(int)
                        retrodict_counts[year][threshold].append(count)

        # Calculate percentiles for each year and threshold

        retrodict_percentile_counts = {year: {percentile: [] for percentile in Config.CI_percentiles} for year in retrodict_years}
        for year in retrodict_years:
            for threshold in Config.retrodict_thresholds:
                for percentile in Config.CI_percentiles:
                    percentile_count = (np.percentile(retrodict_counts[year][threshold], percentile)).astype(int)
                    retrodict_percentile_counts[year][percentile].append(percentile_count)

        dfs_retrodict = {}
        for percentile in Config.CI_percentiles:
            dfs_retrodict[percentile] = pd.DataFrame(
                {year: retrodict_percentile_counts[year][percentile] for year in retrodict_years},
                index=[f'{10**t:.2e}' for t in Config.retrodict_thresholds]
            )
            dfs_retrodict[percentile].index.name = 'Threshold'

        # Take cumulative sum across years for both dataframes
        df_observed_cumulative = df_observed.cumsum(axis=1)
        dfs_retrodict_cumulative = {percentile: df.cumsum(axis=1) for percentile, df in dfs_retrodict.items()}


        # Create dataframe with observed and retrodicted values for each percentile
        combined_df = pd.DataFrame(index=df_observed_cumulative.index)

        for year in df_observed_cumulative.columns:
            combined_df[year] = [f"{obs} ({','.join(str(x) for x in ret)})" for obs, ret in zip(
                df_observed_cumulative[year],
                zip(*[dfs_retrodict_cumulative[percentile][year] for percentile in Config.CI_percentiles])
            )]

        # Calculate the difference between observed and retrodicted values for each percentile
        difference_df = pd.DataFrame(index=df_observed_cumulative.index)

        for year in df_observed_cumulative.columns:
            differences = []
            for obs, *rets in zip(df_observed_cumulative[year], 
                                *[dfs_retrodict_cumulative[percentile][year] for percentile in Config.CI_percentiles]):
                differences.append(f"({obs-rets[0]}, {obs-rets[1]}, {obs-rets[2]})")
            difference_df[year] = differences



        absolute_threshold_retrodicted = combined_df
        absolute_threshold_retrodicted_difference = difference_df


        if Config.COMPUTE_FRONTIER_COUNTS:
            print('Computing retrodicted frontier counts')
            # Group data into specified periods
            df['period'] = round_dates(df['date'], Config.period_freq)
            df['log_compute'] = np.log10(df['compute'])

            frontier_counts = {}

            for year in Config.fit_years:
                year_filtered_df = df[df['date'].dt.year == year]
                frontier_counts[year] = {}
                for width in Config.threshold_widths:
                    width_year_counts = 0
                    for idx, period in enumerate(sorted(year_filtered_df['period'].unique())):
                        largest_model = df[df['period'] < period]['compute'].max()  # get largest model before this period
                        period_data = df[df.period == period]
                        within_threshold_condition = ((np.log10(largest_model) - np.log10(period_data['compute'])) <= width) & ((np.log10(largest_model) - np.log10(period_data['compute'])) > 0)
                        above_frontier_condition = period_data['compute'] > largest_model
                        count = within_threshold_condition.sum() + above_frontier_condition.sum()
                        width_year_counts += count
                    frontier_counts[year][width] = width_year_counts

            # Calculate frontier counts for each percentile
            # Process each simulation

            sample_frontier_counts = {year: {width: [] for width in Config.threshold_widths} for year in Config.fit_years}


            for sim, sim_data in COMPUTE_SAMPLE_DATA.items():
                
                # Pre-compute dates and periods for all years to avoid repeated conversions
                for year, year_data in sim_data.items():
                    year_data['period'] = round_dates(pd.to_datetime(year_data['date']), Config.period_freq)
                    year_data['log_compute'] = np.log10(year_data['samples'])
                
                # Pre-compute largest models for each period across all years
                all_periods = sorted(set(period for data in sim_data.values() for period in data['period'].unique()))
                largest_models = {}
                all_samples = np.concatenate([np.array(data['samples']) for data in sim_data.values()])
                all_dates = np.concatenate([np.array(data['date']) for data in sim_data.values()])
                for period in all_periods:
                    largest_models[period] = np.max(all_samples[all_dates < period])
                
                # Process each year
                for year in Config.fit_years:
                    year_data = sim_data[year]
                    year_samples = np.array(year_data['samples'])
                    year_periods = year_data['period'].unique()
                    
                    # Process each threshold width
                    for width in Config.threshold_widths:
                        width_year_counts = 0
                        
                        # Vectorized operations for each period
                        for period in sorted(year_periods):
                            period_mask = year_data['period'] == period
                            period_samples = year_samples[period_mask]
                            largest_model = largest_models[period]
                            
                            # Combine conditions in single vectorized operation
                            log_ratio = np.log10(largest_model) - np.log10(period_samples)
                            counts = np.sum((log_ratio <= width) & (log_ratio > 0)) + np.sum(period_samples > largest_model)
                            width_year_counts += counts
                            
                        sample_frontier_counts[year][width].append(width_year_counts)

                    
            # Calculate percentile counts for each year and width
            percentile_frontier_counts = {year: {width: {percentile: [] for percentile in Config.CI_percentiles} for width in Config.threshold_widths} for year in Config.fit_years}
            for year in Config.fit_years:
                for width in Config.threshold_widths:
                    for percentile in Config.CI_percentiles:
                        percentile_count = (np.percentile(sample_frontier_counts[year][width], percentile)).astype(int)
                        percentile_frontier_counts[year][width][percentile] = percentile_count

            # Create combined dataframe with observed and retrodicted values
            combined_df = pd.DataFrame(index=Config.threshold_widths)
            combined_df.index.name = 'width'

            for year in Config.fit_years:
                combined_df[year] = [f"{frontier_counts[year][width]} ({','.join(str(percentile_frontier_counts[year][width][p]) for p in Config.CI_percentiles)})" for width in Config.threshold_widths]

            # Calculate differences between observed and retrodicted values
            difference_df = pd.DataFrame(index=Config.threshold_widths)
            difference_df.index.name = 'width'

            for year in Config.fit_years:
                differences = []
                for width in Config.threshold_widths:
                    obs = frontier_counts[year][width]
                    rets = [percentile_frontier_counts[year][width][p] for p in Config.CI_percentiles]
                    differences.append(f"({obs-rets[0]}, {obs-rets[1]}, {obs-rets[2]})")
                difference_df[year] = differences



            frontier_threshold_retrodicted = combined_df
            frontier_threshold_retrodicted_difference = difference_df


    if 1: # predictions

        #predictions for absolute Config.thresholds
        ## regular counts
        print('Computing predicted absolute counts')
        threshold_counts_all_simulations = {year: {threshold: [] for threshold in Config.thresholds} for year in Config.pred_years.astype(int).ravel()}

        # Iterate over each simulation
        for sim in range(len(COMPUTE_SAMPLE_DATA)):
            for year, samples in COMPUTE_SAMPLE_DATA[sim].items():
                if year in Config.pred_years:
                    for threshold in Config.thresholds:
                        count = sum(x >= 10**threshold for x in samples['samples'])
                        threshold_counts_all_simulations[year][threshold].append(count)

        # Calculate counts for each percentile in Config.CI_percentiles
        threshold_counts_summary = {year: [] for year in Config.pred_years.astype(int).ravel()}
        for year in Config.pred_years.astype(int).ravel():
            for threshold in Config.thresholds:
                counts = threshold_counts_all_simulations[year][threshold]
                percentile_counts = [np.percentile(counts, p) for p in Config.CI_percentiles]
                threshold_counts_summary[year].append(f"{percentile_counts[1]:.0f} ({percentile_counts[0]:.0f}-{percentile_counts[2]:.0f})")

        # Create DataFrames for each percentile
        percentile_dfs = {}
        for percentile in Config.CI_percentiles:
            percentile_dfs[percentile] = pd.DataFrame(
                {year: [int(round(np.percentile(threshold_counts_all_simulations[year][threshold], percentile))) 
                        for threshold in Config.thresholds] 
                for year in Config.pred_years.astype(int).ravel()},
                index=[f'>1e{t}' for t in Config.thresholds]
            )

        if Config.SET_2024_COUNTS:
            for percentile in percentile_dfs:
                percentile_dfs[percentile].loc['>1e25', 2024] = 20 #set to 20 models released over 1e25 in 2024
                percentile_dfs[percentile].loc['>1e26', 2024] = 0 #0 models released in 2025 over 1e26
        
        # Make cumulative across years
        percentile_dfs_cumulative = {
            percentile: df.cumsum(axis=1) 
            for percentile, df in percentile_dfs.items()
        }


        for percentile in percentile_dfs_cumulative:
            percentile_dfs_cumulative[percentile].loc['>1e25'] = percentile_dfs_cumulative[percentile].loc['>1e25'] + 4

        

        

        # Combine into a single DataFrame
        df_combined_cumulative = pd.DataFrame()
        for year in percentile_dfs_cumulative[50].columns:
            for idx in percentile_dfs_cumulative[50].index:
                values = [str(percentile_dfs_cumulative[p].loc[idx, year]) for p in Config.CI_percentiles]
                df_combined_cumulative.loc[idx, year] = f"[{', '.join(values)}]"

        absolute_threshold_predicted = df_combined_cumulative


        #period_data = pd.date_range(start='2024-01-01', end='2029-01-01', freq=Config.period_freq).strftime('%Y-%m-%d %H:%M:%S').tolist()
        if Config.COMPUTE_FRONTIER_COUNTS:
            print('Computing predicted frontier counts')
            frontier_counts_all_simulations = {year: {width: [] for width in Config.threshold_widths} for year in Config.pred_years}

            # Process each simulation
            for sim in range(len(COMPUTE_SAMPLE_DATA)):
                sim_data = COMPUTE_SAMPLE_DATA[sim]
                
                # Pre-compute dates and periods for all years
                for year, year_data in sim_data.items():
                    year_data['period'] = round_dates(pd.to_datetime(year_data['date']), Config.period_freq)
                    year_data['log_compute'] = np.log10(year_data['samples'])
                
                # Pre-compute largest models for each period across all years
                all_periods = sorted(set(period for data in sim_data.values() for period in data['period'].unique()))
                largest_models = {}
                all_samples = np.concatenate([np.array(data['samples']) for data in sim_data.values()])
                all_dates = np.concatenate([np.array(data['date']) for data in sim_data.values()])
                for period in all_periods:
                    largest_models[period] = np.max(all_samples[all_dates < period])
                
                # Process each year
                for year in Config.pred_years:
                    year_data = sim_data[year]
                    year_samples = np.array(year_data['samples'])
                    year_periods = year_data['period'].unique()
                    
                    # Process each threshold width
                    for width in Config.threshold_widths:
                        width_year_counts = 0
                        
                        # Vectorized operations for each period
                        for period in sorted(year_periods):
                            period_mask = year_data['period'] == period
                            period_samples = year_samples[period_mask]
                            largest_model = largest_models[period]
                            
                            # Combine conditions in single vectorized operation
                            log_ratio = np.log10(largest_model) - np.log10(period_samples)
                            counts = np.sum((log_ratio <= width) & (log_ratio > 0)) + np.sum(period_samples > largest_model)
                            width_year_counts += counts
                            
                        frontier_counts_all_simulations[year][width].append(width_year_counts)


            # Create DataFrames for each percentile
            frontier_percentile_dfs = {}
            for percentile in Config.CI_percentiles:
                frontier_percentile_dfs[percentile] = pd.DataFrame(
                    {year: [int(round(np.percentile(frontier_counts_all_simulations[year][width], percentile)))
                            for width in Config.threshold_widths]
                    for year in Config.pred_years},
                    index=[f'Within {width} OOM' for width in Config.threshold_widths]
                )

            # Combine into a single DataFrame
            df_frontier_combined = pd.DataFrame()
            for year in frontier_percentile_dfs[50].columns:
                for idx in frontier_percentile_dfs[50].index:
                    values = [str(frontier_percentile_dfs[p].loc[idx, year]) for p in Config.CI_percentiles]
                    df_frontier_combined.loc[idx, year] = f"[{', '.join(values)}]"

            frontier_threshold_predicted = df_frontier_combined


    if 1: #display and save

        if not Config.SAVE_RESULTS:

            logging.info("Displaying results...\n")
            logging.info("=== Retrodicted Config.thresholds ===")
            logging.info("=== Absolute Threshold Retrodicted ===")
            display(absolute_threshold_retrodicted)
            logging.info("=== Predicted Config.thresholds ===")
            logging.info("=== Absolute Threshold Predicted ===")
            display(absolute_threshold_predicted)

            if Config.COMPUTE_FRONTIER_COUNTS:
                logging.info("=== Frontier Threshold Retrodicted ===")
                display(frontier_threshold_retrodicted)
                logging.info("=== Frontier Threshold Predicted ===")
                display(frontier_threshold_predicted)
        

        if Config.SAVE_RESULTS:

            #assert Config.COMPUTE_FRONTIER_COUNTS, "Frontier counts must be computed to save results"
            
            # Create results directory if it doesn't exist
            if not os.path.exists(Config.save_folder):
                os.makedirs(Config.save_folder)

            # Get current date
            time.sleep(1) #just to get different file names
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            filename = f"{Config.name}-{datetime.now().strftime('%H-%M-%S')}"

            # Save tables to results file
            with open(f'{Config.save_folder}/{filename}.csv', 'w') as f:
                for key, value in SAVE_Config.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                f.write("Absolute Threshold Retrodicted:\n")
                absolute_threshold_retrodicted.to_csv(f,sep='\t')
                f.write('\n\n')
                f.write("Absolute Threshold Predicted:\n")
                absolute_threshold_predicted.to_csv(f,sep='\t')
                f.write('\n\n')
                if Config.COMPUTE_FRONTIER_COUNTS:
                    f.write("Frontier Threshold Retrodicted:\n")
                    frontier_threshold_retrodicted.to_csv(f,sep='\t')
                    f.write('\n\n')
                    f.write("Frontier Threshold Predicted:\n")
                    frontier_threshold_predicted.to_csv(f,sep='\t')

        if Config.WANDB_LOGGING:
            wandb.init(project=Config.wandb_project)
            wandb.log({"absolute_threshold_retrodicted":absolute_threshold_retrodicted})
            wandb.log({"absolute_threshold_predicted":absolute_threshold_predicted})
            if Config.COMPUTE_FRONTIER_COUNTS:
                wandb.log({"frontier_threshold_retrodicted":frontier_threshold_retrodicted})
                wandb.log({"frontier_threshold_predicted":frontier_threshold_predicted})
            wandb.finish()


#main(Config)