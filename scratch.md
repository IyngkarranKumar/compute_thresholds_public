- To do
  - Rolling window grouping, instead of naive year-to-year grouping
  - bootstrapping groups for summary statistic estimation
  - Figuring out how to inject *skew* into distribution


- Continuosuuly estimate model parameters
  - I don't really like the 'single year' estimation that we're doing at the moment.
  - I'd rather to smth like: rolling window, parameter estimation, etc.
    - Problem: What about when we're doing one-year window around start of 2017?
    - Solution - start at place where 1 year distributions are well defined (midway through 2017 for us, through to midway 2023)
    - 

- Could we *derive* skewness shape?
  - could do best fit with historial distributions. Would want to do a model selection, BIC, other


- Skew modelling:
  - A distribution that is becoming increasingly left skewed (tail on the left) implies that for a *fixed* compute budget, we're deciding to spend budget on *fewer, larger* models, rather than *more, smaller* models, or keeping distribution same
    - limiting case: for fixed compute budget of approx. 1e28 FLOP
      - In year 1: we mostly train 1000 1e25s, and spread remainder over other bins
      - In year 2: We mostly train 100 1e26s, and spread remainder over other bins
      - In year 3: We mostly train 10 1e27s, and spread remainder over other bins
  - Again, we can do a *normal distribution* trajectory first. We then want to compare to an increasingly skewed distribution. But how do we model that?
    - Again, start with fixed compute spending. Initially we allocate this according to normal. But then things start moving

  - Stick with all prob mass in 3 bins either side of mean compute. Fix mean compute also.
    - then skewing looks like re-distributing prob mass.
    - 

- pdfs
  - Continuous extensions of pmfs

- Gaussian mixture
  - Idea being that there is the 'foundation model' distribution, and the 'large scale' distribution coming along in later years
  - Clustering algorithm - allocates each data point to a particular gaussian
    - I.e: for a gmm(n=2), data point assumed to be generated from one or other gaussian
  - Could do a GMM best fit
  - Again, idea that data is generated from a mixture of n normal distributions
    - low n mixture is a candidate when there are a few peaks in the distributions
  - For our case, I could see our current trend as being a mix of the 'large scale' model distribution and the 'other foundation model' distribution

- I would say don't worry tremendously about data fit *yet* - just run predictions, see what we get, and interpret after. So modularise the whole code.
  - Extrapolating these for 2024, 2025, will also get interesting. But just choose smth sensible and run with it.

- Model number count?


- Summary so far:
  - A mixture of gaussians might be a better fit to the data
    - This could be a *complementary trajectory* to the pure normal distribution fit
    - How did we not land on this before? ChatGPT gave us this idea, note
  - On the fit so far
    - We'll have to extrapolate gmm trajectories. Right now the ordering seems a bit off on the trajectories (swapping reds and blues)
  - I don't really care how good the fit is - I just want to be modularise the code and run many trajectories
    - Still need to look at models counts


- How to best modularise the code?
  - Could but this into class methods - where does dataset processing come in then?
  - Advantage of class methods over pure functional approach is that I get state variables
  - How does Epoch do it?
    - Pretty complex imo. Could just pass *filtered df* to class instance
    - not keen to do this in one workflow and have a *variable mess*
    - Let's try a class method. 

- Model counts --> could fit a *kinked linear* to the data
- Data class --> store model config


- To do
  - Perform bucketing and visualise distributions for various window freqs, etc. For standard *year* we should obtain recognistable distributions 
  - Extrapolaate normal dist params
  - Perform analysis (extrap, counting)
  - Implement backtest verif checks


- for each year in retr_years
  - predict means, model counts
  - get std from real data - why not
  - generate log compute array and populate df

- What to do after retrodiction?
  - Run *all fits* and see what we get

- Fits to implement
  - Model counts
    - Exponential
    - Kinked linear (?)
  - Distributions
    - gaussian mixture (2)
  
  
- What to do after *all* fits are run?
  - Run with rollouts/integrate CIs into fits

- 