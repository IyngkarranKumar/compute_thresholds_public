- experimenting as alternating between messy dev and refactoring, cleaning, jettisoning experiments that don't work
  - we probably want to save copies of messy dev (pre-clean) and post-clean

- to do
  - get mwp incl. verification set up for threshold
    - can incl. just editing analysis_2 file
  - want clear way to modularise different scenarios

- what does my ideal experimental workflow look like here?
  - every time I press enter, I get a new table out?
  - yes - where is my table of results that's being updated every run?
  
- should have an execution cell and editing cell. Execution cell can be small

- figuring out how to use git for debugging could be useful

- i don't care if you're stuck, I just need you to iterate/write quickly
- at a given point in time, how are these data points distributed around this line?
  - line gives mean. 
  - what's the type sig? input: mean, params, n_models. Output: distribution




- the simplest thing to try here would be to do normal dist + exp counts and see what we get
  - critiques: training costs are rising.
    - to visuaise, we can plot training cost v.s time. Let's do that
- this would mean that average company is spending Meta's 2023 R&D budget on a trainnig run for the model. Few companies in the world have/will have Meta's capital to spend in the coming years. 



- TL;DR: in the near-term, we're not going to get to a point where there will be 100 companies, and the modal company will be spending Meta's annual R&D budget to train a model. This isn't sustainable.
- we won't see a scenario in the near term in which of the 100 models trained that year, all were trained with greater than facebook's R&D budget
  - and moreso, of the 100 models trained that year, 40% were trained with compute provided by the world GDP (:))b
  - model: building thinking scaffolds
  

- I can't set a concrete y=x line at which this tails off. But I could get this training compute line to tail off smoothly.
  - one thing to try: normal dist, exp counts, but growth line tailing off
    - growth line tailing off is a requirement - I can't see any reasonable modelling that keeps it at 4x per year.
    - though we're not factoring in falling FLOP/$ costs. But constant is fine for now.
  - if no tailing growth line, what else?
    - 

- 
-  stagnation point:we've been looking at the same graph with little progress for a long time
-  



- looking at this plot, assume that the models were released in same ordering of model size. Then at the start of the year, we might know
  that total FLOP will be ~1.8e26, but know that will only be neared around Dec time - i.e: when final models (biggest) are being trained.
- TO DO: Plot the cumulative vs individual for other years, and ask: "what might other scenarios look like here?". I.e: there's another scenario 
    in which around 10^22 FLOPs (individual size) we move straight to the red line, and no other models aere released that year.




  - looking at log-lin and log-log plots
    - on log-log, they are very linear
      - moving up the y-scale, each unit means that we chip away at fraction of total. If total is 1e23, the moving from 1e22, to 1e23
      - means that we've gone from 10% -> 100%. Moving from 1e21 to 1e22 is moving from 1% -> 10%. 
      - from 2020: summing all the models less than §10^19 in the plot, gets you to approx 1e-5 the total compute for that year
      - from 2023: summing all the models less than 10^22 gets you to ~1e-4 of aggregate training compute
        - we could gets parameters for these linear fits. These would impose a constraint on the distribution of models for extrap years
        - say we have n_models prediction. Would this do us in?
      - what's my question here? my question is:
        - **assuming we know the total training cost used for models in year N (this seems like the major constraint), then 
        - how do we distribute this compute acros compute space?**
      - how do we distibute this compute across compute space?
        - can we normalise the x-axis to some reference point? let's take the **mean** training compute
    - looking at the normed charts
      - 2020: if you sum up all the models less than the mean log, you are ~1e-3 away from total compute. Something similar for other years too

  - assuming you know the total training compute in a given year, and the mean log compute, how do you distribute models in log space?
    - a simpler version of this is:
      - "how many models are roughly same size as mean? How many models are roughly 1 OOM less? How many models are roughly 2 OOMs less? Etc.
    - and what's our type sig? What do we want out?
      - A scattering of points in (log) compute space. How do I get this set of points?
    - lets leave this. These plots are nice.





- how can we CLEAN_OUT()?
  - constraint to build in: 
    - moving from discrete to continuous. But this isn't a big problem I feel - leave it for now. 
    - moving from fixed k to variable k - again shouldn't be too difficult
  - mean constraint - mean of distribution should be within these bounds
  - shape of distribution over sample space [a,b]
  - what are the constraints that should be applied on the shape of the distribution over the sample space?
    - what are the categories that we can place the distribution in? I'm thinking var, skewness and kurtosis. Let's calclcate the skewness and kurtosis for some of these distributions by hand.
    - qualitatively, what are the distinctions in distributions?
      - high kurtosis - tails off quick, stretches over few OOMs from the mean. Low kurtosis - tails off slowly, stretches over few OOMs of FLOP from the mean.
      - high skew - distribution anchored firmly to left, right. mean far away from median, roughly. 
    - To do - visualise extreme cases of skew and kurtosis - then ask, what's the rationale for extreme cases of skew and kurtosis?
    - skewness = displacement of median from mean, normalised by distribution spread (std.dev)
    - kurtosis
      - I don't get the mathematical definiton yet. But for intuition:
        - consider the subset of samples in between the 3rd s.d. and the min/max of the sample (or population). How distributed is the sample? If it stretches over a broad range we have high kurtosis (fat tails) which stretch far. An exaple of a distribution with high kurtosis is stock market returns. 
        - In the compute space, high kurtosis (fat tails) means that substantial chance of seeing samples far from the mean (outliers). Low kurtosis means v.little chance of seeing samples v.far from mean.
    - Let's look at skewness first before we look at kurtosis
      - In compute distibution, what are the limiting examples of skewness
  - To cpature mean, med distinction:
    - imagine set of 10 samples - [1,2,3,4,5,6,7,8,9,10]. This has mean=med=5
    - now we have - [1,4,4,4,5,6,7,8,9,10] - this has med=5, but mean<5
      - we could even have [1,1,1,1,5,6,7,8,9,10] - this has med=5, but mean<5 still
  - Now what's the situation with a distribution that has extreme skews?
    - we could have log_compute = [25,25,25,25,25,26,27,28,29,30]. 
    - I want to paint some extreme case worlds
      - we have a plethora of developers at the lower end, that's where most 'raw models' are made. But we have a few big dogs at the high end who dominate compute spending.
      - my sense is that we're heading to a world with a high negative skew. 
      - or we could have developers skewed equally on the side of the mean compute
  - What sort of thinking scaffolding do I need here?
    - assume we have arrays of compute samples from the 2028 distribution. How much do each of these take out of the compute pie?
      - model classes - models trained with less than 1OOM of mean, less than 2OOM of mean, less than 3OOM of mean, 
    - then we can get '% of total compute' that is taken up by model classes
    - how do we categorise the individual models?
      - 'models trained on less than 1 OOM of mean' 
      - 'models trained on less than 2 OOM of mean'
      - 'models trained on less than 3 OOM of mean' 
    - mean is arbitrary. Model classes should be <1OOM of total, <2OOm total, less than 3OOM total etc. 
    - are there other ways to look at this rather than cumulative plot?
      - let's try to build that out.

- how can we get more out of the total-ind graph?
  - 2020: summing models that were 1e-6 the size of total compute gets you ~1e-5.5 of the way there. 
  - 2021: if you sum all of the models 1e-4 of total compute or less, you get 0.0001% of the way to total compute. 
  - Ideally we want to quantify this - get numbers that smmarise the information in the graph. Summary stats for the graph.
- These lines are pretty linear - we can get fits. What would these lines tell us?
- This basically says that if we 10x model compute, we chip away another OOM from aggregate. If we fit plots to this we can be more precise - increasing model size by X means that we chip away at total compute by Y.
- The rough linear shaping allows us to say that models in lower classes don't 'over contribute'
- Working backwards, if we estimate total compute in year N, I can get compute allocations to model classes. 
- But once we have that, how do we get model number? 
  - we can say something like: models within 0.01-0.1M of total contribute 10% of training compute. One option is uniform allocation - that should be our default, then we can try other allocs and see how distribution changes.
- TO DO - get linear fits for summary statistics of these C-I plots, and interpret. Then generate model counts based on them
- also do some github management
- equations are so information dense
- scaling up m by 10 means we chip away at a further 10^a of the cumulative compute.
- 
- we could find a state-space model between these parameters?
  - the compute alloc seems to be separate from the number problem



- we know that increasing model size by fctor 10 chips away at total compute by 10^A. But how are models distributed within this space [1,10] scale up?
  - problem: total compute has gone from T_1 to T_2. This T_2-T_1 difference has been filled up by models drawn from space [a,b]. How do we select models from this space [a,b] to satisfy T_2-T_1 difference
    - this is basically the problem we solved before on a smaller scale
    - (T_2-T_1)/mid(a,b)
    - Let N=T_2-T_1
      - one option: choose n_a models of size a such that n_a*a=N - 10 models of size 1
      - one option: choose n_b models of size b such that n_b*b=N - e.g: 1 model of size 10
      - one option: choose n_mid models of size mid(a,b) such that n_mid*mid(a,b)=N
      - so at most we could be out by an OOM
    - we have data from 2020-2023
    - I need more thinking scaffolding (TS) for this. What can we do?

- again we know constraints on statistics of the dataset (models within (a,b) must contribute X FLOP), and we want to map this to concrete data points.
  - **there is no unique mapping** - there are many samples that could ahve thsi distribution
    - we could generate a large number and get CIs
    - unless there is any other clear constraint?
    - what are the contraints that we're looking for? If not, and we're trying to generate a dataset from summary statistics, we might as well just random sample and get CIs
      - we could also just implement this; that would be great thinking scaffolding. So we've got an implementation task that's open.
      - distributions over normalised m space seem sensible. Let's do that.


- what question are we trying to answer here?
  - I want a sample dataset over compute space 
  - **assume** that the KDE plot gave us perfect normals 
    - then I'd simply extrapolate that out. 
      - but what if that extrapolation gave us unrealistic compute spending, as it did before?
        - well then I'd extrpolate out compute spending first 
        - so there are two ways to do this 
          - extrapolate out distributions - but then we get unrealistic compute spending 
          - extraplateu out compute spending - then apply constraints to distributions
      - Let's take the second approach 
        - extraplating compute spending: There are many compute samples for each total compute spendig. We want to clean out some of these. 
        - I could just plot a lot and filter out the ones that seem silly
      - How can we get a toy experiment here? I'd like it to be in real compute space
        - let's go discrete for now - [23,30]
        - let's get 

- float needed for making sure exp additions check out.
- for the integer case, more sparse than I assumed

- the integer case gives far fewer results than I expected. for two cases I tried, unique soln

- for sum constraint, there are **these** samples that satisfy. How do we constrain them more?
  - we need a toy experiment here. What's the setup?
  - I want the samples to be in real compute space. 
  - I don't think the sum needs to be realistic, for now. Let's try that



- how could we clean out some of these distributions, knowing that they **all** satisfy our sum rule
  - mean constraint 
  - T(m) constraint
  - I wonder what we get applying these two constraints **only**
  - importantly this puts no contraints on the **number** of models released in the future
    - this seems off - I certainly don't expect there to be 3 1e28 models released in 2028, even if
    satisfies our total and mean constraint
    - so what's the cosntraint we're missing? we should have some basic disribution shape constraint, alongside our total,mean constraint
  - a kurtosis constraint doesn't naturally appear to me; a skew one does


- one high-level approach
  - generate all samples that satisfy:
    - total compute constraint
    - T(m) distribution contraint
    - mean constraint
    - skew constraint (found from 2020-2023 dist)
  - could take the parametric approach - normal, skew normal
    - then filter all samples that don't satisfy our aggregate compute constraint/compute disribution constraint
    - i.e: parametric approach, then CLEAN_OUT operations based on aggregate compute constraints
  - can we bring in some state-space style modelling to link our mean, skew, var etc. 
    - what are the intuitive properties we should expect these to satisfy?


- intuitively, why might compute distributions skew left?
  - this means that the fraction of models within 1OOM of the total is increasing. 
  - In the limit, all models released are within 1OOM of the total compute produced that year 
    - This world could arise if it's hard for model to devs to keep up with rising costs of pre-training.
    - i.e: it's not profitable for those in the within 1OOM-2OOM band to train a model and deploy it. It's only profitable for those with 1OOM.
    - I need more TS for this idea
  

- whn is it not profitable to train?
  - training cost > revenue
    - if total compute spending is 1 trillion dollars, then we'll 
    - keep revenues of companies fixed


- i need to be doing quick iter with this - setting up smoething to test quick, get quick signal
  - 