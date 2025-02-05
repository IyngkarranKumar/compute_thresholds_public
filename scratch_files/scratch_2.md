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
  - and moreso, of the 100 models trained that year, 40% were trained with compute provided by the world GDP (:))
  - model: building thinking scaffolds
  

- I can't set a concrete y=x line at which this tails off. But I could get this training compute line to tail off smoothly.
  - one thing to try: normal dist, exp counts, but growth line tailing off
    - growth line tailing off is a requirement - I can't see any reasonable modelling that keeps it at 4x per year.
    - though we're not factoring in falling FLOP/$ costs. But constant is fine for now.
  - if no tailing growth line, what else?
    - 

- 
-  stagnation point:we've been looking at the same graph with little progress for a long time