# Trends in Frontier AI Model Count: A Forecast to 2028

Code for tracking the number of models above training compute thresholds: [Paper](https://arxiv.org/abs/2504.16138). 

## Abstract

Governments are starting to impose requirements on AI models based on how much compute was used to train them. For example, the EU AI Act imposes requirements on providers of general-purpose AI with systemic risk, which includes systems trained using greater than $10^{25}$ floating point operations (FLOP). In the United States' AI Diffusion Framework, a training compute threshold of $10^{26}$ FLOP is used to identify ''controlled models`` which face a number of requirements. We explore how many models such training compute thresholds will capture over time. We estimate that by the end of 2028, there will be between 103-306 foundation models exceeding the $10^{25}$ FLOP threshold put forward in the EU AI Act (90\% CI), and 45-148 models exceeding the $10^{26}$ FLOP threshold that defines controlled models in the AI Diffusion Framework (90\% CI). We also find that the number of models exceeding these absolute compute thresholds each year will increase superlinearly -- that is, each successive year will see more new models captured within the threshold than the year before. Thresholds that are defined with respect to the largest training run to date (for example, such that all models within one order of magnitude of the largest training run to date are captured by the threshold) see a more stable trend, with a median forecast of 14-16 models being captured by this definition annually from 2025-2028.



## Quick start

Clone the repository: 

``` bash 
git clone https://github.com/IyngkarranKumar/compute_thresholds_public 
```

Setup environment with conda and pip:

``` bash
conda create -n tracking_models python=3.9
pip install -r requirements.txt
```

To get all the scenarios presented in the paper, run:

``` bash
python3 scenarios_run.py --save_folder SAVE_FOLDER
```

Additional configurations can be added in the `scenarios_config.py` file. 


## Interactive Analysis

For detailed analysis and visualization:
1. Open `analysis.ipynb` in Jupyter
2. Explore model parameters
3. Generate visualizations for:
   - Historical distributions over training compute
   - Training compute forecasts
   - Linear fits to historical compute allocation
   - ...


## Citation

If you use this code, please cite our paper:

```
@misc{kumar2025trendsfrontieraimodel,
      title={Trends in Frontier AI Model Count: A Forecast to 2028}, 
      author={Iyngkarran Kumar and Sam Manning},
      year={2025},
      eprint={2504.16138},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2504.16138}, 
}
```