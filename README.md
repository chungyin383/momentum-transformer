# HKU MSc(CompSc) Project – Trading with the Momentum Transformer
## About
This is the codebase for the captioned project. We aim to improve the performance of the TFT model mentioned in the paper [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/pdf/2112.08534.pdf). The codes are based on [Github](https://github.com/kieranjwood/trading-momentum-transformer) of the reference paper. **At the END of each python file in this Github repository, we documented the parts that were edited by us.**

## Requirements
1. Create a virtual environment: `python -m venv momentum-transformer`

1. Install Python version 3.7
1. Install packages: `pip install -r requirements.txt`

## Using the code

### Downloading raw data
1. If you want to download Quandl data, create a Nasdaq Data Link account to access the [free Quandl dataset](https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation). This dataset provides continuous contracts for 600+ futures, built on top of raw data from CME, ICE, LIFFE etc.
1. Change the list `ALL_QUANDL_CODES` for Quandl, or `TICKERS` for cryptocurrency data for  Coingecko. For Quandl, we use the 100 futures tickers which have i) the longest history ii) more than 90% of trading days have data iii) data up until at least Dec 2021. For Coingecko, we use 100 tickers with the highest market cap as of 1st Jan 2018. 
1. Download the Quandl data with: `python -m data.download_quandl_data <<API_KEY>>`, or download the Coingecko data with: `python -m data.download_coingecko_data`.

### Create CPD features (Optional)
1. Run the changepoint detection module: `python -m examples.concurent_cpd <<CPD_WINDOW_LENGTH>>`, for example `python -m examples.concurent_cpd 21` and `python -m examples.concurent_cpd 126`. For cryptocurrency, use 30 and 183 instead.
1. If `concurent_cpd` does not work, try `python -m examples.sequential_cpd <<CPD_WINDOW_LENGTH>>`.

### Create input featues (without CPD module)
1. Create Momentum Transformer input features with: `python -m examples.create_features`. The default settings would prepare a csv file with normalized returns and MACD. The following arguments can be added in the command:
- `--rsi`: whether to add RSI as additional input features
- `--kd`: whether to add stochastic oscillator as additional input features
- `--volume`: whether to add volume and VWAP as additional input features (only available for cryptocurrency)
- `--categorical`: whether to convert MACD, RSI, stochastic oscillator values into signal labels
- `--crypto`: whether your raw data is cryptocurrency data

### Create input featues (with CPD module)

1. Create Momentum Transformer input features including CPD module features with: `python -m examples.create_features 21` after the changepoint detection module has completed. (or 30 for cryptocurrency data)
1. To create a features file with multiple changepoint detection lookback windows: `python -m examples.create_features 126 21` after the 126 (183 for crypto) day LBW changepoint detection module has completed and a features file for the 21 (30 for crypto) day LBW exists.
1. Additional flags can be added when calling the python module, same as those described in the previous section.

### Run experiment

1. Run one of the experiments with `python -m examples.run_dmn_experiment <<EXPERIMENT_NAME>>`. Optional arguments include:-
- `--rsi`: whether to add RSI as additional input features
- `--kd`: whether to add stochastic oscillator as additional input features
- `--volume`: whether to add volume and VWAP as additional input features (only available for cryptocurrency)
- `--categorical`: whether to convert MACD, RSI, stochastic oscillator values into signal labels
- `--crypto`: whether your raw data is cryptocurrency data
- `--GLU_Variant` which GLU variant to use (Bilinear / ReGLU / GEGLU / SwiGLU), default is GLU. e.g. `--GLU_Variant "Bilinear"`
- `--train_start`: indicate the starting year of training data, e.g. `--train_start 2018`
- `--test_start`: indicate the starting year of testing data, e.g. `--test_start 2021`
- `--test_end`: indicate the ending year of testing data, e.g. `--test_end 2023`
- `--num_repeats`: indicate the number of repeats that the experiment would be conducted, e.g. `--num_repeats 5`

### Run classical strategies (Long only / TSMOM)
This script is for generating the results of Long only and TSMOM for benchmarking.
1. Create features file for the reference experiment (See section *Create input featues*)
2. Change necessary variables in `run_classical_strategies.py` and execute the code with `python -m examples.run_classical_strategies`.

### Output variable importance
This script is for generating variable importance for the input features of a model. This script can only be run after the model has been built and stored under the folder `results`.
1. Change the necessary parameters in `variable_importance.py`.
2. Execute the code with `python -m examples.variable_importance`. An csv file will be generated in for each experiment in the `results` folder.

### Plot returns graph
This script is for simulating the change of fund balance given an initial fund of $100. Multiple experiments can be plotted on the same graph. This script can only be run after the models have been built and stored under the folder `results`.
1. Change the necessary parameters in `plot_returns_graph.py`.
2. Execute the code with `python -m examples.plot_returns_graph`. A graph will be displayed.

### Plot activation functions
This script is for visualizing the difference between activation functions.
1. Execute the code with `python -m examples.plot_activation_functions`. A graph will be displayed.

## Experiment Results
The results and models of each trial of each experiment in our report has been uploaded under the folder `results`.