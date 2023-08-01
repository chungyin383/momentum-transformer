import os
from mom_trans.backtest import run_classical_methods
from settings.default import TICKERS

INTERVALS = [(1990, y, y + 1) for y in range(2016, 2022)]
# INTERVALS = [(1990, y, y + 1) for y in range(2021, 2023)]

REFERENCE_EXPERIMENT = "experiment_quandl_100assets_tft_cpnone_len252_notime_div_v1"
# REFERENCE_EXPERIMENT = "experiment_crypto_5assets_tft_cpnone_len365_notime_div_crypto_v1"

ASSET_CLASS_MAPPING = dict(zip(TICKERS, ["COMB"] * len(TICKERS)))

features_file_path = os.path.join(
    "data",
    "cpd_nonelbw.csv",
)

run_classical_methods(features_file_path, INTERVALS, REFERENCE_EXPERIMENT, ASSET_CLASS_MAPPING, crypto=True)


'''
############################ LIST OF AMENDMENTS ##########################################

Lines 3-18:
    - generalize for crypto and non-crypto experiments

##########################################################################################
'''