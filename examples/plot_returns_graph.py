import pandas as pd
import os
import matplotlib.pyplot as plt
from empyrical import annual_volatility

####### Parameters to change #######
EXPERIMENTS = [
    'experiment_quandl_100assets_long_only',
    'experiment_quandl_100assets_tsmom',
    'experiment_quandl_100assets_tft_cpnone_len252_notime_div_v1',
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_v4',
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_rsi_kd_v3',
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_rsi_kd_cat_v4'
    #'experiment_crypto_100assets_long_only',
    #'experiment_crypto_100assets_tsmom',
    #'experiment_crypto_100assets_tft_cpnone_len365_notime_div_crypto_v1',
    #'experiment_crypto_100assets_tft_cp18330_len365_notime_div_crypto_v2',
    #'experiment_crypto_100assets_tft_cpnone_len365_notime_div_rsi_kd_volume_crypto_v4',
    #'experiment_crypto_100assets_tft_cp18330_len365_notime_div_rsi_kd_volume_crypto_v3',
    #'experiment_crypto_100assets_tft_cpnone_len365_notime_div_rsi_kd_volume_cat_crypto_v3',
    #'experiment_crypto_100assets_tft_cp18330_len365_notime_div_rsi_kd_volume_cat_crypto_v2',
]

LEGENDS = [
    'Long only',
    'TSMOM',
    'TFT',
    'TFT+CPD',
    'TFT+CPD (RSI+KD)',
    'TFT+CPD (RSI+KD labels)'
]
TEST_YEAR_START = 2016
TEST_YEAR_END = 2021
CRYPTO = False
ANNUALIZATION = 365 if CRYPTO else None
VOL_TARGET = 0.15
STARTING_BALANCE = 100
##########################################

for experiment in EXPERIMENTS:

    all_years_df = []

    for year in range(TEST_YEAR_START, TEST_YEAR_END):

        csv = os.path.join("results", experiment, f"{year}-{year+1}", "captured_returns_sw.csv")
        df = pd.read_csv(csv, index_col='time')
        df.index = pd.to_datetime(df.index)
        df = df[['identifier', 'captured_returns']]
        num_identifiers = len(df.dropna()["identifier"].unique())
        srs = df.dropna().groupby(level=0)["captured_returns"].sum() / num_identifiers # average return of all assets per day
        vol = annual_volatility(srs, annualization=ANNUALIZATION)
        srs = srs * VOL_TARGET / vol
        all_years_df.append(srs)

    combined_srs = pd.concat(all_years_df, axis=0)
    fund_balance = (1 + combined_srs).cumprod() * STARTING_BALANCE
    fund_balance.plot()

plt.title('Fund Balance Over Time')
plt.xlabel('Date')
plt.ylabel('Fund Balance ($)')
plt.legend(LEGENDS)
plt.show()


'''
############################ LIST OF AMENDMENTS ##########################################

This file is entirely written by our team.

##########################################################################################
'''