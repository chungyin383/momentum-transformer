import argparse
import os
import pandas as pd
import numpy as np
import json
from typing import Tuple, List, Dict

from mom_trans.deep_momentum_network import LstmDeepMomentumNetworkModel
from mom_trans.momentum_transformer import TftDeepMomentumNetworkModel
from mom_trans.model_inputs import ModelFeatures
from mom_trans.backtest import _get_directory_name
from settings.hp_grid import HP_MINIBATCH_SIZE
from settings.fixed_params import MODLE_PARAMS
from settings.default import TICKERS
from mom_trans.model_inputs import InputTypes

def var_importance_single_year(
    experiment_name: str,
    features_file_path: str,
    train_interval: Tuple[int, int, int],
    params: dict,
    changepoint_lbws: List[int],
    asset_class_dictionary: Dict[str, str] = None,
    hp_minibatch_size: List[int] = HP_MINIBATCH_SIZE,
):
    directory = _get_directory_name(experiment_name, train_interval)

    raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
    raw_data["date"] = raw_data["date"].astype("datetime64[ns]")

    model_features = ModelFeatures(
        raw_data,
        params["total_time_steps"],
        start_boundary=train_interval[0],
        test_boundary=train_interval[1],
        test_end=train_interval[2],
        changepoint_lbws=changepoint_lbws,
        split_tickers_individually=params["split_tickers_individually"],
        train_valid_ratio=params["train_valid_ratio"],
        add_ticker_as_static=(params["architecture"] == "TFT"),
        time_features=params["time_features"],
        lags=params["force_output_sharpe_length"],
        asset_class_dictionary=asset_class_dictionary,
        rsi=params["rsi"],
        kd=params["kd"],
        volume=params["volume"],
    )

    hp_directory = os.path.join(directory, "hp")
    
    if params["architecture"] == "LSTM":
        dmn = LstmDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
        )
    elif params["architecture"] == "TFT":
        dmn = TftDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
            **{
                "column_definition": model_features.get_column_definition(),
                "num_encoder_steps": 0,
                "stack_size": 1,
                "num_heads": 4,
            },
        )

    with open(os.path.join(directory, "best", "hyperparameters.json")) as user_file:
        best_hp = json.load(user_file)
    best_model = dmn.load_model(best_hp)
    best_model.load_weights(os.path.join(directory, "best", "checkpoints", "checkpoint"))

    att = dmn.get_attention(model_features.train, 32)
    var_importance = np.mean(att['historical_flags'], axis=(0,1))
    columns = [col[0] for col in model_features.get_column_definition() if col[2] == InputTypes.KNOWN_INPUT]
    
    return pd.DataFrame([var_importance], columns=columns, index=[train_interval[1]])


def var_importance_all_years(
    experiment_name: str,
    features_file_path: str,
    train_intervals: List[Tuple[int, int, int]],
    params: dict,
    changepoint_lbws: List[int],
    asset_class_dictionary=Dict[str, str],
):  
    all_results = []
    for interval in train_intervals:
        row = var_importance_single_year(
            experiment_name,
            features_file_path,
            interval,
            params,
            changepoint_lbws,
            asset_class_dictionary=asset_class_dictionary,
        )
        all_results.append(row)
    return pd.concat(all_results, axis=0)



def main(
    experiment: str,
    architecture: str,
    changepoint_lbws: List[int],
    lstm_time_steps: int,
    train_start: int,
    test_start: int,
    test_end: int,
    test_window_size: int,
    GLU_Variant: str,
    rsi: bool,
    kd: bool,
    volume: bool,
    categorical: bool,
    crypto: bool,
):

    intervals = [
        (train_start, y, y + test_window_size)
        for y in range(test_start, test_end)
    ]
    
    if changepoint_lbws:
        features_file_path = os.path.join(
            "data",
            f"cpd_{np.max(changepoint_lbws)}lbw.csv",
        )
    else:
        features_file_path = os.path.join(
            "data",
            "cpd_nonelbw.csv",
        )

    params = MODLE_PARAMS.copy()
    params["total_time_steps"] = lstm_time_steps
    params["architecture"] = architecture
    params["evaluate_diversified_val_sharpe"] = EVALUATE_DIVERSIFIED_VAL_SHARPE
    params["train_valid_ratio"] = TRAIN_VALID_RATIO
    params["time_features"] = TIME_FEATURES
    params["force_output_sharpe_length"] = FORCE_OUTPUT_SHARPE_LENGTH
    params["rsi"] = rsi
    params["kd"] = kd
    params["volume"] = volume
    params["GLU_Variant"] = GLU_Variant
    params["categorical"] = categorical
    params["crypto"] = crypto

    var_importance_all_years(
        experiment,
        features_file_path,
        intervals,
        params,
        changepoint_lbws,
        ASSET_CLASS_MAPPING,
    ).to_csv(
        os.path.join("results", experiment, "variable_importance.csv")
    )

####### Parameters to change #######
EXPERIMENTS = [
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_rsi_kd_cat_v1',
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_rsi_kd_cat_v2',
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_rsi_kd_cat_v3',
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_rsi_kd_cat_v4',
    'experiment_quandl_100assets_tft_cp12621_len252_notime_div_rsi_kd_cat_v5',
]
ARCHITECTURE = 'TFT'
CPD_LENGTHS = [126, 21]
RSI = True
KD = True
VOLUME = False
CATEGORICAL = True
CRYPTO = False
GLU_VARIANT = "GLU"
LSTM_TIME_STEPS = 365 if CRYPTO else 252
TRAIN_START = 1990
TEST_START = 2016
TEST_END = 2021
ASSET_CLASS_MAPPING = dict(zip(TICKERS, ["COMB"] * len(TICKERS)))
TRAIN_VALID_RATIO = 0.90
TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = True
###############################################

for e in EXPERIMENTS:
    main(
        experiment=e,
        architecture=ARCHITECTURE,
        changepoint_lbws=CPD_LENGTHS,
        lstm_time_steps=LSTM_TIME_STEPS,
        train_start=TRAIN_START,
        test_start=TEST_START,
        test_end=TEST_END,
        test_window_size=1,
        GLU_Variant=GLU_VARIANT,
        rsi=RSI,
        kd=KD,
        volume=VOLUME,
        categorical=CATEGORICAL,
        crypto=CRYPTO,
    )


'''
############################ LIST OF AMENDMENTS ##########################################

This file is entirely written by our team.

##########################################################################################
'''