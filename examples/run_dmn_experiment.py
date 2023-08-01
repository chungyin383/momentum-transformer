import os
import argparse
from settings.hp_grid import HP_MINIBATCH_SIZE
import pandas as pd
from settings.default import TICKERS
from settings.fixed_params import MODLE_PARAMS
from mom_trans.backtest import run_all_windows
import numpy as np
from functools import reduce

# define the asset class of each ticker here - for this example we have not done this
TEST_MODE = False
ASSET_CLASS_MAPPING = dict(zip(TICKERS, ["COMB"] * len(TICKERS)))
TRAIN_VALID_RATIO = 0.90
TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = True

def main(
    experiment: str,
    train_start: int,
    test_start: int,
    test_end: int,
    test_window_size: int,
    num_repeats: int,
    GLU_Variant: str,
    rsi: bool,
    kd: bool,
    volume: bool,
    categorical: bool,
    crypto: bool,
):

    NAME = "experiment_crypto_100assets" if crypto else "experiment_quandl_100assets" 

    if experiment == "LSTM":
        architecture = "LSTM"
        lstm_time_steps = 63
        changepoint_lbws = None
    elif experiment == "LSTM-CPD-21":
        architecture = "LSTM"
        lstm_time_steps = 63
        changepoint_lbws = [21]
    elif experiment == "LSTM-CPD-63":
        architecture = "LSTM"
        lstm_time_steps = 63
        changepoint_lbws = [63]
    elif experiment == "TFT":
        architecture = "TFT"
        lstm_time_steps = 365 if crypto else 252 
        changepoint_lbws = None
    elif experiment == "TFT-CPD-126-21":
        architecture = "TFT"
        lstm_time_steps = 365 if crypto else 252 
        changepoint_lbws = [126, 21]
    elif experiment == "TFT-SHORT":
        architecture = "TFT"
        lstm_time_steps = 91 if crypto else 63
        changepoint_lbws = None
    elif experiment == "TFT-SHORT-CPD-21":
        architecture = "TFT"
        lstm_time_steps = 63
        changepoint_lbws = [21]
    elif experiment == "TFT-SHORT-CPD-63":
        architecture = "TFT"
        lstm_time_steps = 63
        changepoint_lbws = [63]
    elif experiment == "TFT-CPD-183-30":
        architecture = "TFT"
        lstm_time_steps = 365 
        changepoint_lbws = [183, 30]
    else:
        raise BaseException("Invalid experiment.")

    versions = range(1, 1 + num_repeats) if not TEST_MODE else [1]

    experiment_prefix = (
        NAME
        + ("_TEST" if TEST_MODE else "")
        + ("" if TRAIN_VALID_RATIO == 0.90 else f"_split{int(TRAIN_VALID_RATIO * 100)}")
    )

    cp_string = (
        "none"
        if not changepoint_lbws
        else reduce(lambda x, y: str(x) + str(y), changepoint_lbws)
    )
    time_string = "time" if TIME_FEATURES else "notime"
    rsi_string = "_rsi" if rsi else ""
    kd_string = "_kd" if kd else ""
    volume_string = "_volume" if volume else ""
    cat_string = "_cat" if categorical else ""
    crypto_string = "_crypto" if crypto else ""
    glu_string = f"_{GLU_Variant}" if GLU_Variant != "GLU" else ""
    _project_name = f"{experiment_prefix}_{architecture.lower()}_cp{cp_string}_len{lstm_time_steps}" \
                    f"_{time_string}_{'div' if EVALUATE_DIVERSIFIED_VAL_SHARPE else 'val'}" \
                    f"{rsi_string}{kd_string}{volume_string}{cat_string}{glu_string}{crypto_string}"

    if FORCE_OUTPUT_SHARPE_LENGTH:
        _project_name += f"_outlen{FORCE_OUTPUT_SHARPE_LENGTH}"
    _project_name += "_v"
    for v in versions:
        PROJECT_NAME = _project_name + str(v)

        intervals = [
            (train_start, y, y + test_window_size)
            for y in range(test_start, test_end)
        ]

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

        if TEST_MODE:
            params["num_epochs"] = 1
            params["random_search_iterations"] = 2

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

        run_all_windows(
            PROJECT_NAME,
            features_file_path,
            intervals,
            params,
            changepoint_lbws,
            ASSET_CLASS_MAPPING,
            [16, 32, 64] if lstm_time_steps == 365 else [32, 64, 128] if lstm_time_steps == 252 else HP_MINIBATCH_SIZE,
            test_window_size,
        )


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run DMN experiment")
        parser.add_argument(
            "experiment",
            metavar="c",
            type=str,
            nargs="?",
            default="TFT-CPD-126-21",
            choices=[
                "LSTM",
                "LSTM-CPD-21",
                "LSTM-CPD-63",
                "TFT",
                "TFT-CPD-126-21",
                "TFT-SHORT",
                "TFT-SHORT-CPD-21",
                "TFT-SHORT-CPD-63",
                "TFT-CPD-183-30"
            ],
            help="Input folder for CPD outputs.",
        )
        parser.add_argument(
            "--train_start",
            metavar="s",
            type=int,
            nargs="?",
            default=1990,
            help="Training start year",
        )
        parser.add_argument(
            "--test_start",
            metavar="t",
            type=int,
            nargs="?",
            default=2016,
            help="Training end year and test start year.",
        )
        parser.add_argument(
            "--test_end",
            metavar="e",
            type=int,
            nargs="?",
            default=2021,
            help="Testing end year.",
        )
        parser.add_argument(
            "--test_window_size",
            metavar="w",
            type=int,
            nargs="?",
            default=1,
            help="Test window length in years.",
        )
        parser.add_argument(
            "--num_repeats",
            metavar="r",
            type=int,
            nargs="?",
            default=1,
            help="Number of experiment repeats.",
        )
        parser.add_argument(
            "--GLU_Variant",
            metavar="g",
            type=str,
            nargs="?",
            default="GLU",
            choices=[
                "GLU",
                "Bilinear",
                "ReGLU",
                "GEGLU",
                "SwiGLU",
            ],
            help="GLU Variants",
        )
        parser.add_argument(
            "--rsi", 
            action='store_true', 
            help='Whether to add RSI as additional input features'
        )
        parser.add_argument(
            "--kd", 
            action='store_true', 
            help='Whether to add stochastic oscillator as additional input features'
        )
        parser.add_argument(
            "--volume", 
            action='store_true', 
            help='Whether to add volume and VWAP as additional input features'
        )
        parser.add_argument(
            "--categorical", 
            action='store_true', 
            help='Whether to prepare indicators in categories, default is raw indicator values'
        )
        parser.add_argument(
            "--crypto", 
            action='store_true', 
            help='crypto data?'
        )

        args = parser.parse_known_args()[0]

        return (
            args.experiment,
            args.train_start,
            args.test_start,
            args.test_end,
            args.test_window_size,
            args.num_repeats,
            args.GLU_Variant,
            args.rsi,
            args.kd,
            args.volume,
            args.categorical,
            args.crypto
        )

    main(*get_args())


'''
############################ LIST OF AMENDMENTS ##########################################

Lines 5, 13, 26-35, 50, 54, 58, 68-71, 89-98, 117-122, 131, 136, 146, 
172, 177, 185, 193, 201, 209, 216-255, 266-271
    - generalize for crypto and non-crypto experiments
    - allow the addition of extra features, including RSI, KD, Volume, 
      VWAP, signal labels
Line 107, 197:
    - bug fix on the original code

##########################################################################################
'''