import argparse
import os

import mom_trans.changepoint_detection as cpd
from mom_trans.data_prep import calc_returns
from data.pull_data import pull_quandl_sample_data

from settings.default import (
    QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
    USE_KM_HYP_TO_INITIALISE_KC
)


def main(lookback_window_length: int):
    if not os.path.exists(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length)):
        os.mkdir(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length))

    for ticker in QUANDL_TICKERS:
        data = pull_quandl_sample_data(ticker)
        data["daily_returns"] = calc_returns(data["close"])
        output_file_path = os.path.join(CPD_QUANDL_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")
        cpd.run_module(
            data, lookback_window_length, output_file_path, "1990-01-01", "2021-12-31", USE_KM_HYP_TO_INITIALISE_KC
        )


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )
        return [
            parser.parse_known_args()[0].lookback_window_length,
        ]

    main(*get_args())
