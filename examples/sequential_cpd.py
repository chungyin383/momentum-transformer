import argparse
import os

import mom_trans.changepoint_detection as cpd
from mom_trans.data_prep import calc_returns
from data.pull_data import pull_sample_data

from settings.default import (
    TICKERS,
    CPD_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
    USE_KM_HYP_TO_INITIALISE_KC
)


def main(lookback_window_length: int, start_date: str, end_date: str):
    if not os.path.exists(CPD_OUTPUT_FOLDER(lookback_window_length)):
        os.mkdir(CPD_OUTPUT_FOLDER(lookback_window_length))

    for ticker in TICKERS:
        data = pull_sample_data(ticker)
        data["daily_returns"] = calc_returns(data["close"])
        output_file_path = os.path.join(CPD_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")
        cpd.run_module(
            data, lookback_window_length, output_file_path, start_date, end_date, USE_KM_HYP_TO_INITIALISE_KC
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
        parser.add_argument(
            "--start_date",
            metavar="s",
            type=str,
            nargs="?",
            default="1990-01-01",
            help="Start date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "--end_date",
            metavar="e",
            type=str,
            nargs="?",
            default="2021-12-31",
            help="End date in format yyyy-mm-dd",
        )

        args = parser.parse_known_args()[0]
        return [
            args.lookback_window_length,
            args.start_date,
            args.end_date
        ]

    main(*get_args())


'''
############################ LIST OF AMENDMENTS ##########################################

This file is entirely written by our team. 

This script serves the same purpose as concurrent_cpd.py, which is in 
the original reference codebase, yet we find it unable to run. 
Therefore we write another version of the code.

##########################################################################################
'''