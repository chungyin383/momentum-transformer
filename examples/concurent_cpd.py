import multiprocessing
import argparse
import os

from settings.default import (
    TICKERS,
    CPD_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
)

N_WORKERS = len(TICKERS)


def main(lookback_window_length: int, start_date: str, end_date: str):
    if not os.path.exists(CPD_OUTPUT_FOLDER(lookback_window_length)):
        os.mkdir(CPD_OUTPUT_FOLDER(lookback_window_length))

    all_processes = [
        f'python -m examples.cpd "{ticker}" "{os.path.join(CPD_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "{start_date}" "{end_date}" "{lookback_window_length}"'
        for ticker in TICKERS
    ]
    process_pool = multiprocessing.Pool(processes=N_WORKERS)
    process_pool.map(os.system, all_processes)


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

Lines 6-7, 11, 14-16
    - to generalize for crypto and non-crypto data
Lines 19-20, 42-63
    - to add new arguments for flexible training start date and end date

##########################################################################################
'''
