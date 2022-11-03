"""performance.py
Computes the performance of the models in the test sample proposed"""

import os, sys
import argparse
import pandas as pd
import time
import subprocess
import random
import matplotlib.pyplot as plt

# Add the parent to the test path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="performance test",
        description="Test the performances and the results.",
    )
    parser.add_argument(
        "method",
        choices={"all", "follows_rnm", "follows_sbr", "citeseer_rnm", "citeseer_sbr"},
        help="Method to test.",
    )
    parser.add_argument(
        "--times",
        type=int,
        required=False,
        default=10,
        help="How many time to run.",
    )
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    args = get_args()
    df_time = pd.DataFrame()
    df = pd.DataFrame()

    if args.method in ["follows_rnm", "follows_sbr", "citeseer_rnm", "citeseer_sbr"]:
        for i in range(args.times):
            seed = 0  # random.randrange(1000)
            start = time.time()
            print(
                subprocess.check_output(
                    ["python", f"rnm/{args.method}.py", "--seed", f"{seed}"]
                )
            )
            end = time.time()
            df_time = df_time.append({"t": end - start}, ignore_index=True)

            if args.method == "follows_rnm":
                df = pd.read_csv(
                    "res_dlm_%d" % seed,
                    sep="\t",
                    names=["perc", "lr", "acc_map", "acc_nn"],
                    skiprows=1,
                )
                fig = plt.figure()
                plt.plot(1 - df["perc"], df["acc_map"], label="Accuracy of MAP")
                plt.plot(
                    1 - df["perc"], df["acc_nn"], label="Accuracy of the NN prediction"
                )
                fig.savefig(f"follows_rnm: {seed}", dpi=fig.dpi)
            elif args.method == "follows_sbr":
                df = pd.read_csv(
                    "res_lyrics_cc_%d" % seed,
                    sep="\t",
                    names=["lr", "perc", "w_rule", "acc_nn", "acc_map"],
                )
                fig = plt.figure()
                plt.plot(1 - df["perc"], df["acc_map"], label="Accuracy of MAP")
                plt.plot(
                    1 - df["perc"], df["acc_nn"], label="Accuracy of the NN prediction"
                )
                fig.savefig(f"follows_sbr: {seed}", dpi=fig.dpi)
            elif args.method == "citeseer_rnm":
                df = pd.read_csv(
                    "res_rnm_10_splits",
                    sep="\t",
                    names=["seed", "test_size", "acc_map", "acc_nn"],
                )
                fig = plt.figure()
                plt.plot(df["test_size"], df["acc_map"], label="Accuracy of MAP")
                plt.plot(
                    df["test_size"], df["acc_nn"], label="Accuracy of the NN prediction"
                )
                fig.savefig(f"follows_sbr: {seed}", dpi=fig.dpi)
            else:
                df = pd.read_csv(
                    "res_rnm_10_splits",
                    sep="\t",
                    names=["seed", "test_size", "acc_map", "acc_nn"],
                )
                fig = plt.figure()
                plt.plot(df["test_size"], df["acc_map"], label="Accuracy of MAP")
                plt.plot(
                    df["test_size"], df["acc_nn"], label="Accuracy of the NN prediction"
                )
                fig.savefig(f"follows_sbr: {seed}", dpi=fig.dpi)

    df_time.to_csv(f"time_analysis_{args.method}.csv")
