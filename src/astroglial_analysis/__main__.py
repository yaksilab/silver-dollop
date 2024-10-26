import argparse
from astroglial_analysis.run_pipline import run_pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This pipline takes a directory containing cellpose masks and \
        creates subsegmented masks and then extracts traces from the subsegmented masks using Suite2p"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="The directory containing the cellpose '*_seg.npy' mask files to be processed",
    )
    args = parser.parse_args()

    run_pipeline(args.data_dir)
