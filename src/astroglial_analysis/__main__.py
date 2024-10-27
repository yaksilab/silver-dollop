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

    parser.add_argument(
        "--segment_length",
        type=int,
        default=12,
        help="The length of the subsegments in the subsegmented masks",
    )

    args = parser.parse_args()

    print("="*70)
    print(f"Starting the pipeline to process masks in: {args.data_dir}")
    print(f"Using segment length: {args.segment_length}")
    print("="*70)
    run_pipeline(args.data_dir, args.segment_length)
    print("="*70)
    print("Pipeline processing completed successfully.")
    print("="*70)
