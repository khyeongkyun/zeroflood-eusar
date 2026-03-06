import argparse
from utils import process

parser = argparse.ArgumentParser(
    description="Check Image Quality from TerraMesh and Get metadata for the future usage.")

parser.add_argument(
    "--split",
    "-s",
    choices=["train", "val", "all"],
    default="all",
    help="Which split info are you interested? train, val, or all (default).",
)
parser.add_argument(
    "--root",
    "-r",
    default='.',
    help="Root path where dataset exist (default: current directory).",
)

if __name__ == "__main__":

    args = parser.parse_args()
    root = args.root

    if args.split == 'all':
        splits = ['val', 'train']
    else:
        splits = [args.split]

    for split in splits:

        process(
            split, root, 
            # max_samples=3,
            save_img=True,
            save_metadata=True)
