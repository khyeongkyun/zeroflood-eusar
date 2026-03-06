import os
import subprocess
import shutil
import argparse

from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(
    description="Download TerraMesh and/or LISFLOOD datasets to a specified root folder."
)
parser.add_argument(
    "--dataset",
    "-d",
    choices=["TerraMesh", "LISFLOOD", "all"],
    default="all",
    help="Which dataset to download: TerraMesh, LISFLOOD, or all (default).",
)

parser.add_argument(
    "--root",
    "-r",
    default='.',
    help="Root path where dataset folders will be created (default: current directory).",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print actions without performing downloads.",
)
parser.add_argument(
    "--modalities",
    "-m",
    default=None,
    help=(
        "Comma-separated list of modalities to download for TerraMesh. "
        "Valid values: DEM,LULC,NDVI,S1GRD,S1RTC,S2L1C,S2L2A,S2RGB. "
        "If omitted, all modalities are downloaded."
    ),
)

def download_terramesh_data(root: str = './', modalities: list = ['DEM','LULC','NDVI','S1GRD','S1RTC','S2L1C','S2L2A','S2RGB']):
    '''
    Downloads the TerraMesh dataset from Hugging Face.
    https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh
    
    '''
    # root = os.path.join(root, 'TerraMesh')
    Path(root).mkdir(parents=True, exist_ok=True)
    for m in modalities:
        snapshot_download(
            repo_id="ibm-esa-geospatial/TerraMesh",
            repo_type="dataset",
            allow_patterns=f"*/{m}/*",
            local_dir=root,
        )

    print(f"TerraMesh Data downloaded to {os.path.abspath(root)}")

def download_lisflood_data(root: str = './'):
    '''
    Downloads the LISFLOOD Flood Hazard Map.
    https://data.jrc.ec.europa.eu/dataset/1d128b6c-a4ee-4858-9e34-6210707f3c81#dataaccess
    '''
    # root = os.path.join(root, 'lisflood')
    Path(root).mkdir(parents=True, exist_ok=True)
    if shutil.which("wget") is None:
        raise RuntimeError("wget not found")
    cmd = [
        "wget",
        "-r", "-np", "-nH",
        "--cut-dirs=3",
        "-c", "--show-progress",
        "--reject=index.html*",
        "-P", f"{root}",
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/flood_hazard/"
    ]
    subprocess.run(cmd, check=True)
    print(f"LISFLOOD: Flood Hazard Map downloaded to {os.path.abspath(root)}")

if __name__ == "__main__":

    args = parser.parse_args()

    choice = args.dataset.lower()
    root = args.root
    modalities = None
    if getattr(args, "modalities", None):
        modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]

    actions = []
    if choice in ("terramesh", "all"):
        actions.append(("terramesh",  os.path.join(root, "TerraMesh")))
    if choice in ("lisflood", "all"):
        actions.append(("lisflood", os.path.join(root, "LISFLOOD")))

    for name, path in actions:
        if name == "terramesh":
            download_terramesh_data(root=path, modalities=modalities)
        elif name == "lisflood":
            download_lisflood_data(root=path)
