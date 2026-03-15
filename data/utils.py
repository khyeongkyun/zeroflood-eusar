from datetime import datetime
import csv
from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from pyproj import CRS
from affine import Affine
import xarray as xr
import zarr
from sklearn.metrics import confusion_matrix

from terramesh import build_terramesh_dataset, timestamp_to_str
from torch.utils.data import DataLoader

from rasterio.transform import array_bounds

def aoi_bounds_from_transform(transform, width, height):
    return array_bounds(height, width, transform)

def read_window(
    mask_tif,
    aoi_bounds_utm,
    aoi_crs,
):
    with rasterio.open(mask_tif) as src:
        mask_crs = src.crs

        # Transform AOI bounds into flood CRS
        bounds_ll = transform_bounds(
            aoi_crs,
            mask_crs,
            *aoi_bounds_utm,
        )

        # Compute raster window
        window = src.window(*bounds_ll)
        window = window.round_offsets().round_lengths()

        if window.width <= 0 or window.height <= 0:
            return None, None, None

        data = src.read(1, window=window)
        transform = src.window_transform(window)

        return data, transform, mask_crs

def reproject_to_aoi(
    mask_data,
    mask_transform,
    mask_crs,
    aoi_transform,
    aoi_crs,
    aoi_shape,
):
    aligned = np.zeros(aoi_shape, dtype=mask_data.dtype)

    reproject(
        source=mask_data,
        destination=aligned,
        src_transform=mask_transform,
        src_crs=mask_crs,
        dst_transform=aoi_transform,
        dst_crs=aoi_crs,
        resampling=Resampling.nearest,
    )

    return aligned

def extract_mask_aoi(
    mask_tif,
    aoi_transform,
    aoi_crs,
    aoi_shape,
    threshold: float = 0.1,
):
    aoi_bounds = aoi_bounds_from_transform(
        aoi_transform,
        aoi_shape[1],
        aoi_shape[0]
    )

    mask_data, mask_transform, mask_crs = read_window(
        mask_tif,
        aoi_bounds,
        aoi_crs
    )

    if mask_data is None:
        return False, None
    
    aligned = reproject_to_aoi(
        mask_data,
        mask_transform,
        mask_crs,
        aoi_transform,
        aoi_crs,
        aoi_shape
    )

    mask = aligned > threshold

    del mask_data, aligned

    return mask

def utm_crs_from_lonlat(lon, lat):
    zone = math.floor((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def calculate_iou(mask1, mask2):

    if not isinstance(mask1, np.ndarray):
        mask1 = np.array(mask1)
    if not isinstance(mask2, np.ndarray):
        mask2 = np.array(mask2)
        
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def calculate_FP(truth, pred):
    # Count False Positive pixels
    fp = np.logical_and(pred == 1, truth == 0).sum()
    return fp

def save_to_zarr(data, out_dir, lat, lon, crs, x, y, time):
    ds = xr.Dataset(
        data_vars={
            "bands": (("band", "y", "x"), data),
            "center_lat": lat,
            "center_lon": lon,
            "crs": crs,
        },
        coords={
            "x": x,
            "y": y,
            "time": time,
        },
    )

    store = zarr.storage.ZipStore(f"{out_dir}.zip", mode="w")
    ds.to_zarr(
        store, 
        mode="w", 
        consolidated=True
        )
    store.close()

def process(
        terramesh_split: str = 'train', 
        root: str = '.',
        max_samples: int = -1,
        save_img: bool = False,
        save_metadata: bool = False,
        ):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Data Processing Start! TerraMesh - {terramesh_split}")

    if save_metadata:
        csv_path = Path(f"./info_{terramesh_split}.csv")
        csv_file = open(csv_path, "w", newline='', encoding="utf-8")
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerow([
            'keys', 'time_s1', 'time_s2', 
            'tn_fn_fp_tp',
            'lon_lat','x0_y0','epsg'
            ])

    # Constants
    TERRAMESH_RES = 10  # meters
    TERRAMESH_SIZE = 264  # pixels
    LISFLOOD_EXTENT_LON = (-24.54, 67.26)  # (min_lon, min_lat, max_lon, max_lat)
    LISFLOOD_EXTENT_LAT = (27.81, 71.13)
    
    # Initialize dataset and dataloader
    dataset = build_terramesh_dataset(
        path=f"{root}/TerraMesh",  # Streaming or local path
        modalities=['S1RTC', "S2L2A", 'S2RGB', "LULC", "DEM"],  
        split=terramesh_split,
        shuffle=False,  # Set false for split="val"
        batch_size=1,
        return_metadata=True,
        time_dim=True,
    )
    dataloader = DataLoader(dataset, 
                            batch_size=None, 
                            num_workers=4,              # NOTE: Match with cpus-per-task in slurm script
                            persistent_workers=True,  
                            prefetch_factor=1         
                            )

    # Load and process samples
    sample_count = 0
    for idx, batch in enumerate(dataloader):
        try:
            # Condition checking and progress logging
            if idx % 5000 == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] {sample_count}/{idx} samples processed.")
                # TODO:
                # Skipping...
                # Spurious      FloodPixel      Low_waterIoU    High FN
                # 0             0               0               0
            keys = list(batch.keys())
            if not ('time_S1RTC' in keys and 'time_S2RGB' in keys):
                continue
            if max_samples != -1 and sample_count >= max_samples:
                break
            
            # Read TerraMesh sample
            keys = batch['__key__'][0]

            time_S1RTC, time_S2L2A = int(batch['time_S1RTC'][0]), int(batch['time_S2L2A'][0])
            lon, lat = batch['center_lon'][0].item(), batch['center_lat'][0].item()
            x, y = batch['x'][0].tolist(), batch['y'][0].tolist()
            crs = batch['crs'][0].item()
            
            S1RTC, S2L2A = batch['S1RTC'].squeeze(), batch['S2L2A'].squeeze()
            RGB, LULC = batch['S2RGB'].squeeze().byte(), batch['LULC'].squeeze().reshape(1, 264, 264)
            DEM = batch['DEM'].squeeze().reshape(1, 264, 264) 

            LULC_W = LULC[0] == 1
            lulc_water = LULC_W.sum().item()
            lulc_snow_ice = (LULC[0] == 7).sum().item()
            lulc_clouds = (LULC[0] == 8).sum().item()

            # Check TerraMesh sample quality
            # 1. (lon, lat) within LISFLOOD data coverage: Europe (-24.54, 67.26; 27.81, 71.13)
            # 2. Exclude samples with snow/ice and clouds in LULC
            # 3. Exclude samples with too much or too less water in LULC
            if not (LISFLOOD_EXTENT_LON[0] < lon < LISFLOOD_EXTENT_LON[1]): continue
            if not (LISFLOOD_EXTENT_LAT[0] < lat < LISFLOOD_EXTENT_LAT[1]): continue
            if lulc_snow_ice or lulc_clouds: continue
            if not (TERRAMESH_SIZE**2 * 0.05 < lulc_water < TERRAMESH_SIZE**2 * 0.95): continue
            
            # Check LISFLOOD sample quality
            # 1. Exclude samples with spurious depth areas
            # 2. Exclude samples with too less flooded pixels - flooded pixels > 1%
            aoi_crs = CRS.from_epsg(crs)
            aoi_transform = Affine(
                    TERRAMESH_RES, 0, x[0] - (TERRAMESH_RES / 2),
                    0, -TERRAMESH_RES, y[0] + (TERRAMESH_RES / 2)
                )
            spurious_mask = extract_mask_aoi(
                f"{root}/LISFLOOD/flood_hazard/Europe_spurious_depth_areas.tif",
                aoi_transform, aoi_crs, (TERRAMESH_SIZE,TERRAMESH_SIZE),
            )
            if np.count_nonzero(spurious_mask) > 0:
                print(f"Skipping... {keys} - Spurious mask exist")
                continue
            flood_mask = extract_mask_aoi(
                f"{root}/LISFLOOD/flood_hazard/Europe_RP10_filled_depth.tif",
                aoi_transform, aoi_crs, (TERRAMESH_SIZE,TERRAMESH_SIZE),
            ).reshape(1, 264, 264)
            flood_mask_ratio = np.count_nonzero(flood_mask[0]) / (TERRAMESH_SIZE**2)
            if flood_mask_ratio < 0.01:
                print(f"Skipping... {keys} - # Flooded pixels: {flood_mask_ratio:.2f} < 0.01 ")
                continue

            # Align TerraMesh and LISFLOOD
            # 1. Skip water mask IoU < 0.5 : LISFLOOD water and LULC water
            # 2. Skip flood mask FN > 0.04 : LULC water mask but not LISFLOOD RP10
            water_mask = extract_mask_aoi(
                f"{root}/LISFLOOD/flood_hazard/Europe_permanent_water_bodies.tif",
                aoi_transform, aoi_crs, (TERRAMESH_SIZE,TERRAMESH_SIZE),
            )
            water_iou = calculate_iou(water_mask, LULC_W)
            if water_iou < 0.5:
                print(f"Skipping... {keys} - Low water mask IoU: {water_iou:.3f} < 0.5")
                continue
            cm = confusion_matrix(LULC_W.ravel(), flood_mask[0].ravel(), labels=[False, True])
            TN, FN, FP, TP= cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
            FN_ratio = FN / (TN+FN+FP+TP)
            if FN_ratio > 0.04:
                print(f"Skipping... {keys} - High 'water not flood': {FN_ratio:.3f} > 0.04")
                continue

            # Save processed data
            if save_img:
                out_dir = Path(f"{root}/ZeroFlood/{terramesh_split}")
                out_dir.mkdir(parents=True, exist_ok=True)
                info = [
                    ("S1RTC", S1RTC, time_S1RTC),
                    ("S2L2A", S2L2A, time_S2L2A),
                    ("S2RGB", RGB, time_S2L2A),
                    ("LULC", LULC, time_S2L2A),
                    ("DEM", DEM, time_S2L2A),
                    ("MASK", flood_mask, None),
                ]
                for tag, data, timestamp in info:
                    out_sub_dir = out_dir / tag
                    out_sub_dir.mkdir(parents=True, exist_ok=True)
                    save_to_zarr(data, out_sub_dir / f"{keys}.zarr", lat, lon, crs, x, y, timestamp)
                    
            if save_metadata:
                writer.writerow(
                    [
                        keys,
                        timestamp_to_str(time_S1RTC), timestamp_to_str(time_S2L2A),
                        (int(TN), int(FN), int(FP), int(TP)),
                        (lon, lat), (x[0], y[0]), crs
                    ]
                )
            del S1RTC, S2L2A, RGB, LULC, LULC_W, DEM, flood_mask, water_mask, spurious_mask
            
            sample_count += 1
        
        except Exception as e:
            print(f"[ERROR] idx={idx}, key={keys}: {e}")
            continue

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {sample_count}/{idx} DONE!")

    return True