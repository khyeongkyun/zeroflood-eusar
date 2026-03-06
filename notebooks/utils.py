import os
from pathlib import Path
import pandas as pd
import ast
import re

import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import logging
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import xarray as xr
import zarr

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from tqdm import tqdm
import seaborn as sns

# OUTPUT_PATH: str | None = "summary.csv"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def viz_perf_lines(df_dict, col_x='risk', col_y='f1'):
    """
    df_dict: dict of {'ModelName': df}
    """
    bins = np.linspace(0, 1, 11)
    # Using bin centers for more accurate x-axis positioning in a line plot
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    plt.figure(figsize=(6, 5))
    
    for name, df in df_dict.items():
        # Group by bins and calculate the mean for each bin
        binned_means = df.groupby(pd.cut(df[col_x], bins=bins), observed=False)[col_y].mean()
        
        # Plot the mean values against the bin centers
        plt.plot(bin_centers, binned_means.values, marker='o', label=name, linewidth=2)

    plt.ylim(0.2, 1)
    plt.xlim(0, 1)
    plt.xlabel(f'Bins of {col_x}')
    plt.ylabel(f'Mean {col_y}')
    plt.title(f'Mean {col_y} across {col_x} Bins')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def viz_perf_distr_multi(df_dict, col_x='risk', col_y='f1', ylim=(0,1)):
    """
    df_dict: dict of {'ModelName': df}
    """
    combined_list = []
    bins = np.linspace(0, 1, 11)
    labels = [f"{b:.1f}" for b in bins[1:]]

    custom_palette = {}
    for name, (df, color) in df_dict.items():
        custom_palette[name] = color
        temp = df[[col_x, col_y]].copy()
        temp['x_bins'] = pd.cut(temp[col_x], bins=bins, labels=labels)
        temp['Source'] = name
        combined_list.append(temp)

    # Merge all into one dataframe
    plot_df = pd.concat(combined_list)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_df, 
        x='x_bins', 
        y=col_y, 
        hue='Source',
        whis=0,
        showfliers=False,
        ax=ax,
        palette=custom_palette,
        gap=0.2
    )

    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def get_sample_keys_auto(test_df: pd.DataFrame, n: int, low_high=(0.3,0.7), by='water_in_risk') -> pd.DataFrame:

    test_df = test_df.sort_values(by=by, ascending=False).reset_index(drop=True)

    # the bottom xx % of data (Low Risk)
    # the top xx % of data (High Risk)
    low_risk_df = test_df.iloc[:int(len(test_df)*low_high[0])]
    high_risk_df = test_df.iloc[int(len(test_df)*low_high[0]):]

    n_low = int(n * low_high[1])
    n_high = n - n_low

    # Combine linear spacing from both parts
    idx_low = np.linspace(0, len(low_risk_df)-1, n_low).astype(int)
    idx_high = np.linspace(0, len(high_risk_df)-1, n_high).astype(int)

    sample_keys = pd.concat([
        low_risk_df.iloc[idx_low], 
        high_risk_df.iloc[idx_high]
    ]).drop_duplicates()

    return sample_keys

STATS_FOR_VIZ = {
    "mean": {
        "S2L1C": [2357.090, 2137.398, 2018.799, 2082.998, 2295.663, 2854.548, 3122.860, 3040.571, 3306.491, 1473.849,
                  506.072, 2472.840, 1838.943],
        "S2L2A": [1390.461, 1503.332, 1718.211, 1853.926, 2199.116, 2779.989, 2987.025, 3083.248, 3132.235, 3162.989,
                  2424.902, 1857.665],
        "S2RGB": [110.349, 99.507, 75.843],
        "S1GRD": [-12.577, -20.265],
        "S1RTC": [-10.93, -17.329],
        "NDVI": [0.327],
        "DEM": [651.663],
    },
    "std": {
        "S2L1C": [1673.639, 1722.641, 1602.205, 1873.138, 1866.055, 1779.839, 1776.496, 1724.114, 1771.041, 1079.786,
                  512.404, 1340.879, 1172.435],
        "S2L2A": [2131.157, 2163.666, 2059.311, 2152.477, 2105.179, 1912.773, 1842.326, 1893.568, 1775.656, 1814.907,
                  1436.282, 1336.155],
        "S2RGB": [69.905, 53.708, 53.378],
        "S1GRD": [5.179, 5.872],
        "S1RTC": [4.391, 4.459],
        "NDVI": [0.322],
        "DEM": [928.168]
    }
}

LULC_COLORS = ['#000000','#0000FF', '#00FF00', "#D0FF00", "#FF00D9", 
               "#574B57", "#FFA600", "#FFFFFF", "#00E1FF", "#FF002B"]
LULC_LABELS = ['NoData',     'Water',        'Trees',    'F-Vegetation', 'Crops',
            'BuiltArea',  'BareGround',   'SnowIce',  'Clouds',       'Rangeland']
LULC_CMAP = ListedColormap(LULC_COLORS)

MASK_COLORS = ['#000000','#0000FF']
MASK_LABELS = ['-', 'Flood_Risk']
MASK_CMAP = ListedColormap(MASK_COLORS)

def normalize(img,mean,std):
    for i in range(img.shape[0]):
        min_value = mean[i] - 2 * std[i]
        max_value = mean[i] + 2 * std[i]
        img[i] = (img[i] - min_value) / (max_value - min_value) * 255.0
        img[i] = np.clip(img[i], 0, 255).astype(np.uint8)
    return img

def get_s2_img(data: np.array):
    """
    Prepare S2 image for plotting.
    If data has 3 bands, return as-is.
    Otherwise, select bands 3,2,1 (B4,B3,B2) and normalize using statistics.
    """
    if data.shape[0] == 3:
        rgb = data
        return np.transpose(rgb, (1, 2, 0))
    
    # Select bands 3,2,1 for RGB (adjust indices if needed)
    rgb = data[[3, 2, 1], :, :]  # shape: (3, H, W)
   
    mean = STATS_FOR_VIZ['mean']['S2L2A'][1:4][::-1] # shape: (3,)
    std = STATS_FOR_VIZ['std']['S2L2A'][1:4][::-1]    # shape: (3,)

    rgb = normalize(rgb,mean,std)
    return np.transpose(rgb, (1, 2, 0))

def get_s1_img(data: np.array):
    """
    Prepare S1 image for plotting.
    Normalize each band using mean and std from statistics.
    """
    mean = STATS_FOR_VIZ['mean']['S1RTC']  # shape: (number of S1 bands,)
    std = STATS_FOR_VIZ['std']['S1RTC']
    
    # # Normalize each band
    img = normalize(data,mean,std)
    return img[1]

def get_img(modality, data):

    vmin, cmap = None, None

    if modality[:2] == 'S1': img = get_s1_img(data)
    elif modality[:2] == 'S2': img = get_s2_img(data)
    elif modality == 'DEM': img, cmap = data[0], 'BrBG_r'
    elif modality == 'LULC': img, vmin, cmap = data[0], 0, LULC_CMAP
    elif modality in ['PRED','MASK']: img, vmin, cmap = data[0], 0, MASK_CMAP
    else: img= data[0]

    return img, cmap, vmin

def calc_performance(df: pd.DataFrame, dataset_root: Path, model_root: Path):
    
    miou_lst, f1_lst, hr_lst, tar_lst = [], [], [], []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        key = row['keys']

        store = zarr.storage.ZipStore(os.path.join(model_root, 'PRED', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        pred = ds['bands'].values.flatten()

        store = zarr.storage.ZipStore(os.path.join(dataset_root, 'MASK', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        mask = ds['bands'].values.flatten()

        # 2. Calculate Precision, Recall, and F1 for class '1'
        tar_lst.append(precision_score(mask, pred, pos_label=1, zero_division=0))
        hr_lst.append(recall_score(mask, pred, pos_label=1, zero_division=0))
        f1_lst.append(f1_score(mask, pred, pos_label=1, zero_division=0))
        miou_lst.append(jaccard_score(mask, pred, average='macro', zero_division=0))

    df['miou'] = miou_lst
    df['f1'] = f1_lst
    df['hr'] = hr_lst
    df['tar'] = tar_lst

    return df

def viz_sample_keys(
        sample_keys: pd.DataFrame, 
        dataset_root: Path, model_root: Path,
        modalities=['S1RTC', 'S2RGB', 'DEM', 'LULC']
        ):

    nrows = len(sample_keys)
    ncols = len(modalities) + 2 + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*ncols, 2*nrows))

    i = 0
    for idx, row in tqdm(sample_keys.iterrows(), total=sample_keys.shape[0]):

        key = row['keys']
        water_in_risk = row['water_in_risk']

        axs[i,0].text(0.5, 0.5, f"{key.split('_')[-1]}\n\nWater in Risk area\n{100*water_in_risk:.3f} %", ha='center', va='center', fontsize=12)
        axs[i,0].axis('off')

        for j, m in enumerate(modalities):
            
            store = zarr.storage.ZipStore(os.path.join(dataset_root, m, f"{key}.zarr.zip"), mode="r")
            ds = xr.open_zarr(store, consolidated=True)
            img, cmap, vmin = get_img(m, ds['bands'].values)
            axs[i, j+1].imshow(img, cmap, vmin=vmin)
            axs[i, j+1].axis('off')
        
        store = zarr.storage.ZipStore(os.path.join(model_root, 'PRED', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        img, cmap, vmin = get_img('PRED', ds['bands'].values)
        axs[i, -2].imshow(img, cmap, vmin=vmin)
        axs[i, -2].axis('off')

        store = zarr.storage.ZipStore(os.path.join(dataset_root, 'MASK', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        img, cmap, vmin = get_img('MASK', ds['bands'].values)
        axs[i, -1].imshow(img, cmap=cmap, vmin=vmin)
        axs[i, -1].axis('off')

        i+=1
        
    # Set column headers
    for ax, title in zip(axs[0], ['Key'] + modalities + ['Prediction', 'Ground Truth']):
        ax.set_title(title, fontsize=14)

    # Create legend if there is LULC in modalities
    if 'LULC' in modalities:
        patches = [
            mpatches.Patch(
            facecolor=LULC_COLORS[i], 
            label=LULC_LABELS[i], 
            edgecolor='#000000', 
            linewidth=0.2
            ) for i in range(len(LULC_COLORS))
            ]

        # Place legend relative to the figure
        fig.legend(
            handles=patches,
            loc='lower right', 
            bbox_to_anchor=(1, 0.04),
            ncol=10,             # Fixed columns usually look cleaner than one long row
            frameon=False,
            fontsize='medium'
            )

    plt.tight_layout(rect=[0.05, 0.05, 1, 1]) 
    # plt.tight_layout()
    plt.show()

def viz_tim_bar_plot(groups, metric, ylim):

    bar_width = 0.1
    item_gap = 0.01
    group_gap = 0.1

    x_positions = []
    labels = []
    values = []
    colors = []
    group_centers = []
    cursor = 0.0
    for i, (group_name, group_df, color) in enumerate(groups):
        if i == 0: continue
        local_positions = []
        for _, row in group_df.reset_index(drop=True).iterrows():
            x_positions.append(cursor)
            local_positions.append(cursor)
            labels.append(row['Model'])
            values.append(float(row[metric]))
            colors.append(color)
            cursor += bar_width + item_gap

        if local_positions:
            group_centers.append((group_name, (local_positions[0] + local_positions[-1]) / 2))
            cursor += group_gap

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(x_positions, values, width=bar_width, color=colors)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Model Group')
    ax.set_ylim(ylim)

    # Dotted horizontal line at the value from tm_results_1 (first sorted item)
    ref_value = float(groups[0][1][metric].iloc[0])
    ax.axhline(ref_value, color='black', linestyle=':', linewidth=1.5)
    for group_name, center in group_centers:
        ax.text(center, ax.get_ylim()[0], group_name, ha='center', va='bottom', fontsize=10)

    # legend_handles = [Patch(facecolor=color, label=name) for name, _, color in groups]
    # ax.legend(handles=legend_handles, title='Groups')

    ax.grid(axis='y', linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.show()

def strip_info_prefix(line: str) -> str:
    """Remove leading logger metadata if present."""
    if line.startswith("INFO - "):
        parts = line.split(" - ", 4)
        if len(parts) == 5:
            return parts[4].strip()
    return line.strip()

def resolve_log_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_file():
        return path
    return path / "train.log-0"

def parse_final_test_metrics(log_path: Path) -> dict[str, dict[str, float]]:
    """Parse final [test] block after checkpoint loading."""

    CHECKPOINT_MARKER = "Loaded checkpoint__best for evaluation"

    LOG_METRICS_TO_SHORT = {
        "IoU": "IoU",
        "F1-score": "F1",
        "Precision": "Prec",
        "Recall": "Recall",
    }

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    marker_idx = -1
    for idx, line in enumerate(lines):
        if CHECKPOINT_MARKER in line:
            marker_idx = idx
    if marker_idx == -1:
        raise ValueError(f"Marker not found: {CHECKPOINT_MARKER}")

    metrics: dict[str, dict[str, float]] = {k: {} for k in LOG_METRICS_TO_SHORT.values()}
    current_metric: str | None = None

    for raw_line in lines[marker_idx + 1 :]:
        line = strip_info_prefix(raw_line)
        if not line:
            continue

        metric_match = re.search(r"\[test\]\s+-+\s+(.+?)\s+-+", line)
        if metric_match:
            long_name = metric_match.group(1).strip()
            current_metric = LOG_METRICS_TO_SHORT.get(long_name)
            continue

        if "Mean Accuracy:" in line:
            break

        if current_metric is None:
            continue

        mean_match = re.search(r"\[test\]\s+Mean\s+([0-9]+(?:\.[0-9]+)?)", line)
        if mean_match:
            metrics[current_metric]["mean"] = float(mean_match.group(1))
            continue

        if line.startswith("[test]"):
            continue

        class_match = re.match(r"([^\s]+)\s+([0-9]+(?:\.[0-9]+)?)$", line)
        if class_match:
            class_name = class_match.group(1)
            score = float(class_match.group(2))
            metrics[current_metric][class_name] = score

    return metrics

def build_row(run_name: str, parsed: dict[str, dict[str, float]]) -> dict[str, list[float]]:

    SELECTED_COLUMNS = [
        "IoU_mean",
        "F1_flood_rp10",
        "Prec_flood_rp10",
        "Recall_flood_rp10",
    ]

    values: list[float] = []
    for col in SELECTED_COLUMNS:
        if "_" not in col:
            values.append(np.nan)
            continue
        metric, target = col.split("_", 1)
        value = parsed.get(metric, {}).get(target)
        values.append(round(value, 2) if value is not None else np.nan)
    return values

def calc_confusion_matrix(df: pd.DataFrame, split: str, print_summary=True, print_hist=True, figsize=(5,5)):
    '''
    Confusion Matrix Information
    
                        FloodRisk_RP10
                      True            False
    LULC_water  --------------------------------
    True         |    TP       |       FN 
    False        |    FP       |       TN 
    '''
    tn_fn_fp_tp = df["tn_fn_fp_tp"].apply(ast.literal_eval).tolist()
    keys = df['keys'].tolist()

    water_in_risk = [
        tp / (tp + fp) if (tp + fp) > 0 else None
        for _, _, fp, tp in tn_fn_fp_tp
    ]

    min_idx = np.argmin(water_in_risk)
    max_idx = np.argmax(water_in_risk)
    mean = np.mean(water_in_risk)
    std = np.std(water_in_risk)
    median = np.median(water_in_risk)

    if print_summary:
        print("---------------------------------")
        print(f"{split}: Water body in Risk Zone")
        print()
        print(f"mean : {mean}")
        print(f"std : {std}")
        print(f"median : {median}")
        print()
        print(f"min : {water_in_risk[min_idx]} ({min_idx})")
        print(f"max : {water_in_risk[max_idx]} ({max_idx})")
        print("---------------------------------")

    if print_hist:
        plt.figure(figsize=figsize)
        bins = np.arange(0,1.05,0.05)
        _ = plt.hist(water_in_risk, bins=bins)
        plt.axvline(np.median(water_in_risk), color='k', linestyle='dashed', linewidth=1)
        plt.xlim(0,1)

    return pd.DataFrame({'keys': keys, 'water_in_risk': water_in_risk})

def viz_lon_lat_heatmap(df: pd.DataFrame, figsize=(5,5), cmap="Blues"):
    lon_lat = df["lon_lat"].apply(ast.literal_eval).tolist()

    lons = np.array([c[0] for c in lon_lat])
    lats = np.array([c[1] for c in lon_lat])

    LON_min, LON_max = -24.54, 67.26
    LAT_min, LAT_max = 27.81, 71.13
    RES = 1  # degree

    lon_bins = np.arange(LON_min, LON_max + RES, RES)
    lat_bins = np.arange(LAT_min, LAT_max + RES, RES)

    # 2D histogram (lat first, lon second)
    heatmap, _, _ = np.histogram2d(
        lats, lons,
        bins=[lat_bins, lon_bins]
    )


    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([LON_min, LON_max, LAT_min, LAT_max], crs=ccrs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")

    # Plot heatmap
    mesh = ax.pcolormesh(
        lon_bins,
        lat_bins,
        heatmap,
        cmap=cmap,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.4)
    plt.title(f"Sample heatmap: {len(df)}")
    cbar.set_label("Sample count")
    plt.tight_layout()
    plt.show()
