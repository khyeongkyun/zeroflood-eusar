import os
import ast
import re
import zarr

import xarray as xr
import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from datetime import timedelta

# OUTPUT_PATH: str | None = "summary.csv"

class color_palette():
    def __init__(self):

        self.LULC_COLORS = ['#000000','#0000FF', '#a6d854', "#66c2a5", "#fc8d62", 
                    "#b3b3b3", "#e5c494", "#FFFFFF", "#00E1FF", "#e78ac3"]
        self.LULC_LABELS = ['NoData',     'Water',        'Trees',    'F-Vegetation', 'Crops',
                    'BuiltArea',  'BareGround',   'SnowIce',  'Clouds',       'Rangeland']
        self.MASK_COLORS = ['#ffffff','#0000FF']
        self.MASK_LABELS = ['-', 'FloodHazard']

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

def viz_perf_distr_multi_model_metric(df_dict, col_x='risk', col_y=['f1'], ylim=(0,1), figsize=(6, 3)):

    # Ensure col_y is a list
    if isinstance(col_y, str):
        col_y = [col_y]

    combined_list = []
    bins = np.linspace(0, 100, 6)
    labels = [f"{int(b)}" for b in bins[1:]]

    custom_palette = {}
    hatch_map = {}
    all_cols = [col_x] + col_y
    for name, (df, color, hatch) in df_dict.items():
        custom_palette[name] = color
        hatch_map[name] = hatch
        temp = df[all_cols].copy()
        temp['x_bins'] = pd.cut(temp[col_x], bins=bins, labels=labels)
        temp['Source'] = name
        combined_list.append(temp)

    plot_df = pd.concat(combined_list)
    hue_order = list(df_dict.keys())
    n_bins = len(labels)
    n_metrics = len(col_y)

    fig, axs = plt.subplots(
        nrows=n_metrics, ncols=1,
        figsize=(figsize[0], figsize[1] * n_metrics),
        sharex=True,  # ← shared x-axis
    )

    # Ensure axs is always iterable
    if n_metrics == 1:
        axs = [axs]

    i = 0
    for ax, metric in zip(axs, col_y):
        sns.boxplot(
            data=plot_df,
            x='x_bins',
            y=metric,
            hue='Source',
            hue_order=hue_order,
            whis=0,
            showfliers=False,
            ax=ax,
            palette=custom_palette,
            gap=0.2,
            saturation=1,
            medianprops={"color": "r", "linewidth": 1},
            native_scale=True,
        )

        # Apply hatches per box group
        patches = [p for p in ax.patches if type(p) == mpatches.PathPatch]
        for idx, patch in enumerate(patches):
            model_name = hue_order[idx // n_bins]
            patch.set_hatch(hatch_map[model_name])
            patch.set_facecolor(custom_palette[model_name])
            if hatch_map[model_name]:
                patch.set_edgecolor('black')

        ax.set_ylim(ylim[i])
        ax.set_ylabel(metric)
        ax.set_xlabel('')       # ← suppress per-subplot x label
        ax.get_legend().remove()  # ← remove per-subplot legend
        ax.grid(True, linestyle='--', alpha=0.6)
        
        i+=1

    # Shared x label on the bottom subplot only
    axs[-1].set_xlabel(col_x)

    # Single shared legend on the bottom subplot
    legend_handles = [
        mpatches.Patch(
            facecolor=custom_palette[name],
            hatch=hatch_map[name],
            edgecolor='black',
            label=name,
        )
        for name in hue_order
    ]
    axs[-1].legend(
        handles=legend_handles,
        loc='lower right',
        ncols=2,
    )

    plt.tight_layout()
    plt.show()

    return plot_df

def viz_perf_distr_multi_model(df_dict, col_x='risk', col_y='f1', ylim=(0,1), figsize=(6, 3)):

    combined_list = []
    bins = np.linspace(0, 100, 6)
    labels = [f"{int(b)}" for b in bins[1:]]

    custom_palette = {}
    hatch_map = {}
    for name, (df, color, hatch) in df_dict.items():
        custom_palette[name] = color
        hatch_map[name] = hatch
        temp = df[[col_x, col_y]].copy()
        temp['x_bins'] = pd.cut(temp[col_x], bins=bins, labels=labels)
        temp['Source'] = name
        combined_list.append(temp)
    
    plot_df = pd.concat(combined_list)
    hue_order = list(df_dict.keys())

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df,
        x='x_bins',
        y=col_y,
        hue='Source',
        hue_order=hue_order,
        whis=0,
        showfliers=False,
        ax=ax,
        palette=custom_palette,
        gap=0.2,
        saturation=1,
        medianprops={"color": "r", "linewidth": 1},
        native_scale=True
    )

    # Apply hatches per box group
    n_bins = len(labels)
    patches = [p for p in ax.patches if type(p) == mpatches.PathPatch]
    for idx, patch in enumerate(patches):
        model_name = hue_order[idx // n_bins]
        patch.set_hatch(hatch_map[model_name])
        patch.set_facecolor(custom_palette[model_name])
        if hatch_map[model_name]:
            patch.set_edgecolor('black')

    ax.set_ylim(ylim)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.grid(True, linestyle='--', alpha=0.6)
    legend_handles = [
            mpatches.Patch(
                facecolor=custom_palette[name],
                hatch=hatch_map[name],
                edgecolor='black',
                label=name,
            )
            for name in hue_order
        ]
    ax.legend(
        handles=legend_handles,
        # title='Model',
        # bbox_to_anchor=(1.05, 1),
        loc='lower right',
        ncols=2
    )

    plt.tight_layout()
    plt.show()

    return plot_df

def get_sample_keys_auto(test_df: pd.DataFrame, n: int, low_high=(0.3,0.7), by='Water body in Flood Hazard (%)') -> pd.DataFrame:

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

    R: VV   G: VH   B: VV/VH
    """
    mean = STATS_FOR_VIZ['mean']['S1RTC']  # shape: (number of S1 bands,)
    std = STATS_FOR_VIZ['std']['S1RTC']
    img = normalize(data,mean,std)

    r = img[0].astype(np.float32)
    g = img[1].astype(np.float32)

    # Compute ratio, avoid division by zero
    # ratio = np.where(g != 0, r / g, 0.0)
    ratio = np.zeros_like(r)
    np.divide(r, g, out=ratio, where=g != 0)

    # Normalize only the ratio channel to [0, 1]
    vmin, vmax = ratio.min(), ratio.max()
    blue = (ratio - vmin) / (vmax - vmin) if vmax - vmin != 0 else np.zeros_like(ratio)
    b = (blue * 255).clip(0, 255)

    return np.stack([r, g, b], axis=-1).astype(np.uint8) 
    # # Normalize each band
    # img = normalize(data,mean,std)
    # print(img[1].shape)
    # return img[1]

def apply_mask_hatch(ax, img, hatch='....', color='#0000FF', alpha=0.4):
    """Overlay a hatch pattern on flood-risk pixels (value == 1)."""
    flood_mask = np.where(img == 1, 1.0, np.nan)
    ax.contourf(
        flood_mask,
        levels=[0.5, 1.5],
        hatches=[hatch],
        colors='none',
        # linewidths=0.5,
    )
    # Draw hatch lines in the desired color
    for collection in ax.collections:
        collection.set_edgecolor(color)

    hatch_patch = [
        mpatches.Patch(
            facecolor=color,
            edgecolor=color,
            hatch=hatch,
            label='FloodHazard',
            alpha=0.4,
            )
        ]

    return hatch_patch

def get_img(modality, data):

    c_info = color_palette()

    lulc_colors = c_info.LULC_COLORS
    lulc_labels = c_info.LULC_LABELS
    lulc_cmap = ListedColormap(lulc_colors)

    mask_colors = c_info.MASK_COLORS
    mask_labels = c_info.MASK_LABELS
    mask_cmap = ListedColormap(mask_colors)

    vmin, cmap = None, None

    if modality[:2] == 'S1': img = get_s1_img(data)
    elif modality[:2] == 'S2': img = get_s2_img(data)
    elif modality == 'DEM': img, cmap = data[0], 'BrBG_r'
    elif modality == 'LULC': img, vmin, cmap = data[0], 0, lulc_cmap
    elif modality == 'LULC_water': img, vmin, cmap, lulc_colors, lulc_labels = data[0]==1 , 0, mask_cmap, mask_colors, mask_labels
    # elif modality in ['PRED','MASK']: img, vmin, cmap = data[0], 0, MASK_CMAP
    else: img= data[0]

    return img, cmap, vmin, lulc_colors, lulc_labels

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
        tar_lst.append(100*precision_score(mask, pred, pos_label=1, zero_division=0))
        hr_lst.append(100*recall_score(mask, pred, pos_label=1, zero_division=0))
        f1_lst.append(100*f1_score(mask, pred, pos_label=1, zero_division=0))
        miou_lst.append(100*jaccard_score(mask, pred, average='macro', zero_division=0))

    df['mIoU'] = miou_lst 
    df['F1'] = f1_lst 
    df['Hit Rate'] = hr_lst 
    df['True Alarm'] = tar_lst 

    return df

def plot_legend(patches, ncol=10, figsize=(12, .5)):
    fig_legend, ax_legend = plt.subplots(figsize=figsize)
    ax_legend.legend(
        handles=patches,
        loc='center',
        ncol=ncol,
        frameon=False,
        fontsize='medium',
    )
    ax_legend.axis('off')
    plt.tight_layout()
    plt.show()

def save_ax_as_img(fig, ax, filepath):
    """Extract and save a single axes from a figure as an individual image."""

    for spine in ax.spines.values():
        spine.set_visible(False)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filepath, bbox_inches=extent)

def viz_sample_multimodel_keys(
        sample_keys: pd.DataFrame, 
        dataset_root: Path, 
        model_root_list: list,
        model_name: list,
        modalities=['S1RTC', 'S2RGB'],
        ref_modalities=['S2RGB'],
        fig_scale=1,
        save_img=False,
        show_legend=True,
        ):

    nrows = len(sample_keys)
    ncols = len(modalities) + len(model_root_list) + 1   # the last one is for the ground truth
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale*ncols, fig_scale*nrows))

    i = 0
    for idx, row in tqdm(sample_keys.iterrows(), total=sample_keys.shape[0]):

        key = row['keys']
        water_in_risk = row['Water body in Flood Hazard (%)']

        for j, m in enumerate(modalities):
            
            store = zarr.storage.ZipStore(os.path.join(dataset_root, m, f"{key}.zarr.zip"), mode="r")
            ds = xr.open_zarr(store, consolidated=True)
            img, cmap, vmin, lulc_colors, lulc_labels = get_img(m, ds['bands'].values)
            axs[i, j].imshow(img, cmap, vmin=vmin)  # ← j instead of j+1
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if j > 0:
                axs[i, j].axis('off')

        # Get water mask from LULC
        store = zarr.storage.ZipStore(os.path.join(dataset_root, 'LULC', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        img_water, cmap, vmin, lulc_water_color, _ = get_img('LULC_water', ds['bands'].values)
        axs[i, -1].imshow(img_water, cmap, vmin=vmin)

        for k, model_root in enumerate(model_root_list):

            # Image of Predictions
            axs[i, j+k+1].imshow(img_water, cmap, vmin=vmin)
            store = zarr.storage.ZipStore(os.path.join(model_root, 'PRED', f"{key}.zarr.zip"), mode="r")
            ds = xr.open_zarr(store, consolidated=True)
            img, _, vmin, _, _ = get_img('PRED', ds['bands'].values)
            hatch_patch = apply_mask_hatch(
                axs[i, j+k+1], img, color=lulc_water_color[-1],
                )
            axs[i, j+k+1].set_xticks([])
            axs[i, j+k+1].set_yticks([])

        # Image of Ground truth
        store = zarr.storage.ZipStore(os.path.join(dataset_root, 'MASK', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        img, cmap, vmin, _, _ = get_img('MASK', ds['bands'].values)
        hatch_patch = apply_mask_hatch(
            axs[i, -1], img, color=lulc_water_color[-1],
            )
        axs[i, -1].set_xticks([])
        axs[i, -1].set_yticks([])

        # Water-in-risk text box at lower right of last column
        axs[i, -1].text(
            0.98, 0.02,
            f"WB in FH: {water_in_risk:.1f} %",
            transform=axs[i, -1].transAxes,
            ha='right', va='bottom',
            fontsize=fig_scale+8,
            bbox=dict(
                boxstyle='round, pad=0.3',
                facecolor='white',
                edgecolor='grey',
                alpha=0.8,
            ),
        )

        i += 1
        
    # Set column headers
    title_lst = modalities + model_name + ['Ground Truth']
    for ax, title in zip(axs[0], title_lst): 
        if title in ref_modalities:
            ax.set_title(r'$\mathrm{' + '('+ title +')' + r'_{ref}}$', fontsize=12,)
        else:
            ax.set_title(title, fontsize=12,)

    if show_legend:

        if 'DEM' in title_lst:
            dem_idx = title_lst.index('DEM')

            # Get the axes at the bottom of the DEM column (last row)
            ax_dem = axs[-1, dem_idx]

            # Create an inset axes at the bottom of the DEM column
            ax_cbar = inset_axes(
                ax_dem,
                width='100%',
                height='10%',
                loc='lower center',
                bbox_to_anchor=(0, -0.18, 1, 1),
                bbox_transform=ax_dem.transAxes,
                borderpad=0,
            )

            # Draw colorbar
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='BrBG_r', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
            cbar.set_ticks([])  # ← remove ticks entirely

            # Place 'Low' and 'High' inside the colorbar
            ax_cbar.text(0.01, 0.5, 'Low',  transform=ax_cbar.transAxes,
                         ha='left',  va='center', fontsize=8, color='white',)
            ax_cbar.text(0.99, 0.5, 'High', transform=ax_cbar.transAxes,
                         ha='right', va='center', fontsize=8, color='white',)
            

        patches = [
            mpatches.Patch(
            facecolor=lulc_colors[i], 
            label=lulc_labels[i], 
            edgecolor='#000000', 
            linewidth=0.2
            ) for i in range(len(lulc_colors))
            ]

        patches = [p for i, p in enumerate(patches) if lulc_labels[i] not in ['NoData', 'SnowIce', 'Clouds']]
        patches = hatch_patch + patches
        patches = patches[2:] + [patches[1]] + [patches[0]] # Reorder 
        if 'LULC' not in modalities:
            patches = patches[-2:]

        fig.legend(
            handles=patches,
            loc='lower right',
            bbox_to_anchor=(0.99, -.015),
            bbox_transform=fig.transFigure,  # ← relative to the whole figure
            ncol=4,
            fontsize=10,
            frameon=True,
        )

    plt.tight_layout() 
    plt.show()

    # Save individual images after tight_layout
    if save_img:
        for idx, row in sample_keys.iterrows():
            key = row['keys']

            for j, m in enumerate(modalities):
                save_ax_as_img(fig, axs[idx, j], f"./{key}_{m}.png")  # ← j instead of j+1

            save_ax_as_img(fig, axs[idx, -2], f"./{key}_PRED.png")
            save_ax_as_img(fig, axs[idx, -1], f"./{key}_MASK.png")

    return patches


def viz_sample_keys(
        sample_keys: pd.DataFrame, 
        dataset_root: Path, model_root: Path,
        modalities=['S1RTC', 'S2RGB', 'DEM', 'LULC'],
        tim_modalities=['S2RGB', 'DEM'],
        ref_modalities=['LULC'],
        fig_scale=1,
        save_img=False,
        show_legend=True,
        ):

    nrows = len(sample_keys)
    ncols = len(modalities) + 1 + 1  # the last one is for the ground truth
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_scale*ncols, fig_scale*nrows))

    i = 0
    for idx, row in tqdm(sample_keys.iterrows(), total=sample_keys.shape[0]):

        key = row['keys']
        water_in_risk = row['Water body in Flood Hazard (%)']

        for j, m in enumerate(modalities):
            
            store = zarr.storage.ZipStore(os.path.join(dataset_root, m, f"{key}.zarr.zip"), mode="r")
            ds = xr.open_zarr(store, consolidated=True)
            img, cmap, vmin, lulc_colors, lulc_labels = get_img(m, ds['bands'].values)
            axs[i, j].imshow(img, cmap, vmin=vmin)  # ← j instead of j+1
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if j > 0:
                axs[i, j].axis('off')
        
        # Get water mask from LULC
        store = zarr.storage.ZipStore(os.path.join(dataset_root, 'LULC', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        img_water, cmap, vmin, lulc_water_color, _ = get_img('LULC_water', ds['bands'].values)
        axs[i, -2].imshow(img_water, cmap, vmin=vmin)
        axs[i, -1].imshow(img_water, cmap, vmin=vmin)

        # Image of Predictions
        store = zarr.storage.ZipStore(os.path.join(model_root, 'PRED', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        img, cmap, vmin, _, _ = get_img('PRED', ds['bands'].values)
        hatch_patch = apply_mask_hatch(
            axs[i, -2], img, color=lulc_water_color[-1],
            )
        axs[i, -2].set_xticks([])
        axs[i, -2].set_yticks([])

        # Image of Ground truth
        store = zarr.storage.ZipStore(os.path.join(dataset_root, 'MASK', f"{key}.zarr.zip"), mode="r")
        ds = xr.open_zarr(store, consolidated=True)
        img, cmap, vmin, _, _ = get_img('MASK', ds['bands'].values)
        hatch_patch = apply_mask_hatch(
            axs[i, -1], img, color=lulc_water_color[-1],
            )
        axs[i, -1].set_xticks([])
        axs[i, -1].set_yticks([])

        # Water-in-risk text box at lower right of last column
        axs[i, -1].text(
            0.98, 0.02,
            f"WB in FH: {water_in_risk:.1f} %",
            transform=axs[i, -1].transAxes,
            ha='right', va='bottom',
            fontsize=fig_scale+8,
            bbox=dict(
                boxstyle='round, pad=0.3',
                facecolor='white',
                edgecolor='grey',
                alpha=0.8,
            ),
        )

        i += 1
        
    # Set column headers
    title_lst = modalities + ['Prediction', 'Ground Truth']
    for ax, title in zip(axs[0], title_lst):
        if title in tim_modalities:
            if title == 'S2RGB': title = 'S2L2A'
            ax.set_title(r'$\mathrm{' + '('+ title +')' + r'_{TiM}}$', fontsize=12,)
        elif title in ref_modalities:
            ax.set_title(r'$\mathrm{' + '('+ title +')' + r'_{ref}}$', fontsize=12,)
        else:
            ax.set_title(title, fontsize=12,)

    if show_legend:
        if 'DEM' in title_lst:
            dem_idx = title_lst.index('DEM')

            # Get the axes at the bottom of the DEM column (last row)
            ax_dem = axs[-1, dem_idx]

            # Create an inset axes at the bottom of the DEM column
            ax_cbar = inset_axes(
                ax_dem,
                width='100%',
                height='10%',
                loc='lower center',
                bbox_to_anchor=(0, -0.18, 1, 1),
                bbox_transform=ax_dem.transAxes,
                borderpad=0,
            )

            # Draw colorbar
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='BrBG_r', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
            cbar.set_ticks([])  # ← remove ticks entirely

            # Place 'Low' and 'High' inside the colorbar
            ax_cbar.text(0.01, 0.5, 'Low',  transform=ax_cbar.transAxes,
                         ha='left',  va='center', fontsize=8, color='white',)
            ax_cbar.text(0.99, 0.5, 'High', transform=ax_cbar.transAxes,
                         ha='right', va='center', fontsize=8, color='white',)
            
        patches = [
            mpatches.Patch(
            facecolor=lulc_colors[i], 
            label=lulc_labels[i], 
            edgecolor='#000000', 
            linewidth=0.2
            ) for i in range(len(lulc_colors))
            ]

        patches = [p for i, p in enumerate(patches) if lulc_labels[i] not in ['NoData', 'SnowIce', 'Clouds']]
        patches = hatch_patch + patches
        patches = patches[2:] + [patches[1]] + [patches[0]] # Reorder 
        if 'LULC' not in modalities:
            patches = patches[-3:]

        fig.legend(
            handles=patches,
            loc='lower right',
            bbox_to_anchor=(0.99, -.025),
            bbox_transform=fig.transFigure,  # ← relative to the whole figure
            ncol=4,
            fontsize=10,
            frameon=True,
        )

    plt.tight_layout() 
    plt.show()

    # Save individual images after tight_layout
    if save_img:
        for idx, row in sample_keys.iterrows():
            key = row['keys']

            for j, m in enumerate(modalities):
                save_ax_as_img(fig, axs[idx, j], f"./{key}_{m}.png")  # ← j instead of j+1

            save_ax_as_img(fig, axs[idx, -2], f"./{key}_PRED.png")
            save_ax_as_img(fig, axs[idx, -1], f"./{key}_MASK.png")

    return patches

def viz_tim_bar_plot(groups, metric, ylim=None, figsize=(5, 3), legend_true=True):

    bar_width = 0.1
    item_gap = 0.01
    group_gap = 0.1

    x_positions = []
    labels = []
    values = []
    colors = []
    group_boundaries = []
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
            group_boundaries.append((group_name, local_positions[0], local_positions[-1]))
            cursor += group_gap

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_positions, values, width=bar_width, color=colors)

    # Center ylim around ref_value
    ref_value = float(groups[0][1][metric])
    ref_label = groups[0][0]
    half_range = max(abs(ref_value - min(values)), abs(max(values) - ref_value))
    if not ylim:
        ylim = (0.99*(ref_value - half_range), 1.01*(ref_value + half_range))
        
    # Labels on top of each bar
    for x, v, label in zip(x_positions, values, labels):
        ax.text(
            x, v + (ylim[1] - ylim[0]) * 0.01,
            label,
            ha='center', va='bottom',
            fontsize=7,
            rotation=45,
        )

    # Remove x tick labels, set ticks at group centers
    ax.set_xticks([
        (x_left + x_right) / 2
        for _, x_left, x_right in group_boundaries
    ])
    ax.set_xticklabels([
        group_name
        for group_name, _, _ in group_boundaries
    ], fontsize=9)

    ax.set_ylabel(metric)
    ax.axhline(ref_value, color='black', linestyle=':', linewidth=1.5)
    if legend_true:
        legend_handles = [
            mlines.Line2D([], [], color='black', linestyle=':', linewidth=1.5, label=ref_label),
            mpatches.Patch(facecolor=groups[1][2], label='TerraMind-TiM'),
        ]
        ax.legend(handles=legend_handles, fontsize=8)   

    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.set_ylim(ylim)

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
        "T_train": "T_train",
        "T_test": "T_test"
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

    matches = re.findall(r'\d+:\d{2}:\d{2}', lines[marker_idx])
    h, m, s = map(int, matches[-1].split(':'))
    metrics["T_train"] = timedelta(hours=h, minutes=m, seconds=s)
    for raw_line in lines[marker_idx + 1 :]:

        matches = re.findall(r'\d+:\d{2}:\d{2}', raw_line)
        if matches:
            h, m, s = map(int, matches[-1].split(':'))
            duration = timedelta(hours=h, minutes=m, seconds=s)

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
    
    metrics["T_test"] = duration - metrics["T_train"]

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
        100 * tp / (tp + fp) if (tp + fp) > 0 else None
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
        bins = np.arange(0,105,5)
        _ = plt.hist(water_in_risk, bins=bins)
        plt.axvline(np.median(water_in_risk), color='k', linestyle='dashed', linewidth=1)
        plt.text(median, plt.gca().get_ylim()[1] * 0.95, f'median={median:.1f}',
                 rotation=90, va='top', ha='right', fontsize=9, color='k')
        plt.xlim(0,100)
        plt.xlabel('Water body in Flood hazard area (%)')
        plt.ylabel('Sample count')

    return pd.DataFrame({'keys': keys, 'Water body in Flood Hazard (%)': water_in_risk})

def viz_lon_lat_heatmap(df: pd.DataFrame, figsize=(5,5), cmap="Blues", title_prefix=None):
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
    heatmap = np.ma.masked_where(heatmap == 0, heatmap)

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
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.35, location='left',  pad=0.02)
    if title_prefix:
        plt.title(f"{title_prefix} heatmap: {len(df)} ea")
    cbar.set_label("Sample count")
    plt.tight_layout()
    plt.show()
