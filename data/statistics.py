from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # stdout or stderr
)
log = logging.getLogger(__name__)

def print_statistics(modality, mean, std, min, max):
    info="\n" + "=" * 60
    info+="\n"+f"Statistics (Modality: {modality})"
    info+="\n"+"=" * 60

    info+="\n"+f"Mean       : {mean}"
    info+="\n"+f"Std         : {std}"
    info+="\n"+f"Min         : {min}"
    info+="\n"+f"Max         : {max}"

    info+="\n"+"=" * 60 + "\n"
    return info

def calc_statistics(total_df, modality):

    for m in modality:
        count = None
        mean = None
        m2 = None
        min_vals = None
        max_vals = None

        cnt = 0
        for row in total_df.itertuples(index=False):
            store = zarr.storage.ZipStore(root / row.split / m / f"{row.keys}.zarr.zip", mode="r")
            ds = xr.open_zarr(store, consolidated=True)
            data = ds['bands'].values.astype(np.float64, copy=False)  # CHW
            ds.close()
            store.close()

            c = data.shape[0]
            flat = data.reshape(c, -1)
            valid = np.isfinite(flat)

            n_b = valid.sum(axis=1).astype(np.int64)
            sum_b = np.where(valid, flat, 0.0).sum(axis=1, dtype=np.float64)
            mean_b = np.divide(sum_b, n_b, out=np.zeros(c, dtype=np.float64), where=n_b > 0)
            diff = np.where(valid, flat - mean_b[:, None], 0.0)
            m2_b = np.sum(diff * diff, axis=1, dtype=np.float64)

            min_b = np.full(c, np.inf, dtype=np.float64)
            max_b = np.full(c, -np.inf, dtype=np.float64)
            for ch in range(c):
                if n_b[ch] > 0:
                    channel_vals = flat[ch, valid[ch]]
                    min_b[ch] = channel_vals.min()
                    max_b[ch] = channel_vals.max()

            if count is None:
                count = np.zeros(c, dtype=np.int64)
                mean = np.zeros(c, dtype=np.float64)
                m2 = np.zeros(c, dtype=np.float64)
                min_vals = np.full(c, np.inf, dtype=np.float64)
                max_vals = np.full(c, -np.inf, dtype=np.float64)
            elif c != count.shape[0]:
                raise ValueError(f"Channel size mismatch in modality {m}: expected {count.shape[0]}, got {c}")

            has_value = n_b > 0
            if np.any(has_value):
                n_old = count[has_value]
                n_batch = n_b[has_value]
                n_total = n_old + n_batch
                delta = mean_b[has_value] - mean[has_value]

                # Welford/Chan Algorithm for streaming calculation of mean and std.
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
                mean[has_value] = mean[has_value] + delta * (n_batch / n_total)
                m2[has_value] = m2[has_value] + m2_b[has_value] + (delta * delta) * (n_old * n_batch / n_total)
                count[has_value] = n_total
                min_vals[has_value] = np.minimum(min_vals[has_value], min_b[has_value])
                max_vals[has_value] = np.maximum(max_vals[has_value], max_b[has_value])
            
            # if (min_vals[1:4] < -900).any():
            #     log.info(f"{row.keys} {min_vals}")
            #     assert False

            cnt += 1
            if cnt % 1000 == 0: log.info(f"{m}: {cnt} / {len(total_df)}")

        variance = np.divide(m2, count, out=np.full_like(m2, np.nan), where=count > 0)
        std = np.sqrt(variance)
        mean = np.where(count > 0, mean, np.nan)
        min_vals = np.where(count > 0, min_vals, np.nan)
        max_vals = np.where(count > 0, max_vals, np.nan)

        log.info(f"{print_statistics(m, mean, std, min_vals, max_vals)}")

if __name__ == '__main__': 

    root = Path('/dss/dsstbyfs02/scratch/07/di54rur/zeroflood/ZeroFlood')

    # Read the CSV file
    train_val_df = pd.read_csv(os.path.join(root, 'metadata', 'info_train.csv'), delimiter='\t')
    test_df = pd.read_csv(os.path.join(root, 'metadata', 'info_val.csv'), delimiter='\t')
    train_val_df['split'] = 'train'
    test_df['split'] = 'val'


    total_df = pd.concat([train_val_df, test_df], ignore_index=True)
    # modality = ['S1RTC', 'S2L2A']
    modality = ['S2L2A']

    log.info(f"Calculate dataset statistics: {modality} {len(total_df)}")
    calc_statistics(total_df, modality)
