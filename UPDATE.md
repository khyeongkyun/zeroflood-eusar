# Changes from the original pangaea-bench

## ./configs
```
./configs
├── dataset
│   ├── zeroflood_sar.yaml              # New
│   └── zeroflood_optical.yaml          # New
├── encoder
│   ├── terramind_base_sar.yaml         # New
│   ├── terramind_base_sar_tim.yaml     # New
│   ├── terramind_large_sar.yaml        # New
│   └── terramind_large_sar_tim.yaml    # New
├── test.yaml                           # Editted (1)
└── train.yaml                          # Editted (1)
```
(1) Added `save_pred`.


## ./pangaea
```
./pangaea
├── datasets
│   └── zeroflood.py                    # New
├── encoder
│   ├── terramind                       # Editted (1)
│   ├── base.py                         # Editted (2)
│   ├── terramind_encoder.py            # Editted (3)
│   ├── terramind_tim_encoder.py        # New
│   └── vit_encoder.py                  # Editted (4)
├── engine
│   └── evaluator.py                    # Editted (5), (6)
└── run.py                              # Editted (6)
```

(1) Reuse of [terratorch source code](https://github.com/terrastackai/terratorch/tree/main/terratorch/models/backbones/terramind).

(2) Set `pt_model_path` in `Encoder.download_model`

(3) Refactoring for `./terramind/*`.

(4) Added `VIT_EncoderSAR`.

(5) Added a function `save_to_zarr`

(6) Added `save_pred`.