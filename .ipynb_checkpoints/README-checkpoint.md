# tli-crm-segmentation
customer segmentation

s3://your-ml-bucket/
├── raw/                # Raw input data (as received, no changes)
│   ├── source1/
│   ├── source2/
│   └── ...
├── processed/          # Cleaned/transformed data (ready for training)
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── feature-store/  # Sometimes features are versioned separately
├── models/             # Saved trained models
│   ├── model-name/
│   │   ├── version1/
│   │   ├── version2/
│   │   └── ...
├── predictions/        # Batch predictions or inference outputs
├── logs/               # Logs of pipeline runs
├── metadata/           # Pipelines, configs, schema, etc.
└── tmp/                # Temporary staging or intermediate files


[Training Pipeline]
    └── train model
    └── evaluate
    └── save model + scaler
    └── log metadata to registry (CSV/DB)
    └── optionally mark as 'selected'

[Serving Pipeline]
    └── read metadata registry
    └── find model with status = 'selected'
    └── load model + scaler
    └── predict