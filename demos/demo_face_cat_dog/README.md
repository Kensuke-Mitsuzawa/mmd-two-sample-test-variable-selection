# Cat and Dog Face Demo

This demo illustrates variable selection between two image distributions (Cat vs Dog faces) using interpretable MMD and Wasserstein independence tests.

## Usage

1. **Dataset Setup**: Download the AFHQ dataset (see the Dataset Setting section below).
2. **Configure**: Update `cat_and_dog_config/cat_and_dog_config.toml` with the dataset paths and hardware options.
3. **Run Assessment**:
   ```bash
   python run_assessment.py [--path_config path/to/config.toml]
   ```
4. **Generate Visualizations**:
   ```bash
   python make_interactive_html.py [--path_config path/to/config.toml]
   ```
   This generates:
   - `workdir_interactive_tool/heatmap.png`: Heatmap of MMD weights across image pixels.
   - `workdir_interactive_tool/masked_images/`: Original and masked images highlighting discrepancy variables.
   - `workdir_interactive_tool/interactive_plotly.html`: Interactive carousel visualization.

Example visual outputs can be found under `./example_output_images`.

---

## Computational Resource Configuration

In `cat_and_dog_config/cat_and_dog_config.toml`, configure the computational resources:

### Concurrent GPU Mode (Default)
Enables multiple worker slots per GPU using NVIDIA MPS and a local Dask cluster:
```toml
[computational_resource]
train_accelerator = "gpu"
distributed_mode = "dask"
k_slots_per_gpu = 2
dask_n_workers = 2
dask_threads_per_worker = 1
dask_scheduler_host = "0.0.0.0"
dask_scheduler_port = 8786
dask_dashboard_address = ":8787"
```

### Single GPU Mode (Sequential)
```toml
[computational_resource]
train_accelerator = "gpu"
distributed_mode = "joblib"
n_workers = 1
```

### CPU Mode
```toml
[computational_resource]
train_accelerator = "cpu"
distributed_mode = "joblib"
n_workers = 4
```

---

## Dataset Setting

You need the AFHQ dataset in your local storage:
1. Follow the instructions on the [AFHQ Github page](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq).
2. Set `path_dir_data_source_x` (e.g. dog faces) and `path_dir_data_source_y` (e.g. cat faces) in `cat_and_dog_config.toml`.
