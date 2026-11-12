# CT MAR

This repository collects shared CT metal artifact reduction code, our proposed model
(PConv-MAR), and the two benchmark implementations, CNN-MAR and DuDoNet:

- `Pconv_mar/`: PyTorch partial-convolution sinogram MAR model.
- `DuDoNet/`: PyTorch reimplementation of DuDoNet (Lin et al., CVPR 2019).
- `cnn_mar/`: TensorFlow/Keras reimplementation of CNN-MAR (Zhang and Yu, IEEE TMI 2018).
- `data_prep/`: builds the case folders, metal traces and the train/validation split.
- `src/`: shared packages used by the models.
- `configs/`: run configurations for training and experiments.

## Environment

Use the repository-level environment for all three models:

```bash
cd CT_MAR
conda env create -f configs/environment.yml
conda activate ct-mar-env
python -m pip install -e .
```

The environment specification includes TensorFlow/Keras for CNN-MAR,
PyTorch/torchvision for PConv-MAR and DuDoNet, CUDA/CuPy/Numba reconstruction
support, shared CT/image dependencies, Optuna and other required packages.
Use this environment for the commands in the model READMEs. PConv-MAR and
DuDoNet training and GPU reconstruction require an NVIDIA CUDA GPU.

## Shared Code

The installable package is defined by `pyproject.toml`.

Important shared modules:

- `src/mar_common/`: case discovery, image I/O, scanner configuration and unit conversion.
- `src/ct_ops/`: forward projection and FDK reconstruction helpers.
- `src/classical_mar/`: classical LI-MAR and NMAR utilities.
- `src/geometry/`: fan-beam geometry helpers.
- `src/metrics/`: NumPy and PyTorch training metrics, plus image and sinogram scoring.

See [METRICS.md](METRICS.md) for the evaluation metrics and training measurements.

## Configs

Main run configs are in `configs/`:

- `configs/dudonet.yaml`
- `configs/pconv.yaml`
- `configs/cnnmar.yaml`

PConv-MAR also has `pconv_holeonly.yaml`, `pconv_hole_img.yaml` and
`pconv_fullstack.yaml` for the model variants.

Before running training, edit the dataset paths, output folder and training settings
in the relevant YAML file.

Training scripts select configs by name:

```text
--config dudonet
--config pconv
--config cnnmar
```

For example, `--config pconv` loads `configs/pconv.yaml`.
`MAR_CONFIG_DIR` can override the config folder. The CNN-MAR command below sets
it explicitly to use the repository-level configs.

## Common Commands

Run each example from the directory containing `CT_MAR/`.

Run PConv-MAR training:

```bash
cd CT_MAR/Pconv_mar
python -m training.trainer --config pconv
```

Run PConv-MAR multi-GPU training:

```bash
cd CT_MAR/Pconv_mar
torchrun --standalone --nproc_per_node=4 -m training.trainer --config pconv
```

Run DuDoNet training:

```bash
cd CT_MAR/DuDoNet
python -m training.trainer --config dudonet
```

Run DuDoNet multi-GPU training:

```bash
cd CT_MAR/DuDoNet
torchrun --standalone --nproc_per_node=4 -m training.trainer --config dudonet
```

Run CNN-MAR training:

```bash
cd CT_MAR/cnn_mar
MAR_CONFIG_DIR=../configs python -m training.trainer --config cnnmar
```

## Data And Outputs

Each case is one folder of ".mha" files, the paths go in "configs/*.yaml":

    dataset_training/
        00001/
            img_bhc.mha
            img_li.mha
            input_img.mha
            ...
        00002/
        ...
    dataset_validation/
        01077/
            (the same files)
        01175/
        ...
    test_images_1000/
        head/   0001 ... 0136
        body/   0137 ... 1000

Training outputs include checkpoints, run configurations and logs. Evaluation
outputs include predictions and metric CSV files. Output locations are selected
through the run config or command-line arguments.

See the model READMEs for model-specific commands:

- [PConv-MAR](Pconv_mar/README.md)
- [DuDoNet](DuDoNet/README.md)
- [CNN-MAR](cnn_mar/README.md)

For test-set inference and the 29 clinical submission images, see:

- [PConv-MAR evaluation](Pconv_mar/evaluation/README.md)
- [DuDoNet evaluation](DuDoNet/evaluation/README.md)
- [CNN-MAR evaluation](cnn_mar/eval/README.md)

The official AAPM clinical scoring program is external to this repository.

## Third-Party Code

- `src/ct_ops/` is from XCIST (General Electric), see the license line in its files.
- `src/geometry/fanbeam.py` and `src/mar_common/scanner_config.py` use the scanner
  geometry of the AAPM CT-MAR challenge code (`AAPMRecon.py`, `AAPMProj.py`,
  https://github.com/xcist/example/tree/main/AAPM_datachallenge/proj_and_recon).
- `src/classical_mar/nmar.py` is a port of the NMAR and LI MATLAB code of
  Zhang and Yu (https://github.com/yanbozhang007/CNN-MAR).
- `cnn_mar/data_pipline/` and `cnn_mar/postprocessing/prior.py` are ports of `dataGen.m`,
  `dataInput.m` and `cnnPrior.m` from the same MATLAB code of Zhang and Yu.
- `Pconv_mar/model/nvidia_partialconv2d.py` is NVIDIA's partial convolution layer,
  under its BSD license (`Pconv_mar/model/NVIDIA_PARTIALCONV_LICENSE.txt`).
