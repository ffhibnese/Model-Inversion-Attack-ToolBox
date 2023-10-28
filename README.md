# ModelInversionAttackBox
A toolbox for model inversion attacks.

## Environments

(TODO: 配置环境)

## Download checkpoints 

Download pre-trained models from [here](https://drive.google.com/drive/folders/1ko8zAK1j9lTSF8FMvacO8mCKHY9evG9L) and place them in `./checkpoints/`. The structure is shown in `./checkpoints_structure.txt`.

Genforces models will be automatic downloaded when running scripts.

## Run Example

Examples of attack algorithms is in `./dev_scripts/`. 

Example for running PLGMI attack:
```sh
python dev_scripts/plgmi.py
```

Results are generated in `./results`