# SecretGen: Privacy Recovery on Pre-trained Models via Distribution Discrimination


## Requirements
Python 3.8 or higher
PyTorch 1.8 or higher
```
$ pip install requirements.txt
```


## Performing Attack
stage1.py: Train the generation backbone on public data.
```
$ python stage1.py --name <taret_model_arch> --mask <type_of_mask>
```
Set `bb` to `True` if it's blackbox case, which will use a public model instead of the target model for diversity loss.

stage2.py: Perform attack.
```
$ python stage2.py --name <taret_model_arch> --mask <type_of_mask> --target <method>
```
For the `target` parameter:
- `pii`: PII (whitebox)
- `pii-bb`: PII (blackbox)
- `gmi`: GMI
- `init-bb`: SecretGen (blackbox)
- `full-bb`: SecretGen (blackbox + ground truth label)
- `init-wb`: SecretGen (whitebox)
- `full`: SecretGen (white + ground truth label)

Set `save` to `True` if you want to run evaluation protocol 2, which requires a completely recovered dataset.


## Pre-trained Checkpoints
We release the checkpoints for our VGG16 target model and the corresponding generation backbones at this link:

https://drive.google.com/drive/folders/149LMfBEmhcFr1S2y6PLXf3WqqPA8-We0?usp=sharing
