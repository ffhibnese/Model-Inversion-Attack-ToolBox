# Steps for adding models.

1. Inherit `BaseTargetModel` class in `base.py` and implement `get_feature_dim` method, which return the dim of the last feature.
2. Resize inputs at the beginning of `forward`.
3. The return of `forward` function should be an instance of `ModelResult` class in `modelresult.py`.
    + result: the output of the model.
    + feat: a list of some features during the `forward` function. The input of the last linear layer (the last feature) should be contained in the list.
    + addition_info: a dict of some other infos.
4. Add the model to `get_model` function in `get_models.py`