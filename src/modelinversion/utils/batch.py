from typing import Callable, Optional

import torch
from tqdm import tqdm

from .io import print_split_line
from .outputs import BaseOutput


def _is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def _gather(outputs, dim=0):
    """Gather the input data.

    Args:
        outputs (_type_): The data to gather.
        dim (int, optional): The specified dimension used when the type of input data is torch.Tensor. Defaults to 0.
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return torch.cat(outputs, dim=dim)
        if isinstance(out, str):
            return outputs
        if out is None:
            return None
        if isinstance(out, BaseOutput):
            # print((out.keys()))
            # exit()
            return type(out)(*gather_map([d.to_tuple() for d in outputs]))

        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        if _is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


def batch_apply(
    fn: Callable,
    *inputs,
    batch_size: int,
    description: Optional[str] = None,
    use_tqdm: bool = False,
    **other_input_kwargs,
):
    """Apply the given function to input data by the specified batch size.

    Args:
        fn (Callable): The given function.
        *inputs: The collected input data.
        batch_size (int): The specified batch size.
        description (Optional[str], optional): The content to print when processing the input data. Defaults to None.
        use_tqdm (bool, optional): Determine whether to use tqdm when printing. Defaults to False.
    """

    def _check_valid(inputs):
        if len(inputs) == 0:
            return
        lens = []
        for i, inp in enumerate(inputs):
            try:
                lens.append(len(inp))
            except:
                raise RuntimeError(f'the {i} inputs have no attr `len`')
        valid_len = lens[0]
        if not all(map(lambda x: x == valid_len, lens)):
            raise RuntimeError('lengths of all inputs are not the same')

    _check_valid(inputs)

    total_len = len(inputs[0])

    results = []
    starts = list(range(0, total_len, batch_size))
    iter_times = len(starts)

    if use_tqdm:
        if description is not None:
            print_split_line(description)
        starts = tqdm(starts, leave=False)

    for i, start in enumerate(starts, start=1):

        if description is not None and not use_tqdm:
            print_split_line(f'{description}: {i} / {iter_times}')

        end = min(total_len, start + batch_size)
        res = fn(*[p[start:end] for p in inputs], **other_input_kwargs)
        # print(res.device)
        results.append(res)
    return _gather(results)
