from typing import Union, Optional

import torch


class ShapeException(Exception):
    pass


def check_shape(
    tensor: torch.Tensor,
    expect_shape: Union[list[Optional[int]], list[Optional[int]]],
    raise_exception=True,
) -> bool:
    """Check if the shape of the tensor matches expectations.

    Args:
        tensor (torch.Tensor): The tensor to check.
        expect_shape (Union[list[Optional[int]], list[Optional[int]]]): The expected shape.
        raise_exception (bool, optional): Whether to raise an exception. Defaults to True.

    Returns:
        bool: The check result.
    """    
    
    tensor_shape = tensor.shape

    if len(tensor_shape) < len(expect_shape):
        if raise_exception:
            raise ShapeException(
                f'expect ndim >= {len(expect_shape)}, but found {len(tensor_shape)}'
            )
        return False
    # torch.Size().
    tensor_shape_raw = tensor_shape
    tensor_shape = tensor_shape[-len(expect_shape) :]

    for i in range(len(expect_shape)):
        if expect_shape[i] is None:
            continue

        if expect_shape[i] != tensor_shape[i]:
            if raise_exception:
                for j in range(len(expect_shape)):
                    if expect_shape[i] is None:
                        expect_shape[i] = '*'
                expect_shape_str = ', '.join(expect_shape)
                tensor_shape_str = ', '.join(list(tensor_shape_raw))
                raise ShapeException(
                    f'expect shape [..., {expect_shape_str}], but found [{tensor_shape_str}]'
                )

            return False
    return True
