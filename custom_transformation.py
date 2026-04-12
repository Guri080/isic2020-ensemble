
''' 
Got this code from:
Author: geezah
Link: https://github.com/pytorch/vision/issues/8699
Desc: Feature request on PyTorch with heading:-
'[Feature Request] PadToSquare: Square Padding to Preserve Aspect Ratios When Resizing Images with Varied Shapes in torchvision.transforms.v2
#8699'
'''

from typing import Any, Dict, List, Literal, Union, Type

import torchvision.transforms.v2.functional as F
from torchvision.transforms import v2
from torchvision.transforms.v2._utils import (
    _check_padding_mode_arg,
    _get_fill,
    _setup_fill_arg,
    _FillType,
)


class PadSquare(v2.Transform):
    """Pad a non-square input to make it square by padding the shorter side to match the longer side.
    Args:
        fill (number or tuple or dict, optional): Pixel fill value used when the  ``padding_mode`` is constant.
            Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively.
            Fill value can be also a dictionary mapping data type to the fill value, e.g.
            ``fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}`` where ``Image`` will be filled with 127 and
            ``Mask`` will be filled with 0.
        padding_mode (str, optional): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is "constant".

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Example:
        >>> import torch
        >>> from torchvision.transforms.v2 import PadSquare
        >>> rectangular_image = torch.randint(0, 255, (3, 224, 168), dtype=torch.uint8)
        >>> transform = PadSquare(padding_mode='constant', fill=0)
        >>> square_image = transform(rectangular_image)
        >>> print(square_image.size())
        torch.Size([3, 224, 224])
    """

    def __init__(
        self,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    ):
        super().__init__()

        _check_padding_mode_arg(padding_mode)

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError(
                "`padding_mode` must be one of 'constant', 'edge', 'reflect' or 'symmetric'."
            )
        self.padding_mode = padding_mode
        self.fill = _setup_fill_arg(fill)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # Get the original height and width from the inputs
        orig_height, orig_width = v2.query_size(flat_inputs)

        # Find the target size (maximum of height and width)
        target_size = max(orig_height, orig_width)

        if orig_height < target_size:
            # Need to pad height
            pad_height = target_size - orig_height
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = 0
            pad_right = 0
        else:
            # Need to pad width
            pad_width = target_size - orig_width
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_top = 0
            pad_bottom = 0

        # The padding needs to be in the format [left, top, right, bottom]
        return dict(padding=[pad_left, pad_top, pad_right, pad_bottom])

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill(self.fill, type(inpt))
        return self._call_kernel(
            F.pad,
            inpt,
            padding=params["padding"],
            padding_mode=self.padding_mode,
            fill=fill
        )