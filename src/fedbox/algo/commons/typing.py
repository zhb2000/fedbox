from typing import Iterable, Union, Any

from torch import Tensor

OptimParams = Union[Iterable[Tensor], Iterable[dict[str, Any]]]
