"""Transforms compatible with 2D log PSD tensors"""
from functools import singledispatchmethod
from typing import Any, Dict, Union

import torch
from PIL import Image
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms._functional_tensor import _blurred_degenerate_image
from torchvision.tv_tensors import BoundingBoxes, Mask


class LogNoise(v2.Transform):
    """
    Transform to add noise to a log-scaled PSD tensor.
    """

    def __init__(self, p: float = 1, noise_power_db: float = -90):
        """


        Arguments
        ---------
        p : float, default 1
            Probability of the psd being noised.
        """
        super().__init__()
        self.p = p
        self.noise_power_db = noise_power_db

    def _apply(self, log_psd: Tensor) -> Tensor:
        """
        Parameters
        ----------
        log_psd : Tensor
            Log PSD to add noise to of shape (HW).


        Returns
        -------
        Tensor
            Noised log PSD of shape (HW).
        """
        if torch.rand(1).item() < self.p:
            noise_power_linear = 10 ** (self.noise_power_db / 10)
            # convert data from dB to linear
            psd_linear = 10 ** (log_psd / 10)
            # add noise in linear domain
            noise = torch.randn_like(psd_linear) * noise_power_linear**0.5
            noisy_psd_linear = psd_linear + noise
            # ensure no negative or zero values before log10
            noisy_psd_linear = torch.clamp(noisy_psd_linear, min=1e-12)
            log_psd = 10 * torch.log10(noisy_psd_linear)
        return log_psd

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_power_db={self.noise_power_db}, p={self.p})"

    @singledispatchmethod
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Default Behavior: Don't modify the input"""
        return inpt

    @_transform.register(torch.Tensor)
    @_transform.register(tv_tensors.Image)
    def _(self, inpt: Union[torch.Tensor, tv_tensors.Image], params: Dict[str, Any]) -> Any:
        """Apply the method to the input tensor"""
        return self._apply(inpt)

    @_transform.register(Image.Image)
    def _(self, inpt: Image.Image, params: Dict[str, Any]) -> Any:
        """Convert the PIL Image to a torch.Tensor to apply the transform"""
        inpt_torch = v2.PILToTensor()(inpt)
        return v2.ToPILImage()(self._transform(inpt_torch, params))

    @_transform.register(BoundingBoxes)
    @_transform.register(Mask)
    def _(self, inpt: Union[BoundingBoxes, Mask], params: Dict[str, Any]) -> Any:
        """Don't modify image annotations"""
        return inpt

    # FIXME: SUPER ANNOYING MONKEYPATCH HERE
    # modern torchvision (>0.20) does not use underscore for transform
    # last version of torchvision for python 3.8 is v0.19.
    # Delete this line if the minimum version of the module becomes >= python 3.9
    transform = _transform
