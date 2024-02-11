import io
import requests
from PIL import Image
import numpy as np
import torch
import torchvision
from codetiming import Timer
from typing import Union
from facetorch.base import BaseReader
from facetorch.datastruct import ImageData
from facetorch.logger import LoggerJsonFile

logger = LoggerJsonFile().logger


class UniversalReader(BaseReader):
    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """UniversalReader can read images from a path, URL, tensor, numpy array, bytes or PIL Image and return an ImageData object containing the image tensor.

        Args:
            transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size.

        """
        super().__init__(transform, device, optimize_transform)

    @Timer("UniversalReader.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(
        self,
        image_source: Union[str, torch.Tensor, np.ndarray, bytes, Image.Image],
        fix_img_size: bool = False,
    ) -> ImageData:
        """Reads an image from a path, URL, tensor, numpy array, bytes or PIL Image and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image.

        Args:
            image_source (Union[str, torch.Tensor, np.ndarray, bytes, Image.Image]): Image source to be read.
            fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.

        Returns:
            ImageData: ImageData object with image tensor and pil Image.
        """
        if isinstance(image_source, str):
            if image_source.startswith("http"):
                return self.read_image_from_url(image_source, fix_img_size)
            else:
                return self.read_image_from_path(image_source, fix_img_size)
        elif isinstance(image_source, torch.Tensor):
            return self.read_tensor(image_source, fix_img_size)
        elif isinstance(image_source, np.ndarray):
            return self.read_numpy_array(image_source, fix_img_size)
        elif isinstance(image_source, bytes):
            return self.read_image_from_bytes(image_source, fix_img_size)
        elif isinstance(image_source, Image.Image):
            return self.read_pil_image(image_source, fix_img_size)
        else:
            raise ValueError("Unsupported data type")

    def read_tensor(self, tensor: torch.Tensor, fix_img_size: bool) -> ImageData:
        return self.process_tensor(tensor, fix_img_size)

    def read_pil_image(self, pil_image: Image.Image, fix_img_size: bool) -> ImageData:
        tensor = torchvision.transforms.functional.to_tensor(pil_image)
        return self.process_tensor(tensor, fix_img_size)

    def read_numpy_array(self, array: np.ndarray, fix_img_size: bool) -> ImageData:
        pil_image = Image.fromarray(array, mode="RGB")
        return self.read_pil_image(pil_image, fix_img_size)

    def read_image_from_bytes(
        self, image_bytes: bytes, fix_img_size: bool
    ) -> ImageData:
        pil_image = Image.open(io.BytesIO(image_bytes))
        return self.read_pil_image(pil_image, fix_img_size)

    def read_image_from_path(self, path_image: str, fix_img_size: bool) -> ImageData:
        try:
            image_tensor = torchvision.io.read_image(path_image)
        except Exception as e:
            logger.error(f"Failed to read image from path {path_image}: {e}")
            raise ValueError(f"Could not read image from path {path_image}: {e}") from e

        return self.process_tensor(image_tensor, fix_img_size)

    def read_image_from_url(self, url: str, fix_img_size: bool) -> ImageData:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch image from URL {url}: {e}")
            raise ValueError(f"Could not fetch image from URL {url}: {e}") from e

        image_bytes = response.content
        return self.read_image_from_bytes(image_bytes, fix_img_size)


class ImageReader(BaseReader):
    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """ImageReader is a wrapper around a functionality for reading images by Torchvision.

        Args:
            transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size.

        """
        super().__init__(
            transform,
            device,
            optimize_transform,
        )

    @Timer("ImageReader.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, path_image: str, fix_img_size: bool = False) -> ImageData:
        """Reads an image from a path and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image.

        Args:
            path_image (str): Path to the image.
            fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.

        Returns:
            ImageData: ImageData object with image tensor and pil Image.
        """
        data = ImageData(path_input=path_image)
        data.img = torchvision.io.read_image(
            data.path_input, mode=torchvision.io.ImageReadMode.RGB
        )
        data.img = data.img.unsqueeze(0)
        data.img = data.img.to(self.device)

        if fix_img_size:
            data.img = self.transform(data.img)

        data.tensor = data.img.type(torch.float32)
        data.img = data.img.squeeze(0).cpu()
        data.set_dims()

        return data


class TensorReader(BaseReader):
    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        """TensorReader is a wrapper around a functionality for reading tensors by Torchvision.

        Args:
            transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True.
            device (torch.device): Torch device cpu or cuda object.
            optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size.

        """
        super().__init__(
            transform,
            device,
            optimize_transform,
        )

    @Timer("TensorReader.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self, tensor: torch.Tensor, fix_img_size: bool = False) -> ImageData:
        """Reads a tensor and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image.

        Args:
            tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width).
            fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.

        Returns:
            ImageData: ImageData object with image tensor and pil Image.
        """
        return self.process_tensor(tensor, fix_img_size)
