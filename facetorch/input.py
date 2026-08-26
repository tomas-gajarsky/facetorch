"""Canonical public image-input contract."""

from dataclasses import dataclass
from typing import List, Literal, Optional
import warnings

import torch

from facetorch.exceptions import InputCoercionWarning, InputError


InputPolicy = Literal["coerce", "strict"]
InputLayout = Literal["HW", "CHW", "HWC", "BCHW", "BHWC"]
InputValueRange = Literal["0_1", "0_255"]
InputColorSpace = Literal["GRAY", "RGB", "BGR", "RGBA"]
AlphaMode = Literal["drop"]


@dataclass(frozen=True)
class InputSpec:
    """Explicitly describes an array or tensor image representation.

    Fields may be omitted in ``coerce`` mode. In ``strict`` mode, callers must
    declare any representation that differs from the source-specific exact
    defaults: uint8 RGB, CHW for Torch, and HWC for NumPy.
    """

    layout: Optional[InputLayout] = None
    value_range: Optional[InputValueRange] = None
    color_space: Optional[InputColorSpace] = None
    alpha_mode: Optional[AlphaMode] = None

    def __post_init__(self):
        normalized = {}
        for field_name, value, operation in (
            ("layout", self.layout, str.upper),
            ("value_range", self.value_range, str.lower),
            ("color_space", self.color_space, str.upper),
            ("alpha_mode", self.alpha_mode, str.lower),
        ):
            if value is not None and not isinstance(value, str):
                raise InputError(
                    f"InputSpec.{field_name} must be a string or None, "
                    f"got {type(value).__name__}."
                )
            normalized[field_name] = operation(value) if value is not None else None
        for field_name, value in normalized.items():
            object.__setattr__(self, field_name, value)

        valid_values = {
            "layout": {"HW", "CHW", "HWC", "BCHW", "BHWC"},
            "value_range": {"0_1", "0_255"},
            "color_space": {"GRAY", "RGB", "BGR", "RGBA"},
            "alpha_mode": {"drop"},
        }
        for field_name, allowed in valid_values.items():
            value = getattr(self, field_name)
            if value is not None and value not in allowed:
                choices = ", ".join(sorted(allowed))
                raise InputError(
                    f"Invalid InputSpec.{field_name}={value!r}; expected one of {choices}."
                )


@dataclass
class CanonicalImage:
    """Internal RGB image representation produced at the public boundary."""

    tensor: torch.Tensor
    warnings: List[str]


def _normalize_policy(input_policy: str) -> InputPolicy:
    policy = str(input_policy).lower().strip()
    if policy not in {"coerce", "strict"}:
        raise InputError(
            f"Unknown input_policy={input_policy!r}; expected 'coerce' or 'strict'."
        )
    return policy  # type: ignore[return-value]


def _default_layout(tensor: torch.Tensor, source_kind: str) -> InputLayout:
    if tensor.ndim == 2:
        return "HW"
    if tensor.ndim == 3:
        return "HWC" if source_kind in {"numpy", "decoded"} else "CHW"
    if tensor.ndim == 4:
        return "BHWC" if source_kind == "numpy" else "BCHW"
    raise InputError(
        f"Unsupported image rank {tensor.ndim}; expected a 2D, 3D, or 4D image."
    )


def _to_bchw(tensor: torch.Tensor, layout: InputLayout) -> torch.Tensor:
    expected_ranks = {
        "HW": 2,
        "CHW": 3,
        "HWC": 3,
        "BCHW": 4,
        "BHWC": 4,
    }
    expected_rank = expected_ranks[layout]
    if tensor.ndim != expected_rank:
        raise InputError(
            f"InputSpec layout {layout} requires rank {expected_rank}, "
            f"but the input has rank {tensor.ndim}."
        )

    if layout == "HW":
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif layout == "CHW":
        tensor = tensor.unsqueeze(0)
    elif layout == "HWC":
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif layout == "BHWC":
        tensor = tensor.permute(0, 3, 1, 2)

    if tensor.shape[0] != 1:
        raise InputError(
            "Batched image input is not supported. "
            f"Expected exactly one source image (B=1), got B={tensor.shape[0]}."
        )
    if tensor.shape[-2] < 1 or tensor.shape[-1] < 1:
        raise InputError(
            f"Image spatial dimensions must be positive, got {tuple(tensor.shape[-2:])}."
        )
    return tensor.contiguous()


def _coercion(messages: List[str], policy: InputPolicy, message: str) -> None:
    if policy == "coerce":
        messages.append(message)
        warnings.warn(message, InputCoercionWarning, stacklevel=4)


def canonicalize_image_tensor(
    tensor: torch.Tensor,
    *,
    source_kind: str,
    input_policy: str = "coerce",
    input_spec: Optional[InputSpec] = None,
) -> CanonicalImage:
    """Convert one image to RGB float32 BCHW ``0..255`` deterministically."""

    policy = _normalize_policy(input_policy)
    if input_spec is not None and not isinstance(input_spec, InputSpec):
        raise InputError(
            f"input_spec must be an InputSpec or None, got "
            f"{type(input_spec).__name__}."
        )
    spec = input_spec or InputSpec()
    messages: List[str] = []

    if not isinstance(tensor, torch.Tensor):
        raise InputError(f"Expected a Torch tensor, got {type(tensor).__name__}.")
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise InputError(f"Unsupported image dtype {tensor.dtype}.")

    layout = spec.layout or _default_layout(tensor, source_kind)
    if (
        spec.layout is None
        and source_kind == "torch"
        and tensor.ndim in {3, 4}
        and int(tensor.shape[-1]) in {1, 3, 4}
        and int(tensor.shape[1] if tensor.ndim == 4 else tensor.shape[0])
        not in {1, 3, 4}
    ):
        suggested_layout = "BHWC" if tensor.ndim == 4 else "HWC"
        raise InputError(
            f"Torch input shape {tuple(tensor.shape)} looks like {suggested_layout}; "
            f'pass InputSpec(layout="{suggested_layout}") explicitly.'
        )
    tensor = _to_bchw(tensor, layout)

    channels = int(tensor.shape[1])
    inferred_color = {1: "GRAY", 3: "RGB", 4: "RGBA"}.get(channels)
    if inferred_color is None:
        raise InputError(
            f"Unsupported channel count {channels}; expected 1, 3, or 4 channels."
        )

    color_space = spec.color_space
    if color_space is None:
        if policy == "strict" and inferred_color != "RGB":
            raise InputError(
                f"Strict mode requires InputSpec.color_space for {inferred_color} input."
            )
        color_space = inferred_color

    expected_channels = {"GRAY": 1, "RGB": 3, "BGR": 3, "RGBA": 4}
    if channels != expected_channels[color_space]:
        raise InputError(
            f"InputSpec color_space {color_space} requires "
            f"{expected_channels[color_space]} channels, got {channels}."
        )
    if spec.alpha_mode is not None and color_space != "RGBA":
        raise InputError("InputSpec.alpha_mode is valid only for RGBA input.")

    numeric = tensor.detach().clone()
    # Preserve float64 precision at the range boundary. Other real dtypes use
    # float32: every integer in the accepted 0..255 interval is represented
    # exactly, and the conversion also supports unsigned dtypes whose native
    # reductions are unavailable in PyTorch.
    bounds = numeric if numeric.dtype == torch.float64 else numeric.to(torch.float32)
    if not torch.isfinite(bounds).all():
        raise InputError("Image values must be finite; NaN and Inf are not supported.")
    minimum = float(bounds.min().item())
    maximum = float(bounds.max().item())
    del bounds

    value_range = spec.value_range
    if value_range is None:
        if policy == "strict":
            if numeric.dtype != torch.uint8:
                raise InputError(
                    "Strict tensor/array input defaults to uint8 0..255. "
                    "Declare InputSpec.value_range for any other dtype or range."
                )
            value_range = "0_255"
        elif numeric.is_floating_point():
            if 0.0 <= minimum and maximum <= 1.0:
                value_range = "0_1"
                _coercion(
                    messages,
                    policy,
                    "Interpreted floating-point image values as 0..1 and scaled to 0..255.",
                )
            elif 0.0 <= minimum and maximum <= 255.0:
                value_range = "0_255"
                _coercion(
                    messages,
                    policy,
                    "Interpreted floating-point image values as 0..255.",
                )
            else:
                raise InputError(
                    f"Floating-point image range [{minimum}, {maximum}] is unsupported; "
                    "expected 0..1 or 0..255."
                )
        else:
            value_range = "0_255"
            if numeric.dtype != torch.uint8:
                _coercion(
                    messages,
                    policy,
                    f"Converted integer image dtype {numeric.dtype} to float32.",
                )

    if value_range == "0_1":
        if minimum < 0.0 or maximum > 1.0:
            raise InputError(
                f"InputSpec declares 0..1 but observed range [{minimum}, {maximum}]."
            )
        numeric = numeric.to(torch.float32) * 255.0
    else:
        if minimum < 0.0 or maximum > 255.0:
            raise InputError(
                f"InputSpec declares 0..255 but observed range [{minimum}, {maximum}]."
            )
        numeric = numeric.to(torch.float32)

    if color_space == "GRAY":
        _coercion(messages, policy, "Expanded grayscale input to three RGB channels.")
        numeric = numeric.repeat(1, 3, 1, 1)
    elif color_space == "BGR":
        _coercion(messages, policy, "Converted explicitly declared BGR input to RGB.")
        numeric = numeric[:, [2, 1, 0], :, :]
    elif color_space == "RGBA":
        if policy == "strict" and spec.alpha_mode is None:
            raise InputError(
                "Strict RGBA input requires an explicit InputSpec.alpha_mode."
            )
        alpha_mode = spec.alpha_mode or "drop"
        if alpha_mode != "drop":  # guarded by InputSpec; defensive for type checkers
            raise InputError(f"Unsupported alpha mode {alpha_mode!r}.")
        _coercion(messages, policy, "Dropped the alpha channel from RGBA input.")
        numeric = numeric[:, :3, :, :]

    return CanonicalImage(tensor=numeric.contiguous(), warnings=messages)
