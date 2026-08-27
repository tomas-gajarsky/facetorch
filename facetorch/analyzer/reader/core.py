"""Public image readers feeding facetorch's canonical input pipeline."""

import http.client
import io
import ipaddress
import os
import queue
import socket
import ssl
import threading
import time
import warnings
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, runtime_checkable
from urllib.parse import urljoin, urlsplit, urlunsplit

import numpy as np
import torch
import torchvision
from codetiming import Timer
from PIL import Image, UnidentifiedImageError

from facetorch.base import BaseReader
from facetorch.datastruct import ImageData
from facetorch.exceptions import FacetorchError, InputCoercionWarning, InputError
from facetorch.input import InputSpec
from facetorch.logger import get_logger

logger = get_logger()

LocalPath = Union[str, os.PathLike]
ImageSource = Union[LocalPath, torch.Tensor, np.ndarray, bytes, Image.Image]
DEFAULT_MAX_DECODED_PIXELS = 16 * 1024 * 1024
_DNS_RESOLUTION_SLOTS = threading.BoundedSemaphore(value=4)


@runtime_checkable
class ReaderProtocol(Protocol):
    """Small public reader extension point used by :class:`FaceAnalyzer`."""

    def run(
        self,
        image_source: ImageSource,
        fix_img_size: bool = False,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        """Decode and canonicalize exactly one source image."""


def _is_remote_reference(value: str) -> bool:
    return "://" in value


def _validate_public_url_target(parsed, deadline: float) -> tuple[str, ...]:
    """Resolve one URL once and return only validated public numeric addresses."""
    if parsed.username is not None or parsed.password is not None:
        raise InputError("Remote image URLs must not contain credentials.")
    hostname = parsed.hostname
    if not hostname:
        raise InputError("Remote image URL has no hostname.")
    try:
        hostname = hostname.encode("idna").decode("ascii")
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except (OSError, ValueError) as exc:
        raise InputError("Remote image hostname could not be resolved safely.") from exc
    if not _DNS_RESOLUTION_SLOTS.acquire(timeout=_remaining_timeout(deadline)):
        raise InputError("Remote image request timed out.")

    resolution = queue.Queue(maxsize=1)

    def resolve_hostname() -> None:
        try:
            resolution.put(
                (
                    socket.getaddrinfo(
                        hostname,
                        port,
                        type=socket.SOCK_STREAM,
                    ),
                    None,
                )
            )
        except Exception as exc:
            resolution.put((None, exc))
        finally:
            _DNS_RESOLUTION_SLOTS.release()

    resolver = threading.Thread(
        target=resolve_hostname,
        name="facetorch-dns",
        daemon=True,
    )
    try:
        resolver.start()
    except RuntimeError:
        _DNS_RESOLUTION_SLOTS.release()
        raise
    try:
        addresses, resolution_error = resolution.get(
            timeout=_remaining_timeout(deadline)
        )
    except queue.Empty as exc:
        raise InputError("Remote image request timed out.") from exc
    if resolution_error is not None:
        if isinstance(resolution_error, (OSError, ValueError)):
            raise InputError(
                "Remote image hostname could not be resolved safely."
            ) from resolution_error
        raise resolution_error
    assert addresses is not None
    if not addresses:
        raise InputError("Remote image hostname did not resolve to an address.")
    validated = []
    for address in addresses:
        raw_address = str(address[4][0]).split("%", 1)[0]
        try:
            resolved = ipaddress.ip_address(raw_address)
        except ValueError as exc:
            raise InputError("Remote image hostname resolved unexpectedly.") from exc
        if (
            not resolved.is_global
            or resolved.is_multicast
            or resolved.is_unspecified
            or resolved.is_loopback
            or resolved.is_link_local
            or resolved.is_reserved
            or getattr(resolved, "is_site_local", False)
        ):
            raise InputError(
                "Remote image URLs must resolve only to public network addresses."
            )
        normalized = str(resolved)
        if normalized not in validated:
            validated.append(normalized)
    return tuple(validated)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose socket target is an already validated numeric IP."""

    def __init__(
        self,
        hostname: str,
        address: str,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout)
        self._validated_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to an IP while retaining hostname TLS checks."""

    def __init__(
        self,
        hostname: str,
        address: str,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(
            hostname,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._validated_address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except Exception:
            raw_socket.close()
            raise


def _open_pinned_response(parsed, address: str, timeout: float):
    """Open one proxy-free request to a validated IP with the original Host/SNI."""
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").encode("idna").decode("ascii")
    default_port = 443 if scheme == "https" else 80
    port = parsed.port or default_port
    connection_type = (
        _PinnedHTTPSConnection if scheme == "https" else _PinnedHTTPConnection
    )
    connection = connection_type(hostname, address, port, timeout)
    display_hostname = f"[{hostname}]" if ":" in hostname else hostname
    host_header = (
        display_hostname if port == default_port else f"{display_hostname}:{port}"
    )
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    try:
        connection.request(
            "GET",
            target,
            headers={
                "Accept": "image/*",
                "Host": host_header,
                "User-Agent": "facetorch-url-reader/1",
            },
        )
        return connection, connection.getresponse()
    except Exception:
        connection.close()
        raise


def _remaining_timeout(deadline: float) -> float:
    """Return the remaining request budget or fail with the public timeout error."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InputError("Remote image request timed out.")
    return remaining


def _set_response_timeout(connection, response, timeout: float) -> None:
    """Apply the shrinking request budget to every reachable response socket."""
    response_file = getattr(response, "fp", None)
    response_raw = getattr(response_file, "raw", response_file)
    sockets = (
        getattr(connection, "sock", None),
        getattr(response_raw, "_sock", None),
    )
    updated = set()
    for response_socket in sockets:
        if response_socket is None or id(response_socket) in updated:
            continue
        response_socket.settimeout(timeout)
        updated.add(id(response_socket))


def _array_to_tensor(array: np.ndarray) -> torch.Tensor:
    if not isinstance(array, np.ndarray):
        raise InputError(f"Expected a NumPy array, got {type(array).__name__}.")
    try:
        return torch.from_numpy(np.ascontiguousarray(array))
    except (TypeError, ValueError) as exc:
        raise InputError(f"Unsupported NumPy image dtype {array.dtype}.") from exc


def _validate_max_decoded_pixels(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError("max_decoded_pixels must be a positive integer.")
    return value


class UniversalReader(BaseReader):
    """Read local paths and in-memory images; network access is intentionally absent."""

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        max_decoded_pixels: int = DEFAULT_MAX_DECODED_PIXELS,
    ):
        super().__init__(transform, device, optimize_transform)
        self.max_decoded_pixels = _validate_max_decoded_pixels(max_decoded_pixels)

    @Timer("UniversalReader.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(
        self,
        image_source: ImageSource,
        fix_img_size: bool = False,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        if isinstance(image_source, (str, os.PathLike)):
            path = os.fspath(image_source)
            if _is_remote_reference(path):
                raise InputError(
                    "Remote image input requires an explicit URLReader configuration."
                )
            return self.read_image_from_path(
                path,
                fix_img_size,
                input_policy=input_policy,
                input_spec=input_spec,
            )
        if isinstance(image_source, torch.Tensor):
            return self.read_tensor(
                image_source,
                fix_img_size,
                input_policy=input_policy,
                input_spec=input_spec,
            )
        if isinstance(image_source, np.ndarray):
            return self.read_numpy_array(
                image_source,
                fix_img_size,
                input_policy=input_policy,
                input_spec=input_spec,
            )
        if isinstance(image_source, bytes):
            return self.read_image_from_bytes(
                image_source,
                fix_img_size,
                input_policy=input_policy,
                input_spec=input_spec,
            )
        if isinstance(image_source, Image.Image):
            return self.read_pil_image(
                image_source,
                fix_img_size,
                input_policy=input_policy,
                input_spec=input_spec,
            )
        raise InputError(
            f"Unsupported image source type {type(image_source).__name__}; expected a "
            "local path, bytes, PIL image, NumPy array, or Torch tensor."
        )

    def read_tensor(
        self,
        tensor: torch.Tensor,
        fix_img_size: bool,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        return self.process_tensor(
            tensor,
            fix_img_size,
            input_policy=input_policy,
            input_spec=input_spec,
            source_kind="torch",
        )

    def read_pil_image(
        self,
        pil_image: Image.Image,
        fix_img_size: bool,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
        path_input: Optional[str] = None,
    ) -> ImageData:
        try:
            width, height = pil_image.size
        except (AttributeError, TypeError, ValueError) as exc:
            raise InputError("Decoded image dimensions are invalid.") from exc
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
        ):
            raise InputError("Decoded image dimensions are invalid.")
        decoded_pixels = width * height
        if width < 1 or height < 1 or decoded_pixels > self.max_decoded_pixels:
            raise InputError(
                f"Decoded image contains {decoded_pixels} pixels; the configured "
                f"limit is {self.max_decoded_pixels}."
            )

        mode = getattr(pil_image, "mode", None)
        source = pil_image
        converted = None
        conversion_message = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                if mode not in {"L", "RGB", "RGBA"}:
                    if str(input_policy).lower().strip() == "strict":
                        raise InputError(
                            f"Strict mode does not accept decoded PIL mode {mode!r}; "
                            "convert it explicitly to L, RGB, or RGBA."
                        )
                    conversion_message = f"Converted decoded PIL mode {mode!r} to RGB."
                    warnings.warn(
                        conversion_message,
                        InputCoercionWarning,
                        stacklevel=3,
                    )
                    converted = pil_image.convert("RGB")
                    source = converted
                array = np.array(source, copy=True)
        except FacetorchError:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InputError("Decoded image exceeds Pillow's safety limit.") from exc
        except (OSError, ValueError) as exc:
            raise InputError(
                "Could not decode or convert the supplied PIL image."
            ) from exc
        finally:
            if converted is not None:
                converted.close()

        data = self.process_tensor(
            _array_to_tensor(array),
            fix_img_size,
            input_policy=input_policy,
            input_spec=input_spec,
            source_kind="decoded",
            path_input=path_input,
        )
        if conversion_message is not None:
            data.warnings.insert(0, conversion_message)
        return data

    def read_numpy_array(
        self,
        array: np.ndarray,
        fix_img_size: bool,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        return self.process_tensor(
            _array_to_tensor(array),
            fix_img_size,
            input_policy=input_policy,
            input_spec=input_spec,
            source_kind="numpy",
        )

    def read_image_from_bytes(
        self,
        image_bytes: bytes,
        fix_img_size: bool,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
        path_input: Optional[str] = None,
    ) -> ImageData:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with io.BytesIO(image_bytes) as buffer, Image.open(buffer) as pil_image:
                    return self.read_pil_image(
                        pil_image,
                        fix_img_size,
                        input_policy=input_policy,
                        input_spec=input_spec,
                        path_input=path_input,
                    )
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InputError("Decoded image exceeds Pillow's safety limit.") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            if isinstance(exc, FacetorchError):
                raise
            raise InputError("The supplied bytes are not a supported image.") from exc

    def read_image_from_path(
        self,
        path_image: str,
        fix_img_size: bool,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path_image) as pil_image:
                    return self.read_pil_image(
                        pil_image,
                        fix_img_size,
                        input_policy=input_policy,
                        input_spec=input_spec,
                        path_input=str(Path(path_image)),
                    )
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise InputError("Decoded image exceeds Pillow's safety limit.") from exc
        except (
            FileNotFoundError,
            PermissionError,
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise InputError(
                f"Could not read local image path {path_image!r}."
            ) from exc

    def read_image_from_url(self, *_args, **_kwargs) -> ImageData:
        """Compatibility guard for callers that previously used implicit networking."""
        raise InputError(
            "Remote image input requires an explicit URLReader configuration."
        )


class ImageReader(BaseReader):
    """Reader restricted to local filesystem paths."""

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        max_decoded_pixels: int = DEFAULT_MAX_DECODED_PIXELS,
    ):
        super().__init__(transform, device, optimize_transform)
        self.max_decoded_pixels = _validate_max_decoded_pixels(max_decoded_pixels)

    read_pil_image = UniversalReader.read_pil_image
    read_image_from_path = UniversalReader.read_image_from_path

    @Timer("ImageReader.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(
        self,
        image_source: LocalPath,
        fix_img_size: bool = False,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        if not isinstance(image_source, (str, os.PathLike)):
            raise InputError(
                f"ImageReader accepts only a local path, got {type(image_source).__name__}."
            )
        path = os.fspath(image_source)
        if _is_remote_reference(path):
            raise InputError(
                "ImageReader does not permit remote URLs; configure URLReader."
            )
        return self.read_image_from_path(
            path,
            fix_img_size,
            input_policy=input_policy,
            input_spec=input_spec,
        )


class TensorReader(BaseReader):
    """Reader restricted to Torch tensors."""

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
    ):
        super().__init__(transform, device, optimize_transform)

    @Timer("TensorReader.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(
        self,
        image_source: torch.Tensor,
        fix_img_size: bool = False,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        if not isinstance(image_source, torch.Tensor):
            raise InputError(
                f"TensorReader accepts only Torch tensors, got "
                f"{type(image_source).__name__}."
            )
        return self.process_tensor(
            image_source,
            fix_img_size,
            input_policy=input_policy,
            input_spec=input_spec,
            source_kind="torch",
        )


class URLReader(UniversalReader):
    """Explicit, bounded HTTP(S) image reader."""

    _REDIRECT_STATUSES = {301, 302, 303, 307, 308}

    def __init__(
        self,
        transform: torchvision.transforms.Compose,
        device: torch.device,
        optimize_transform: bool,
        allowed_schemes: Sequence[str] = ("https",),
        timeout: float = 10.0,
        max_redirects: int = 3,
        max_bytes: int = 10 * 1024 * 1024,
        max_decoded_pixels: int = DEFAULT_MAX_DECODED_PIXELS,
    ):
        super().__init__(
            transform,
            device,
            optimize_transform,
            max_decoded_pixels=max_decoded_pixels,
        )
        if isinstance(allowed_schemes, str):
            allowed_schemes = (allowed_schemes,)
        self.allowed_schemes = tuple(
            str(scheme).lower().strip() for scheme in allowed_schemes
        )
        if not self.allowed_schemes or any(
            scheme not in {"http", "https"} for scheme in self.allowed_schemes
        ):
            raise InputError("allowed_schemes must contain only 'http' and/or 'https'.")
        if timeout <= 0:
            raise InputError("URLReader timeout must be greater than zero.")
        if max_redirects < 0:
            raise InputError("URLReader max_redirects must be non-negative.")
        if max_bytes < 1:
            raise InputError("URLReader max_bytes must be at least one byte.")
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_bytes = max_bytes

    @Timer("URLReader.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(
        self,
        image_source: str,
        fix_img_size: bool = False,
        *,
        input_policy: str = "coerce",
        input_spec: Optional[InputSpec] = None,
    ) -> ImageData:
        if not isinstance(image_source, str):
            raise InputError(
                f"URLReader accepts only a URL string, got {type(image_source).__name__}."
            )

        current_url = image_source
        deadline = time.monotonic() + self.timeout
        for redirect_count in range(self.max_redirects + 1):
            _remaining_timeout(deadline)
            parsed = urlsplit(current_url)
            if parsed.scheme.lower() not in self.allowed_schemes or not parsed.netloc:
                raise InputError("URL scheme is not allowed or the URL has no host.")
            addresses = _validate_public_url_target(parsed, deadline)
            _remaining_timeout(deadline)
            connection = None
            response = None
            last_error = None
            for address in addresses:
                try:
                    connection, response = _open_pinned_response(
                        parsed,
                        address,
                        _remaining_timeout(deadline),
                    )
                    try:
                        _remaining_timeout(deadline)
                    except InputError:
                        response.close()
                        connection.close()
                        raise
                    break
                except InputError:
                    raise
                except TimeoutError as exc:
                    last_error = exc
                    if time.monotonic() >= deadline:
                        raise InputError("Remote image request timed out.") from exc
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    last_error = exc
            if connection is None or response is None:
                if isinstance(last_error, TimeoutError) or time.monotonic() >= deadline:
                    raise InputError("Remote image request timed out.") from last_error
                raise InputError(
                    "Remote image request failed or timed out."
                ) from last_error

            try:
                if response.status in self._REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if location is None:
                        raise InputError("Remote image redirect omitted its target.")
                    if redirect_count >= self.max_redirects:
                        raise InputError("Remote image exceeded the redirect limit.")
                    current_url = urljoin(current_url, location)
                    continue

                if response.status < 200 or response.status >= 300:
                    raise InputError("Remote image returned an unsuccessful response.")
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise InputError(
                            "Remote image returned an invalid Content-Length header."
                        ) from exc
                    if declared_size > self.max_bytes:
                        raise InputError(
                            "Remote image exceeds the configured size limit."
                        )

                chunks = []
                received = 0
                while True:
                    remaining = _remaining_timeout(deadline)
                    _set_response_timeout(connection, response, remaining)
                    chunk = response.read(64 * 1024)
                    _remaining_timeout(deadline)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > self.max_bytes:
                        raise InputError(
                            "Remote image exceeds the configured size limit."
                        )
                    chunks.append(chunk)
            except InputError:
                raise
            except TimeoutError as exc:
                raise InputError("Remote image request timed out.") from exc
            except (OSError, ValueError, http.client.HTTPException) as exc:
                raise InputError(
                    "Remote image returned an unsuccessful response."
                ) from exc
            finally:
                response.close()
                connection.close()

            hostname = parsed.hostname or ""
            if ":" in hostname:
                hostname = f"[{hostname}]"
            safe_netloc = hostname
            if parsed.port is not None:
                safe_netloc = f"{safe_netloc}:{parsed.port}"
            safe_url = urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))
            return self.read_image_from_bytes(
                b"".join(chunks),
                fix_img_size,
                input_policy=input_policy,
                input_spec=input_spec,
                path_input=safe_url,
            )
