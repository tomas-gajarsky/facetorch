"""Total network deadlines through the real HTTP/TLS buffering layers."""

import io
import shutil
import socket
import ssl
import subprocess
import threading
import time

import pytest
import torch
from PIL import Image

from facetorch.analyzer.reader import URLReader
from facetorch.analyzer.reader import core as reader_core
from facetorch.exceptions import InputError

pytestmark = pytest.mark.release_blocker


@pytest.fixture(scope="module")
def tls_contexts(tmp_path_factory):
    if not shutil.which("openssl"):
        pytest.skip("openssl is needed for the local TLS fixture")
    root = tmp_path_factory.mktemp("url-tls")
    key, cert = root / "key.pem", root / "cert.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert, key)
    client = ssl.create_default_context(cafile=str(cert))
    return server, client


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize(
    "phase", ["headers", "body", "chunk-size", "chunk-body", "redirect", "complete"]
)
def test_trickle_responses_obey_the_total_deadline(monkeypatch, request, tls, phase):
    server_context = None
    if tls:
        server_context, client_context = request.getfixturevalue("tls_contexts")
        monkeypatch.setattr(
            reader_core.ssl, "create_default_context", lambda: client_context
        )
    image = io.BytesIO()
    Image.new("RGB", (8, 8)).save(image, format="PNG")
    payload = image.getvalue()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    listener.settimeout(2)
    port = listener.getsockname()[1]
    stopped = threading.Event()
    errors = []

    def serve():
        try:
            with listener:
                for number in range(2 if phase == "redirect" else 1):
                    raw, _ = listener.accept()
                    with raw:
                        conn = (
                            server_context.wrap_socket(raw, server_side=True)
                            if tls
                            else raw
                        )
                        with conn:
                            conn.settimeout(2)
                            conn.recv(4096)
                            if phase == "redirect" and number == 0:
                                time.sleep(0.07)
                                conn.sendall(
                                    b"HTTP/1.1 302 Found\r\nLocation: /slow\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                                )
                                continue
                            headers = (
                                f"HTTP/1.1 200 OK\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                            ).encode()
                            if phase == "complete":
                                conn.sendall(headers + payload)
                                return
                            if phase == "headers":
                                delayed = headers + payload
                            elif phase.startswith("chunk"):
                                conn.sendall(
                                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                                )
                                if phase == "chunk-size":
                                    delayed = (
                                        b"1;extension="
                                        + b"x" * 64
                                        + b"\r\nx\r\n0\r\n\r\n"
                                    )
                                else:
                                    conn.sendall(f"{len(payload):x}\r\n".encode())
                                    delayed = payload + b"\r\n0\r\n\r\n"
                            else:
                                conn.sendall(headers)
                                delayed = payload
                            for byte in delayed:
                                conn.sendall(bytes([byte]))
                                time.sleep(0.025)
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
            pass  # Client cancellation must tear down the real connection.
        except Exception as exc:
            errors.append(exc)
        finally:
            stopped.set()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    # Isolate the fixture locally; public address policy has separate coverage.
    # The first numeric address refuses connections, exercising the retry path.
    monkeypatch.setattr(
        reader_core,
        "_validate_public_url_target",
        lambda *_: ("127.0.0.2", "127.0.0.1"),
    )
    scheme = "https" if tls else "http"
    reader = URLReader(
        None, torch.device("cpu"), False, allowed_schemes=(scheme,), timeout=0.2
    )
    started = time.monotonic()
    try:
        if phase == "complete":
            result = reader.run(f"{scheme}://localhost:{port}/image.png")
            assert result.tensor.shape == (1, 3, 8, 8)
        else:
            with pytest.raises(InputError, match="timed out"):
                reader.run(f"{scheme}://localhost:{port}/image.png")
        assert time.monotonic() - started < 0.7
        assert stopped.wait(1), "request cancellation left the server streaming"
        assert not errors
    finally:
        listener.close()
        worker.join(3)


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), True, "1", 0, -1])
def test_url_deadline_configuration_must_be_finite(timeout):
    with pytest.raises(InputError, match="finite"):
        URLReader(None, torch.device("cpu"), False, timeout=timeout)
