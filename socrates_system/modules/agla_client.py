"""
AGLA API Client

Thin HTTP client for calling a remote AGLA verification service (e.g., deployed via Modal).
The remote service is expected to implement the FastAPI defined in the AGLA repo's api_server.py:
  POST /verify  (multipart/form-data)
    - image: file
    - claim: str
    - use_agla: bool (optional)
    - alpha: float (optional)
    - beta: float (optional)
    - return_debug: bool (optional)

We additionally send 'socratic_question' for context; the remote API may ignore it.
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional, Union

import requests
from PIL import Image

try:
    from ..utils.logger import setup_logger  # type: ignore
    logger = setup_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class AGLAClient:
    """HTTP client for a remote AGLA (Adaptive Grounding and Language Alignment) service.

    The remote service is expected to expose a FastAPI endpoint that accepts
    a multipart/form-data POST with an image file and claim text and returns
    a JSON body containing at minimum a ``"verdict"`` field.

    Attributes:
        base_url: Base URL of the remote service (trailing slash stripped).
        verify_path: Endpoint path for claim verification.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, base_url: str, verify_path: str = "/verify", timeout: int = 120) -> None:
        """Initialize the AGLAClient.

        Args:
            base_url: Base URL of the remote AGLA service
                (e.g. ``"https://myapp.modal.run"``).
            verify_path: Path to the verification endpoint.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.verify_path = verify_path
        self.timeout = timeout

    def _coerce_image_bytes(self, image: Union[str, bytes, bytearray, Image.Image]) -> bytes:
        """Convert a variety of image representations to raw JPEG bytes.

        Args:
            image: One of: a local file path (``str``), an HTTP/HTTPS URL
                (``str``), raw ``bytes`` / ``bytearray``, or a
                ``PIL.Image.Image`` instance.

        Returns:
            JPEG-encoded bytes ready for multipart upload.

        Raises:
            TypeError: If ``image`` is none of the supported types.
            requests.HTTPError: If a URL cannot be fetched.
            FileNotFoundError: If a local path does not exist.
        """
        if isinstance(image, (bytes, bytearray)):
            return bytes(image)
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                resp = requests.get(image, timeout=self.timeout)
                resp.raise_for_status()
                return resp.content
            with open(image, "rb") as f:
                return f.read()
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG")
            return buf.getvalue()
        raise TypeError("Unsupported image type. Provide a file path, raw bytes, or a PIL.Image.Image.")

    def verify(
        self,
        image: Union[str, bytes, bytearray, Image.Image],
        claim: str,
        socratic_question: Optional[str] = None,
        use_agla: Optional[bool] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """Send a claim and image to the remote AGLA service for verification.

        Args:
            image: Image to verify against (path, URL, bytes, or PIL Image).
            claim: Textual claim to verify.
            socratic_question: Optional Socratic question for additional
                context; forwarded to the service but may be ignored.
            use_agla: Whether to use the AGLA contrastive decoding method
                on the server side; ``None`` means use the server default.
            alpha: AGLA alpha hyperparameter; ``None`` uses server default.
            beta: AGLA beta hyperparameter; ``None`` uses server default.
            return_debug: Whether to request debug information in the
                response body.

        Returns:
            Parsed JSON response dict from the remote service, including at
            minimum a ``"verdict"`` key (``"True"``, ``"False"``, or
            ``"Uncertain"``).

        Raises:
            requests.HTTPError: If the remote service returns a non-2xx
                status code.
        """
        url = f"{self.base_url}{self.verify_path}"
        img_bytes = self._coerce_image_bytes(image)

        data: Dict[str, Any] = {
            "claim": claim,
            "return_debug": str(bool(return_debug)).lower(),
        }
        if use_agla is not None:
            data["use_agla"] = str(bool(use_agla)).lower()
        if alpha is not None:
            data["alpha"] = str(alpha)
        if beta is not None:
            data["beta"] = str(beta)
        if socratic_question:
            data["socratic_question"] = socratic_question

        files = {"image": ("image.jpg", img_bytes, "image/jpeg")}

        logger.info(f"Calling AGLA API at {url}...")
        resp = requests.post(url, data=data, files=files, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
