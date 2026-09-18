from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from goldilocks_core.assets.records import AssetFile

_CHUNK_SIZE = 1024 * 1024
_TIMEOUT_SECONDS = 300
_RETRIES = Retry(
    total=3,
    backoff_factor=0.5,
    # 500 is in this list alongside the usual transient codes because PSDI's
    # presigned-S3 file endpoint (data-collections.psdi.ac.uk) intermittently
    # answers a plain GET with a bodyless "500 UnknownError" (reproduced
    # directly against S3 with plain curl, independent of this client).
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    raise_on_status=False,
)
# A retry within one request reuses the exact signed URL a redirect already
# resolved to -- fine for an ordinary transient blip, but PSDI's failure mode
# above sometimes keeps failing on that *specific* signed URL while a fresh
# request (a new redirect hop, thus a new signature) succeeds (observed
# directly, 2026-09-18: same file, alternating success/500 across separate
# requests). This retries the whole fetch, not just the final hop.
_SOURCE_ATTEMPTS = 3
_SOURCE_RETRY_BACKOFF_SECONDS = 1.0


class ChecksumMismatch(ValueError):
    pass


def _session() -> requests.Session:
    """Return a one-shot session with a transient-failure retry policy."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRIES)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download(file: AssetFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(file.url)
    if parsed.scheme == "file":
        with Path(parsed.path).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=_CHUNK_SIZE)
    else:
        _fetch_remote(file, destination)
    verify_source(file, destination)


def _fetch_remote(file: AssetFile, destination: Path) -> None:
    for attempt in range(_SOURCE_ATTEMPTS):
        destination.unlink(missing_ok=True)
        try:
            with (
                _session().get(
                    file.url, stream=True, timeout=_TIMEOUT_SECONDS
                ) as response,
                destination.open("xb") as target,
            ):
                response.raise_for_status()
                for chunk in response.iter_content(_CHUNK_SIZE):
                    target.write(chunk)
            return
        except requests.RequestException:
            if attempt == _SOURCE_ATTEMPTS - 1:
                raise
            time.sleep(_SOURCE_RETRY_BACKOFF_SECONDS * (attempt + 1))


def verify_source(file: AssetFile, path: Path) -> None:
    size = path.stat().st_size
    if file.size is not None and size != file.size:
        raise ChecksumMismatch(
            f"{file.role} size mismatch: expected {file.size}, downloaded {size}"
        )
    if file.checksum is None:
        return
    algorithm, separator, expected = file.checksum.partition(":")
    if not separator or not expected:
        raise ValueError(f"checksum must be '<algorithm>:<digest>': {file.checksum!r}")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as error:
        raise ValueError(f"unsupported checksum algorithm: {algorithm}") from error
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ChecksumMismatch(
            f"{file.role} checksum mismatch: expected {expected}, downloaded {actual}"
        )
