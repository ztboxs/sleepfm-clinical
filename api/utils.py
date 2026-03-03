import os
import tempfile
import shutil
from contextlib import contextmanager
from urllib.parse import urlparse, unquote

import requests as _requests
from loguru import logger


@contextmanager
def temp_directory(prefix: str = "sleepfm_"):
    """Create a temporary directory that is automatically cleaned up."""
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def save_upload_file(upload_file, dest_path: str, chunk_size: int = 1024 * 1024):
    """Save an UploadFile to disk in chunks to avoid loading entire file into memory."""
    size = 0
    with open(dest_path, "wb") as f:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            size += len(chunk)
    logger.info(f"Saved upload file: {size / 1024 / 1024:.1f} MB -> {dest_path}")


def download_file_from_url(url: str, dest_dir: str, timeout: int = 600) -> str:
    """Download a file from URL to dest_dir. Returns the local file path."""
    logger.info(f"Downloading file from URL: {url}")

    resp = _requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()

    filename = _extract_filename(url, resp)
    dest_path = os.path.join(dest_dir, filename)

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    file_size = os.path.getsize(dest_path)
    logger.info(f"Downloaded {file_size / 1024 / 1024:.1f} MB -> {dest_path}")
    return dest_path


def _extract_filename(url: str, resp: _requests.Response) -> str:
    """Try to extract a meaningful filename from response headers or URL."""
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        parts = cd.split("filename=")[-1].strip().strip('"').strip("'")
        if parts:
            return parts

    path = urlparse(url).path
    name = unquote(os.path.basename(path))
    if name and "." in name:
        return name

    return "downloaded.edf"
