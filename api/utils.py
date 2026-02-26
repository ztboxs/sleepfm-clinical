import os
import tempfile
import shutil
from contextlib import contextmanager


@contextmanager
def temp_directory(prefix: str = "sleepfm_"):
    """Create a temporary directory that is automatically cleaned up."""
    tmp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def save_upload_file(upload_file, dest_path: str):
    """Save an UploadFile to a given path."""
    with open(dest_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)
