import hashlib
import os
import shutil
import sys
import tempfile

from urllib.request import urlopen, Request
from tqdm import tqdm


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    try:
        u = urlopen(req)
    except Exception as e:
        with open('to_download.txt', 'a') as f:
            f.write(f'url: {url}\t file path: {dst}')
        raise e
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(
            total=file_size,
            disable=not progress,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(
                    'invalid hash value (expected "{}", got "{}")'.format(
                        hash_prefix, digest
                    )
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
