import logging
import os
import pickle
from typing import Optional, Union

import brotli
import lz4.block
import lz4.frame
import numpy as np
import PIL
from filelock import BaseFileLock, FileLock, Timeout

COMPRESSED_EXT = ".br"
LZ4_COMPRESSED_EXT = ".lz4"


def endswith(a, end_str):
    if a[-len(end_str) :] == end_str:
        return True
    return False


class CacheElement(object):
    def __init__(
        self, folder, id, ext, should_lock=True, compress_pickle=True, read_only=False, disable_cache: bool = False
    ):
        """Caches objects / images, loads them as needed
        :param folder: Cache folder.
        :param id: Unique ID for that object.
        :param ext: png or pkl, functions as type (image or pkl object) as well.
        :param disable_cache: disable caching
        """
        super().__init__()
        if read_only:
            # Override should_lock if read_only is true
            should_lock = False
        self.should_lock = should_lock
        self.folder = folder
        self.id = id
        self.read_only = read_only
        if not read_only:
            os.makedirs(folder, exist_ok=True)
        self.filename = os.path.join(folder, id + "." + ext)
        self.lockfile = self.filename + ".lock"
        self.image_file_formats = ["png", "jpg", "webp"]
        # Cache file that's been checked for existence
        self._cached_filename = None

        self.ext = ext
        # Use lz4 compression by default, it's on the order of 100x faster than brotli
        self.use_lz4 = True
        self.compress_pickle = compress_pickle
        self.compress_threshold = 20 * 1e6  # 20MB
        self.disable_cache = disable_cache
        assert ext in ["pkl"] + self.image_file_formats

    def _check_exist(self, filename):
        return (not self.disable_cache) and os.path.exists(filename) and os.stat(filename).st_size > 0

    def is_cached(self):
        exist = self._check_exist(self.filename)
        if exist:
            self._cached_filename = self.filename
            return exist
        if self.ext == "pkl":
            # Check for compressed version
            exist = self._check_exist(self.filename + COMPRESSED_EXT)
            if exist:
                self._cached_filename = self.filename + COMPRESSED_EXT
                return exist
            exist = self._check_exist(self.filename + LZ4_COMPRESSED_EXT)
            if exist:
                self._cached_filename = self.filename + LZ4_COMPRESSED_EXT
                return exist
        return False

    def remove_from_cache(self):
        if self.should_lock:
            compressed_filename = self.filename + COMPRESSED_EXT
            if not self.is_cached():
                raise FileNotFoundError(f"Unable to find cache file at {self.filename} or {compressed_filename}")

            if self.ext == "pkl":
                with FileLock(self.lockfile, timeout=60) as lock:
                    os.remove(self._cached_filename)
            else:
                with FileLock(self.lockfile, timeout=60) as lock:
                    os.remove(self.filename)

    def _load_pickle(self, filename, compressed, lz4_format=False):
        if compressed:
            if lz4_format:
                filename += LZ4_COMPRESSED_EXT
            else:
                filename += COMPRESSED_EXT
        try:
            with open(filename, "rb") as fp:
                try:
                    raw_data = fp.read()
                    if compressed:
                        if lz4_format:
                            try:
                                pdata = lz4.frame.decompress(raw_data)
                            except lz4.block.LZ4BlockError as err:
                                logging.warning("Skipping cache {}, failed with an error:\n{}".format(filename, err))
                                return None
                        else:
                            try:
                                pdata = brotli.decompress(raw_data)
                            except brotli.error as err:
                                logging.warning("Skipping cache {}, failed with an error:\n{}".format(filename, err))
                                return None
                    else:
                        pdata = raw_data
                except EOFError as err:
                    logging.warning("Skipping cache {}, failed with an error:\n{}".format(filename, err))
                    raise err

                try:
                    return pickle.loads(pdata)
                except pickle.UnpicklingError as err:
                    logging.warning("Skipping cache {}, failed with an error:\n{}".format(filename, err))
                    return None
        except FileNotFoundError:
            return None

    def _load(self):
        if self.ext == "pkl":

            if not self.is_cached() or self._cached_filename is None:
                return None
            # Try to load both compressed and uncompressed
            if endswith(self._cached_filename, ".pkl"):
                obj = self._load_pickle(self.filename, False)
            elif endswith(self._cached_filename, COMPRESSED_EXT):
                obj = self._load_pickle(self.filename, True)
            elif endswith(self._cached_filename, LZ4_COMPRESSED_EXT):
                obj = self._load_pickle(self.filename, True, lz4_format=True)
            else:
                return None
            return obj
        elif self.ext in self.image_file_formats:
            try:
                with PIL.Image.open(self.filename) as img:
                    img = img.convert("RGB")
                    return img
            except PIL.UnidentifiedImageError:
                return None
            except FileNotFoundError:
                return None
            except EOFError as err:
                print("Failed with an EOFError error when opening {}:\n{}".format(self.filename, err))

    def load(self):
        if self.disable_cache:
            return None
        if self.should_lock:
            with FileLock(self.lockfile, timeout=60):
                return self._load()
        else:
            return self._load()

    def _save_pickle(self, pdata: bytes, compress):
        filename = self.filename
        if compress:
            if self.use_lz4:
                filename += LZ4_COMPRESSED_EXT
            else:
                filename += COMPRESSED_EXT

        with open(filename, "wb") as fp:
            if compress:
                if self.use_lz4:
                    pdata = lz4.frame.compress(pdata, compression_level=3)
                else:
                    pdata = brotli.compress(pdata)
            fp.write(pdata)

    def _save(self, obj):
        if self.ext == "pkl":
            pdata = pickle.dumps(obj)
            if len(pdata) > self.compress_threshold:
                if self.compress_pickle:
                    logging.warning(
                        f"Compressing big pickle file is disabled, filename {self.filename}, file size: {len(pdata)}"
                    )
            if len(pdata) < self.compress_threshold and self.compress_pickle:
                self._save_pickle(pdata, True)
            else:
                self._save_pickle(pdata, False)

        elif self.ext in self.image_file_formats:
            im = PIL.Image.fromarray(np.int8(obj), "RGB")
            im.save(self.filename)
        else:
            raise Exception("Unimplemented file format requested in CacheElement")

    def save(self, obj):
        # Skip saving if cache is disabled.
        if self.disable_cache:
            return
        if self.read_only:
            raise RuntimeError("Cache is read only, can't save.")

        if self.should_lock:
            with FileLock(self.lockfile) as lock:
                self._save(obj)
        else:
            self._save(obj)


class InMemoryCache(object):
    """Caches smaller objects, without multiprocessing problems, currently in memory. Currently, just a dictionary."""

    disable_cache = False

    def __init__(self):
        super().__init__()
        self.cache = dict()

    def remove_from_cache(self, key):
        del self.cache[key]

    def is_cached(self, key):
        return not self.disable_cache and key in self.cache

    def load(self, key):
        return self.cache[key]

    def save(self, obj, key):
        if not self.disable_cache:
            self.cache[key] = obj
