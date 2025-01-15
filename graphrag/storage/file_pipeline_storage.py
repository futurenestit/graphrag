# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'FileStorage' and 'FilePipelineStorage' models."""

import logging
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import boto3
import pandas as pd
import aiofiles
from aiofiles.os import remove
from aiofiles.ospath import exists
from botocore.config import Config
from botocore.exceptions import ClientError
from django.conf import settings

from graphrag.logger.base import ProgressLogger
from graphrag.logger.progress import Progress
from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)


class FilePipelineStorage(PipelineStorage):
    """File storage class definition."""

    _root_dir: str
    _encoding: str

    def __init__(self, root_dir: str = "", encoding: str = "utf-8"):
        """Init method definition."""
        self._root_dir = root_dir.replace("\\", "/")
        self._encoding = encoding
        # Path(self._root_dir).mkdir(parents=True, exist_ok=True)  # noqa: ERA001
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=settings.AWS_S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
            region_name=settings.AWS_S3_REGION_NAME,
        )
        self.bucket_name = settings.AWS_STORAGE_BUCKET_NAME

    def find(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        progress: ProgressLogger | None = None,
        file_filter: dict[str, Any] | None = None,
        max_count=-1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find files in the storage using a file pattern, as well as a custom filter function."""

        def item_filter(item: dict[str, Any]) -> bool:
            if file_filter is None:
                return True
            return all(re.match(value, item[key]) for key, value in file_filter.items())

        search_path = Path(self._root_dir) / (base_dir or "")
        log.info("search %s for files matching %s", search_path, file_pattern.pattern)
        all_files = list(search_path.rglob("**/*"))
        num_loaded = 0
        num_total = len(all_files)
        num_filtered = 0
        for file in all_files:
            match = file_pattern.match(f"{file}")
            if match:
                group = match.groupdict()
                if item_filter(group):
                    filename = f"{file}".replace(self._root_dir, "")
                    if filename.startswith(os.sep):
                        filename = filename[1:]
                    yield (filename, group)
                    num_loaded += 1
                    if max_count > 0 and num_loaded >= max_count:
                        break
                else:
                    num_filtered += 1
            else:
                num_filtered += 1
            if progress is not None:
                progress(_create_progress_status(num_loaded, num_filtered, num_total))

    def find_s3(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        progress: Any
        | None = None,  # Replace `ProgressLogger` with your custom logger if needed
        file_filter: dict[str, Any] | None = None,
        max_count=-1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find files in S3 storage using a file pattern and optional filters."""

        def item_filter(item: dict[str, Any]) -> bool:
            if file_filter is None:
                return True
            return all(re.match(value, item[key]) for key, value in file_filter.items())

        search_prefix = f"{self._root_dir}/" if self._root_dir else ""

        log.info(
            "Searching S3 bucket '%s' with prefix '%s' for files matching pattern '%s'",
            self.bucket_name,
            search_prefix,
            file_pattern.pattern,
        )

        num_loaded = 0
        num_filtered = 0

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name, Prefix=search_prefix
            )

            for page in page_iterator:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    match = file_pattern.match(key)
                    if match:
                        group = match.groupdict()
                        if item_filter(group):
                            filename = key[len(search_prefix) :]
                            yield (filename, group)
                            num_loaded += 1
                            if max_count > 0 and num_loaded >= max_count:
                                return
                        else:
                            num_filtered += 1
                    else:
                        num_filtered += 1

                    if progress is not None:
                        progress({
                            "num_loaded": num_loaded,
                            "num_filtered": num_filtered,
                            "current_file": key,
                        })

        except ClientError:
            log.exception("Failed to list objects in S3")
            raise


    async def get(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """Get method definition."""
        file_path = join_path(self._root_dir, key)

        if await self.has(key):
            return await self._read_file_s3(file_path, as_bytes, encoding)
        if await exists(key):
            # Lookup for key, as it is pressumably a new file loaded from inputs
            # and not yet written to storage
            return await self._read_file_s3(key, as_bytes, encoding)

        return None

    async def get_s3(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """
        Get method for S3-based storage.

        Args:
            key (str): The key of the file in S3.
            as_bytes (bool): Whether to return the file as bytes.
            encoding (str): The encoding to use when returning the file as a string.

        Returns
        -------
            Any: The file content or None if the file doesn't exist.
        """
        s3_key = f"{self._root_dir}/{key}".strip("/")  # Construct the S3 key

        try:
            # Check if the file exists in S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response["Body"].read()

            if as_bytes:
                return content  # Return as bytes
            return content.decode(
                encoding or "utf-8"
            )  # Decode with the specified encoding

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                # File not found
                return None
            raise  # Re-raise other exceptions



    async def _read_file(
        self,
        path: str | Path,
        as_bytes: bool | None = False,
        encoding: str | None = None,
    ) -> Any:
        """Read the contents of a file."""
        read_type = "rb" if as_bytes else "r"
        encoding = None if as_bytes else (encoding or self._encoding)

        async with aiofiles.open(
            path,
            cast("Any", read_type),
            encoding=encoding,
        ) as f:
            return await f.read()

    async def _read_file_s3(
        self,
        path: str | Path,
        as_bytes: bool | None = False,
        encoding: str | None = None,
    ) -> Any:
        """
        Read the contents of a file from S3.

        Args:
            path (str | Path): Path of the file in S3 bucket.
            as_bytes (bool | None): If True, return file contents as bytes. Otherwise, as string.
            encoding (str | None): Encoding to use for decoding the file contents.

        Returns
        -------
            Any: File contents as bytes or string.
        """
        key = str(path)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            body = response["Body"].read()

            if as_bytes:
                return body
            encoding = encoding or self.default_encoding
            return body.decode(encoding)
        except ClientError as e:
            msg = f"Failed to read {key} from S3: {e}"
            raise FileNotFoundError(msg) from e



    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set method definition."""
        is_bytes = isinstance(value, bytes)
        write_type = "wb" if is_bytes else "w"
        encoding = None if is_bytes else encoding or self._encoding
        async with aiofiles.open(
            join_path(self._root_dir, key),
            cast("Any", write_type),
            encoding=encoding,
        ) as f:
            await f.write(value)

    async def set_s3(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set method definition."""
        try:
            if isinstance(value, pd.DataFrame):
                from io import BytesIO

                buffer = BytesIO()
                value.to_parquet(buffer)
                value = buffer.getvalue()
            else:
                is_bytes = isinstance(value, bytes)
                if not is_bytes:
                    encoding = encoding or "utf-8"
                    value = value.encode(encoding)

            s3_key = f"{self._root_dir}/{key}".replace("\\", "/")

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=value,
            )

            log.info("Successfully uploaded %s to S3 at %s", key, s3_key)
        except ClientError as e:
            error_message = f"Failed to upload {key} to S3"
            raise RuntimeError(error_message) from e

        log.info("Successfully uploaded %s to S3 at %s", key, s3_key)



    async def has(self, key: str) -> bool:
        """Has method definition."""
        return await exists(join_path(self._root_dir, key))

    async def has_s3(self, key: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            key (str): The path (key) of the file in the S3 bucket.

        Returns
        -------
            bool: True if the file exists, False otherwise.
        """
        try:
            path = f"{self._root_dir}/{key}"  # Construct the S3
            self.s3_client.head_object(Bucket=self.bucket_name, Key=path)
            return True  # noqa: TRY300
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise  # Other errors should be raised



    async def delete(self, key: str) -> None:
        """Delete method definition."""
        if await self.has(key):
            await remove(join_path(self._root_dir, key))

    async def clear(self) -> None:
        """Clear method definition."""
        for file in Path(self._root_dir).glob("*"):
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

    def child(self, name: str | None) -> "PipelineStorage":
        """Create a child storage instance."""
        if name is None:
            return self
        return FilePipelineStorage(str(Path(self._root_dir) / Path(name)))

    def keys(self) -> list[str]:
        """Return the keys in the storage."""
        return [item.name for item in Path(self._root_dir).iterdir() if item.is_file()]


def join_path(file_path: str, file_name: str) -> Path:
    """Join a path and a file. Independent of the OS."""
    return Path(file_path) / Path(file_name).parent / Path(file_name).name


def create_file_storage(**kwargs: Any) -> PipelineStorage:
    """Create a file based storage."""
    base_dir = kwargs["base_dir"]
    log.info("Creating file storage at %s", base_dir)
    return FilePipelineStorage(root_dir=base_dir)


def _create_progress_status(
    num_loaded: int, num_filtered: int, num_total: int
) -> Progress:
    return Progress(
        total_items=num_total,
        completed_items=num_loaded + num_filtered,
        description=f"{num_loaded} files loaded ({num_filtered} filtered)",
    )
