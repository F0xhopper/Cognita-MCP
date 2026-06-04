"""File storage abstraction — local filesystem or S3-compatible."""

import uuid
from pathlib import Path

import aiofiles
import aiofiles.os

from cognita.core.config import settings
from cognita.core.exceptions import StorageError
from cognita.core.logging import get_logger

logger = get_logger(__name__)


class LocalStorage:
    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, user_id: str, filename: str) -> Path:
        user_dir = self._base / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / filename

    async def save(self, user_id: str, data: bytes, original_name: str) -> str:
        ext = Path(original_name).suffix.lower()
        key = f"{uuid.uuid4().hex}{ext}"
        dest = self._path_for(user_id, key)
        try:
            async with aiofiles.open(dest, "wb") as f:
                await f.write(data)
            return str(dest)
        except OSError as exc:
            raise StorageError(f"Failed to write {dest}: {exc}") from exc

    async def read(self, path: str) -> bytes:
        try:
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except OSError as exc:
            raise StorageError(f"Failed to read {path}: {exc}") from exc

    async def delete(self, path: str) -> None:
        try:
            await aiofiles.os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise StorageError(f"Failed to delete {path}: {exc}") from exc


class S3Storage:
    def __init__(self) -> None:
        import boto3
        self._s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self._bucket = settings.AWS_S3_BUCKET

    async def save(self, user_id: str, data: bytes, original_name: str) -> str:
        import asyncio
        ext = Path(original_name).suffix.lower()
        key = f"{user_id}/{uuid.uuid4().hex}{ext}"
        try:
            await asyncio.to_thread(
                self._s3.put_object,
                Bucket=self._bucket,
                Key=key,
                Body=data,
            )
            return key
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    async def read(self, path: str) -> bytes:
        import asyncio
        try:
            resp = await asyncio.to_thread(
                self._s3.get_object,
                Bucket=self._bucket,
                Key=path,
            )
            return resp["Body"].read()
        except Exception as exc:
            raise StorageError(str(exc)) from exc

    async def delete(self, path: str) -> None:
        import asyncio
        try:
            await asyncio.to_thread(
                self._s3.delete_object,
                Bucket=self._bucket,
                Key=path,
            )
        except Exception as exc:
            raise StorageError(str(exc)) from exc


def get_storage() -> LocalStorage | S3Storage:
    if settings.STORAGE_BACKEND == "s3":
        return S3Storage()
    return LocalStorage(settings.STORAGE_LOCAL_PATH)


_storage_instance: LocalStorage | S3Storage | None = None


def storage() -> LocalStorage | S3Storage:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = get_storage()
    return _storage_instance
