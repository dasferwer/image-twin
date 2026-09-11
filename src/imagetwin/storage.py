import hashlib
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import settings


@lru_cache(maxsize=1)
def client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name="us-east-1",
        config=Config(
            connect_timeout=3,
            read_timeout=10,
            retries={"max_attempts": 2},
            s3={"addressing_style": "path"},
        ),
    )


def ensure_bucket():
    try:
        client().head_bucket(Bucket=settings.s3_bucket)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
            raise
        client().create_bucket(Bucket=settings.s3_bucket)


def put(key, data):
    client().put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType="image/png",
        Metadata={"sha256": hashlib.sha256(data).hexdigest()},
    )


def get(key, expected_sha=None):
    response = client().get_object(Bucket=settings.s3_bucket, Key=key)
    with response["Body"] as body:
        data = body.read(settings.max_pixels * 4 + 1048576)
    if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Stored image checksum mismatch")
    return data
