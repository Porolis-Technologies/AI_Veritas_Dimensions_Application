from urllib.parse import urlparse

import boto3

from deploy.config import get_logger

logger = get_logger(__name__)

s3_client = boto3.client("s3")


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parses an S3 URI into bucket name and key."""
    parsed_url = urlparse(s3_uri)

    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip("/")

    return bucket_name, key


def download_file_from_s3(s3_uri: str, file_name: str):
    try:
        bucket, object_name = parse_s3_uri(s3_uri)
        s3_client.download_file(bucket, object_name, file_name)
        logger.info(f"Downloaded file from {s3_uri} and saved as {file_name}")
    except Exception as e:
        raise FileNotFoundError(f"File not found in {s3_uri} or not accessible. Error: {e}")