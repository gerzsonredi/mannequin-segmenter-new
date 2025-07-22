import boto3
from pathlib import Path
from .env_utils import get_env_variable
from .logger import EVFSAMLogger

def download_evf_sam_from_s3() -> str:
    """
    Download EVF-SAM model from S3 if not already present locally.
    
    Returns:
        Path to the EVF-SAM directory
        
    Raises:
        ValueError: If AWS credentials are missing
        RuntimeError: If download fails and no local copy exists
    """
    evfsam_logger = EVFSAMLogger()
    
    evf_sam_path = Path("EVF-SAM")
    if evf_sam_path.exists() and evf_sam_path.is_dir():
        evfsam_logger.log("EVF-SAM folder already exists locally. Skipping download.")
        return str(evf_sam_path)

    AWS_ACCESS_KEY_ID = get_env_variable("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY")
    AWS_S3_REGION = get_env_variable("AWS_S3_REGION")

    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_REGION]):
        evfsam_logger.log("Missing AWS credentials in .env file")
        raise ValueError("Missing AWS credentials in .env file")

    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION
    )

    bucket_name = "artifactsredi"
    s3_prefix = "models/EVF-SAM/"

    evfsam_logger.log(f"Downloading EVF-SAM from S3 bucket: {bucket_name}/{s3_prefix}")

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

        downloaded_files = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    if s3_key.endswith('/'):
                        continue

                    local_path = Path(s3_key.replace(s3_prefix, "EVF-SAM/"))
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    evfsam_logger.log(f"Downloading: {s3_key} -> {local_path}")
                    s3_client.download_file(bucket_name, s3_key, str(local_path))
                    downloaded_files += 1

        if downloaded_files == 0:
            evfsam_logger.log(f"No files found in S3 bucket {bucket_name} with prefix {s3_prefix}")
            raise RuntimeError(f"No files found in S3 bucket {bucket_name} with prefix {s3_prefix}")

        evfsam_logger.log(f"Successfully downloaded {downloaded_files} files from S3")
        return str(evf_sam_path)

    except Exception as e:
        evfsam_logger.log(f"Error downloading EVF-SAM from S3: {e}")
        if not evf_sam_path.exists():
            raise RuntimeError(f"Failed to download EVF-SAM from S3 and no local copy exists: {e}")
        else:
            evfsam_logger.log("Using existing local EVF-SAM folder")
            return str(evf_sam_path) 