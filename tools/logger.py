import os
import boto3
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .env_utils import get_env_variable
except ImportError:
    from env_utils import get_env_variable

class AppLogger:
    """
    Simple logger for BiRefNet mannequin segmentation application.
    Logs are saved locally and uploaded to S3 logs-redi bucket.
    Each day a new log file is created, named with the date and 'birefnet' marker.
    """
    def __init__(self):
        self.bucket_name = "logs-redi"
        self.aws_access_key_id = get_env_variable("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = get_env_variable("AWS_SECRET_ACCESS_KEY")
        self.aws_s3_region = get_env_variable("AWS_S3_REGION")
        self.s3_client = None
        if all([self.aws_access_key_id, self.aws_secret_access_key, self.aws_s3_region]):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_s3_region
            )
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = self._get_log_file_path()

    def _get_log_file_path(self):
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"birefnet_{today}.log"
        return os.path.join(self.log_dir, filename)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logline = f"[{timestamp}] {message}\n"
        self.log_file = self._get_log_file_path()
        with open(self.log_file, "a") as f:
            f.write(logline)
        self._upload_to_s3()

    def _upload_to_s3(self):
        if self.s3_client is None:
            return
        try:
            s3_key = f"birefnet/{os.path.basename(self.log_file)}"
            self.s3_client.upload_file(self.log_file, self.bucket_name, s3_key)
        except Exception as e:
            # If upload fails, just skip (do not crash the app)
            pass 