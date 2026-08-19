import json
import logging

from minio import Minio
from processor.import_processor import import_config

# 添加调试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cfg = import_config.get_config()

# ===== 调试：打印配置值 =====
logger.info(f"=== MinIO 配置调试 ===")
logger.info(f"endpoint: [{cfg.minio_endpoint}]")
logger.info(f"access_key: [{cfg.minio_access_key}]")
logger.info(f"secret_key: [{cfg.minio_secret_key[:4]}***]")
logger.info(f"bucket: [{cfg.minio_bucket}]")
logger.info(f"=====================")
# ===========================

# cfg = import_config.get_config()

try:
    # 创建MinIO客户端
    minio_client = Minio(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        secure=False
    )
    if not minio_client.bucket_exists(cfg.minio_bucket):
        minio_client.make_bucket(cfg.minio_bucket)
    # 设置存储桶策略为 Public Read (只读权限开放给匿名用户)
    # 这样前端可以直接通过 URL 访问图片，而不需要预签名 URL
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{cfg.minio_bucket}/*",
            },
        ],
    }
    minio_client.set_bucket_policy(cfg.minio_bucket, json.dumps(policy))
except Exception as e:
    print("Minio init failed:{e}")
    minio_client = None

def get_minio_client():
    return minio_client

if __name__ == "__main__":
    get_minio_client()