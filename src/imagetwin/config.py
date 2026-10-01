from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://imagetwin:imagetwin@database:5432/imagetwin"
    jwt_secret: str = Field(default="local-demo-imagetwin-replace-before-deployment", min_length=32)
    token_minutes: int = Field(default=60, ge=1, le=1440)
    testing: bool = False
    amqp_url: str = "amqp://imagetwin:imagetwin@rabbitmq:5672/"
    worker_interval: float = Field(default=0.5, ge=0.05, le=60)
    lease_seconds: float = Field(default=20, ge=2, le=120)
    max_attempts: int = Field(default=3, ge=1, le=10)
    cleanup_batch_size: int = Field(default=20, ge=1, le=100)
    cleanup_lease_seconds: float = Field(default=120, ge=30, le=600)
    cleanup_recheck_seconds: float = Field(default=300, ge=30, le=86400)
    inference_delay_seconds: float = Field(default=0, ge=0, le=30)
    model_dir: str = "models"
    s3_endpoint: str = "http://storage:9000"
    s3_access_key: str = "imagetwin"
    s3_secret_key: str = "imagetwin-local-demo"
    s3_bucket: str = "imagetwin-images"
    max_upload_bytes: int = Field(default=10485760, ge=1024, le=20971520)
    max_pixels: int = Field(default=4000000, ge=100, le=16000000)
    max_collection_images: int = Field(default=500, ge=1, le=5000)


settings = Settings()
