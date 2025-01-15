# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pathlib import Path
import boto3
import botocore.exceptions
from django.conf import settings

from pydantic import Field

import graphrag.config.defaults as defs
from graphrag.config.models.llm_config import LLMConfig


class SummarizeDescriptionsConfig(LLMConfig):
    """Configuration section for description summarization."""

    prompt: str | None = Field(
        description="The description summarization prompt to use.", default=None
    )
    max_length: int = Field(
        description="The description summarization maximum length.",
        default=defs.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.", default=None
    )

    def resolved_strategy(self, root_dir: str) -> dict:
        """Get the resolved description summarization strategy."""
        from graphrag.index.operations.summarize_descriptions import (
            SummarizeStrategyType,
        )

        return self.strategy or {
            "type": SummarizeStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "summarize_prompt": (Path(root_dir) / self.prompt)
            .read_bytes()
            .decode(encoding="utf-8")
            if self.prompt
            else None,
            "max_summary_length": self.max_length,
        }

    def resolved_strategy_s3(self, root_dir: str) -> dict:
        """Get the resolved entity extraction strategy."""
        from graphrag.index.operations.extract_entities import (
            ExtractEntityStrategyType,
        )

        s3_client = boto3.client(
            "s3",
            endpoint_url=settings.AWS_S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_S3_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME,
        )
        bucket_name = settings.AWS_STORAGE_BUCKET_NAME

        prompt_key = f"{root_dir}/{self.prompt}" if self.prompt else None
        summarize_prompt = None

        if prompt_key:
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=prompt_key)
                summarize_prompt = response["Body"].read().decode("utf-8")
            except s3_client.exceptions.NoSuchKey:
                summarize_prompt = "Default prompt content"
            except botocore.exceptions.BotoCoreError as e:
                error_message = f"Failed to fetch prompt from S3: {e}"
                raise ValueError(error_message) from e

        return self.strategy or {
            "type": ExtractEntityStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "summarize_prompt": summarize_prompt,
            "max_summary_length": self.max_length,
        }