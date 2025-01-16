# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from pydantic import Field

import graphrag.config.defaults as defs
from graphrag.config.models.llm_config import LLMConfig


class EntityExtractionConfig(LLMConfig):
    """Configuration section for entity extraction."""

    prompt: str | None = Field(
        description="The entity extraction prompt to use.", default=None
    )
    entity_types: list[str] = Field(
        description="The entity extraction entity types to use.",
        default=defs.ENTITY_EXTRACTION_ENTITY_TYPES,
    )
    max_gleanings: int = Field(
        description="The maximum number of entity gleanings to use.",
        default=defs.ENTITY_EXTRACTION_MAX_GLEANINGS,
    )
    strategy: dict | None = Field(
        description="Override the default entity extraction strategy", default=None
    )
    encoding_model: str | None = Field(
        default=None, description="The encoding model to use."
    )

    def resolved_strategy(self, root_dir: str, encoding_model: str | None) -> dict:
        """Get the resolved entity extraction strategy."""
        from graphrag.index.operations.extract_entities import (
            ExtractEntityStrategyType,
        )

        return self.strategy or {
            "type": ExtractEntityStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "extraction_prompt": (Path(root_dir) / self.prompt)
            .read_bytes()
            .decode(encoding="utf-8")
            if self.prompt
            else None,
            "max_gleanings": self.max_gleanings,
            "encoding_name": encoding_model or self.encoding_model,
        }

    def resolved_strategy_s3(self, root_dir: str, encoding_model: str | None) -> dict:
        """Get the resolved entity extraction strategy."""
        from graphrag.index.operations.extract_entities import (
            ExtractEntityStrategyType,
        )

        prompt_key = f"{root_dir}/{self.prompt}" if self.prompt else None
        extraction_prompt = None

        if prompt_key:
            try:
                if default_storage.exists(prompt_key):
                    with default_storage.open(prompt_key) as file:
                        extraction_prompt = file.read().decode("utf-8")
                else:
                    extraction_prompt = "Default prompt content"
            except Exception as e:
                error_message = f"Failed to fetch prompt from S3: {e}"
                raise ValueError(error_message) from e

        return self.strategy or {
            "type": ExtractEntityStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "extraction_prompt": extraction_prompt,
            "max_gleanings": self.max_gleanings,
            "encoding_name": encoding_model or self.encoding_model,
        }
