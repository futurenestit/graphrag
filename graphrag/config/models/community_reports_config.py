# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pathlib import Path

from django.core.files.storage import default_storage
from pydantic import Field

import graphrag.config.defaults as defs
from graphrag.config.models.llm_config import LLMConfig


class CommunityReportsConfig(LLMConfig):
    """Configuration section for community reports."""

    prompt: str | None = Field(
        description="The community report extraction prompt to use.", default=None
    )
    max_length: int = Field(
        description="The community report maximum length in tokens.",
        default=defs.COMMUNITY_REPORT_MAX_LENGTH,
    )
    max_input_length: int = Field(
        description="The maximum input length in tokens to use when generating reports.",
        default=defs.COMMUNITY_REPORT_MAX_INPUT_LENGTH,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.", default=None
    )

    def resolved_strategy(self, root_dir) -> dict:
        """Get the resolved community report extraction strategy."""
        from graphrag.index.operations.summarize_communities import (
            CreateCommunityReportsStrategyType,
        )

        return self.strategy or {
            "type": CreateCommunityReportsStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "extraction_prompt": (Path(root_dir) / self.prompt)
            .read_bytes()
            .decode(encoding="utf-8")
            if self.prompt
            else None,
            "max_report_length": self.max_length,
            "max_input_length": self.max_input_length,
        }

    def resolved_strategy_s3(self, root_dir) -> dict:
        """Get the resolved community report extraction strategy."""
        from graphrag.index.operations.summarize_communities import (
            CreateCommunityReportsStrategyType,
        )


        prompt_key = f"{root_dir}/{self.prompt}" if self.prompt else None
        community_report_prompt = None

        if prompt_key:
            try:
                if not default_storage.exists(prompt_key):
                    community_report_prompt = ("Default prompt content")
                else:
                    response = default_storage.open(prompt_key)
                    community_report_prompt = response.read().decode(encoding="utf-8")
                
            except Exception as e:
                error_message = f"Failed to fetch prompt from S3: {e}"
                raise ValueError(error_message) from e
        return self.strategy or {
            "type": CreateCommunityReportsStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "extraction_prompt": community_report_prompt,
            "max_report_length": self.max_length,
            "max_input_length": self.max_input_length,
        }
