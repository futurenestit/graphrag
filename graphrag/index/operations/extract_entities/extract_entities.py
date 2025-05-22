# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing entity_extract methods."""

import logging
from typing import Any

import pandas as pd
from datashaper import (
    AsyncType,
    VerbCallbacks,
    derive_from_rows,
)

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.index.bootstrap import bootstrap
from graphrag.index.operations.extract_entities.typing import (
    Document,
    EntityExtractStrategy,
    ExtractEntityStrategyType,
)

log = logging.getLogger(__name__)


DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


async def extract_entities(
    text_units: pd.DataFrame,
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    text_column: str,
    id_column: str,
    strategy: dict[str, Any] | None,
    async_mode: AsyncType = AsyncType.AsyncIO,
    entity_types=DEFAULT_ENTITY_TYPES,
    num_threads: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract entities from a piece of text.

    ## Usage
    ```yaml
    args:
        column: the_document_text_column_to_extract_entities_from
        id_column: the_column_with_the_unique_id_for_each_row
        to: the_column_to_output_the_entities_to
        strategy: <strategy_config>, see strategies section below
        summarize_descriptions: true | false /* Optional: This will summarize the descriptions of the entities and relationships, default: true */
        entity_types:
            - list
            - of
            - entity
            - types
            - to
            - extract
    ```

    ## Strategies
    The entity extract verb uses a strategy to extract entities from a document. The strategy is a json object which defines the strategy to use. The following strategies are available:

    ### graph_intelligence
    This strategy uses the [graph_intelligence] library to extract entities from a document. In particular it uses a LLM to extract entities from a piece of text. The strategy config is as follows:

    ```yml
    strategy:
        type: graph_intelligence
        extraction_prompt: !include ./entity_extraction_prompt.txt # Optional, the prompt to use for extraction
        completion_delimiter: "<|COMPLETE|>" # Optional, the delimiter to use for the LLM to mark completion
        tuple_delimiter: "<|>" # Optional, the delimiter to use for the LLM to mark a tuple
        record_delimiter: "##" # Optional, the delimiter to use for the LLM to mark a record

        encoding_name: cl100k_base # Optional, The encoding to use for the LLM with gleanings

        llm: # The configuration for the LLM
            type: openai # the type of llm to use, available options are: openai, azure, openai_chat, azure_openai_chat.  The last two being chat based LLMs.
            api_key: !ENV ${GRAPHRAG_OPENAI_API_KEY} # The api key to use for openai
            model: !ENV ${GRAPHRAG_OPENAI_MODEL:gpt-4-turbo-preview} # The model to use for openai
            max_tokens: !ENV ${GRAPHRAG_MAX_TOKENS:6000} # The max tokens to use for openai
            organization: !ENV ${GRAPHRAG_OPENAI_ORGANIZATION} # The organization to use for openai

            # if using azure flavor
            api_base: !ENV ${GRAPHRAG_OPENAI_API_BASE} # The api base to use for azure
            api_version: !ENV ${GRAPHRAG_OPENAI_API_VERSION} # The api version to use for azure
            proxy: !ENV ${GRAPHRAG_OPENAI_PROXY} # The proxy to use for azure

    ```

    ### nltk
    This strategy uses the [nltk] library to extract entities from a document. In particular it uses a nltk to extract entities from a piece of text. The strategy config is as follows:
    ```yml
    strategy:
        type: nltk
    ```
    """
    log.debug("entity_extract strategy=%s", strategy)
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    strategy = strategy or {}
    strategy_exec = _load_strategy(
        strategy.get("type", ExtractEntityStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    num_started = 0

    async def run_strategy(row):
        nonlocal num_started
        text = row[text_column]
        id_val = row[id_column]
        result = await strategy_exec(
            [Document(text=text, id=id_val)],
            entity_types,
            callbacks,
            cache,
            strategy_config,
        )
        num_started += 1
        # ---- START DEBUG PRINTS ----

        print(f"DEBUG: For text unit id {id_val}, raw result.entities: {result.entities}")
        if result.entities:
            try:
                temp_df = pd.DataFrame(result.entities)
                print(f"DEBUG: For text unit id {id_val}, temp_df columns: {temp_df.columns.tolist()}")
                if 'title' not in temp_df.columns:
                    print(f"WARNING: 'title' column MISSING in entities from text unit id {id_val}")
                    print(f"DEBUG: Sample of entities without 'title': {temp_df.head().to_dict(orient='records')}")
            except Exception as e:
                print(f"ERROR: Could not create DataFrame from result.entities for id {id_val}. Error: {e}")
                print(f"DEBUG: result.entities was: {result.entities}")
        # ---- END DEBUG PRINTS ----

        return [result.entities, result.relationships, result.graph]

    results = await derive_from_rows(
        text_units,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=num_threads,
    )

    entity_dfs = []
    relationship_dfs = []
    print(f"Started {num_started} tasks, completed {len(results)} tasks")
    for i, res_tuple in enumerate(results):
        if res_tuple:
            print(f"Processing result index {i}: {res_tuple}")
            current_entities_data = res_tuple[0]
            current_relationships_data = res_tuple[1]

            # --- Entity Processing (unchanged or adjust as needed) ---
            if current_entities_data is not None:
                if isinstance(current_entities_data, list):
                    if not current_entities_data:
                        # print(f"DEBUG: Result index {i} had an empty list for entities. Appending empty DataFrame.")
                        entity_dfs.append(pd.DataFrame())
                    else:
                        try:
                            df_entities = pd.DataFrame(current_entities_data)
                            # print(f"DEBUG: Result index {i}, df_entities columns: {df_entities.columns.tolist()}")
                            entity_dfs.append(df_entities)
                        except Exception as e:
                            print(f"ERROR: Failed to create DataFrame for entities at index {i}. Data: {current_entities_data}. Error: {e}")
                            entity_dfs.append(pd.DataFrame())
                else:
                    print(f"WARNING: Result index {i}, entities data is not a list: {type(current_entities_data)}. Appending empty DataFrame.")
                    entity_dfs.append(pd.DataFrame())
            else:
                # print(f"DEBUG: Result index {i} had None for entities. Appending empty DataFrame.")
                entity_dfs.append(pd.DataFrame())

            # --- Relationship Processing ---
            if current_relationships_data is not None:
                if isinstance(current_relationships_data, pd.DataFrame): # If already a DataFrame
                    if current_relationships_data.empty and not list(current_relationships_data.columns):
                        # print(f"DEBUG: Result index {i}, relationships data is an empty DataFrame with no columns. Appending as is.")
                        relationship_dfs.append(current_relationships_data) # Possibly a completely empty DF
                    elif not current_relationships_data.empty or list(current_relationships_data.columns):
                        # print(f"DEBUG: Result index {i}, relationships data is a DataFrame. Appending as is. Columns: {current_relationships_data.columns.tolist()}")
                        relationship_dfs.append(current_relationships_data)
                    else: # Completely empty DataFrame (no rows, no columns)
                        relationship_dfs.append(pd.DataFrame()) # Ensure an empty DF is appended
                elif isinstance(current_relationships_data, list): # If it's a list
                    if not current_relationships_data:
                        # print(f"DEBUG: Result index {i} had an empty list for relationships. Appending empty DataFrame.")
                        relationship_dfs.append(pd.DataFrame())
                    else:
                        try:
                            df_relationships = pd.DataFrame(current_relationships_data)
                            # print(f"DEBUG: Result index {i}, df_relationships columns: {df_relationships.columns.tolist()}")
                            relationship_dfs.append(df_relationships)
                        except Exception as e:
                            print(f"ERROR: Failed to create DataFrame for relationships at index {i}. Data: {current_relationships_data}. Error: {e}")
                            relationship_dfs.append(pd.DataFrame())
                else:
                    print(f"WARNING: Result index {i}, relationships data is not a list or DataFrame: {type(current_relationships_data)}. Appending empty DataFrame.")
                    relationship_dfs.append(pd.DataFrame())
            else:
                # print(f"DEBUG: Result index {i} had None for relationships. Appending empty DataFrame.")
                relationship_dfs.append(pd.DataFrame())


    entities = _merge_entities(entity_dfs)
    relationships = _merge_relationships(relationship_dfs)

    return (entities, relationships)


def _load_strategy(strategy_type: ExtractEntityStrategyType) -> EntityExtractStrategy:
    """Load strategy method definition."""
    match strategy_type:
        case ExtractEntityStrategyType.graph_intelligence:
            from graphrag.index.operations.extract_entities.graph_intelligence_strategy import (
                run_graph_intelligence,
            )

            return run_graph_intelligence

        case ExtractEntityStrategyType.nltk:
            bootstrap()
            # dynamically import nltk strategy to avoid dependency if not used
            from graphrag.index.operations.extract_entities.nltk_strategy import (
                run as run_nltk,
            )

            return run_nltk
        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)


def _merge_entities(entity_dfs) -> pd.DataFrame:
    # Filter out truly empty DataFrames (no columns, no rows) before concat
    # DataFrames created from empty lists `[]` will have columns=[]
    # DataFrames with columns but no rows are fine.
    valid_entity_dfs = [df for df in entity_dfs if not df.empty or list(df.columns)]

    if not valid_entity_dfs:
        print("WARNING: _merge_entities received no valid DataFrames. Returning empty DataFrame with expected schema.")
        return pd.DataFrame(columns=['title', 'type', 'description', 'text_unit_ids']) # Ensure schema

    all_entities = pd.concat(valid_entity_dfs, ignore_index=True)

    print(f"DEBUG: In _merge_entities, all_entities columns: {all_entities.columns.tolist()}")
    print(f"DEBUG: In _merge_entities, all_entities head:\n{all_entities.head().to_string()}")

    if 'title' not in all_entities.columns or 'type' not in all_entities.columns:
        # If 'title' or 'type' is critical and missing, this is a problem.
        # Log the issue and decide how to proceed.
        # For now, let's create them if missing to avoid KeyError, but this hides the real issue
        missing_cols_error_msg = []
        if 'title' not in all_entities.columns:
            print("ERROR: 'title' column is missing in all_entities before groupby. This indicates an issue with entity extraction.")
            missing_cols_error_msg.append("'title'")
            # Potentially add an empty 'title' column to prevent crash, but this hides the real issue
            # all_entities['title'] = None
        if 'type' not in all_entities.columns:
            print("ERROR: 'type' column is missing in all_entities before groupby. This indicates an issue with entity extraction.")
            missing_cols_error_msg.append("'type'")
            # all_entities['type'] = None

        if missing_cols_error_msg:
             # If these columns are essential for groupby, we cannot proceed meaningfully.
             # Raise a more informative error or return a structured empty DataFrame.
             raise ValueError(f"Critical columns {', '.join(missing_cols_error_msg)} missing in concatenated entities. Check entity extraction strategy output.")

    # Ensure 'source_id' and 'description' exist, or handle their absence in agg
    # For example, if they might be missing, provide a default or skip aggregation for them
    agg_spec = {}
    if 'description' in all_entities.columns:
        agg_spec['description'] = ('description', list)
    else:
        print("WARNING: 'description' column missing in all_entities. It will not be aggregated.")
        # Optionally, create an empty list for description if the schema requires it
        # agg_spec['description'] = pd.NamedAgg(column='description', aggfunc=lambda x: [])


    if 'source_id' in all_entities.columns:
        agg_spec['text_unit_ids'] = ('source_id', list)
    else:
        print("WARNING: 'source_id' column missing in all_entities (expected for text_unit_ids). It will not be aggregated.")

    if not agg_spec: # If no columns to aggregate are present
        print("WARNING: No columns available for aggregation in _merge_entities. Groupby result might be minimal.")
        # If only grouping, no agg is needed. But here we expect description and source_id.
        # If they are truly optional, this is fine. If not, it's an issue.
    return (
        all_entities.groupby(["title", "type"], sort=False, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )


def _merge_relationships(relationship_dfs) -> pd.DataFrame:
    valid_relationship_dfs = [df for df in relationship_dfs if not df.empty or list(df.columns)]

    if not valid_relationship_dfs:
        print("WARNING: _merge_relationships received no valid DataFrames. Returning empty DataFrame with expected schema.")
        # Ensure the returned DataFrame has at least the columns required for groupby and aggregation
        return pd.DataFrame(columns=['source', 'target', 'description', 'text_unit_ids', 'weight'])

    all_relationships = pd.concat(valid_relationship_dfs, ignore_index=True) # ignore_index=False for relationships? Check original code. Original: False

    print(f"DEBUG: In _merge_relationships, all_relationships columns: {all_relationships.columns.tolist()}")
    print(f"DEBUG: In _merge_relationships, all_relationships head:\n{all_relationships.head().to_string()}")

    # Check for columns required by groupby
    required_groupby_cols = ["source", "target"]
    missing_groupby_cols = [col for col in required_groupby_cols if col not in all_relationships.columns]
    if missing_groupby_cols:
        # If groupby columns are missing, this is a serious problem
        raise ValueError(f"Critical groupby columns {', '.join(missing_groupby_cols)} missing in concatenated relationships. Check entity extraction strategy output for relationships.")

    # Build aggregation spec, handle possible missing columns
    agg_spec = {}
    if 'description' in all_relationships.columns:
        agg_spec['description'] = ('description', list)
    else:
        print("WARNING: 'description' column missing in all_relationships for merge. It will not be aggregated.")

    if 'source_id' in all_relationships.columns: # 'source_id' comes from original data, used for text_unit_ids
        agg_spec['text_unit_ids'] = ('source_id', list)
    else:
        print("WARNING: 'source_id' column missing in all_relationships for merge (expected for text_unit_ids). It will not be aggregated.")

    if 'weight' in all_relationships.columns:
        agg_spec['weight'] = ('weight', "sum")
    else:
        print("WARNING: 'weight' column missing in all_relationships for merge. It will not be aggregated.")

    # If agg_spec is empty, after groupby no .agg() is needed, or .agg({})
    if not agg_spec:
         print("WARNING: No columns available for aggregation in _merge_relationships. Groupby result will be minimal.")
         return (
            all_relationships.groupby(["source", "target"], sort=False, dropna=False)
            .size().reset_index(name='count_temp') 
            
         )

    return (
        all_relationships.groupby(["source", "target"], sort=False, dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
