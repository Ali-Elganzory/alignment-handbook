import argparse
import inspect
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


# --- Helper: Configure YAML to dump multiline strings as blocks (|) ---
def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)


CHAT_TEMPLATE = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""


def make_base_config() -> dict:
    """Returns a mutable base configuration."""

    return {
        # Model arguments
        "model_name_or_path": "",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        # Data arguments
        "chat_template": CHAT_TEMPLATE,
        "dataset_mixture": {"datasets": []},
        "dataset_num_proc": 32,
        # SFT trainer config
        "bf16": True,
        "eval_strategy": "no",
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        "gradient_checkpointing": True,
        "learning_rate": 2.0e-5,
        "log_level": "info",
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine",
        "max_length": 2048,
        "output_dir": "",
        "overwrite_output_dir": False,
        "remove_unused_columns": True,
        "warmup_ratio": 0.03,
        "seed": 42,
        # Saving
        "save_strategy": "steps",
        "save_steps": 157,
        "save_total_limit": 2,
        # Reporting
        "report_to": ["wandb"],
        "logging_steps": 2,
        # Batching & steps
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        # TODO: MODIFY TO EXPERIMENT TARGET
        "max_steps": 10,
    }


################################################################################
# Data transformations
################################################################################


# --- Helper: Extract source code from a function ---
def get_function_source(func) -> str:
    """Extracts the source code of a function and dedents it."""
    source = inspect.getsource(func)
    return textwrap.dedent(source).strip()


def transform_smoltalk2(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms examples from the HuggingFaceTB/smoltalk2 dataset for SFT.
    - Merges consecutive turns from the same role.
    - Excludes examples that contain any tools in chat_template_kwargs['python_tools'] or ['xml_tools'].
    """
    chat_template_kwargs = example.get("chat_template_kwargs", {})
    # Exclude if there are any python_tools or xml_tools
    if (
        chat_template_kwargs.get("python_tools")
        and len(chat_template_kwargs["python_tools"]) > 0
    ) or (
        chat_template_kwargs.get("xml_tools")
        and len(chat_template_kwargs["xml_tools"]) > 0
    ):
        return {"messages": []}

    messages = [example["messages"][0]]
    for message in example["messages"][1:]:
        if message["role"] == messages[-1]["role"]:
            messages[-1]["content"] += "\n" + message["content"]
        else:
            messages.append(message)

    return {"messages": messages}


def transform_perfectblend(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms examples from the mlabonne/open-perfectblend dataset for SFT.
    - Renames roles: "human" -> "user", "gpt" -> "assistant"
    - Extracts message content under the "value" key.
    - Merges consecutive turns from the same role.
    """
    conversations = example["conversations"]
    if not conversations or not isinstance(conversations, list):
        return {"messages": []}

    role_map = {"human": "user", "gpt": "assistant"}
    messages = []
    for turn in conversations:
        role = turn["from"]
        content = turn["value"]
        current_role = role_map[role]
        current_content = content.strip()
        if messages and messages[-1]["role"] == current_role:
            messages[-1]["content"] += "\n" + current_content
        else:
            messages.append({"role": current_role, "content": current_content})

    return {"messages": messages}


def transform_orca_agentinstruct_1m(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms examples from the microsoft/orca-agentinstruct-1m-v1 dataset for SFT.
    - Converts the "messages" from string to list of dictionaries.
    """
    import json

    return {"messages": json.loads(example["messages"])}


def transform_anthropic_hh_rlhf(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms Anthropic/hh-rlhf text blobs into messages.
    Parses 'Human:' and 'Assistant:' markers.
    MERGES consecutive turns from the same role.
    """
    import re

    text = example["chosen"]
    # Regex to find (Role):(Content) blocks
    # Captures "Human" or "Assistant", then all text until the next role marker or end of string
    pattern = r"(Human|Assistant):([\s\S]+?)(?=(?:Human|Assistant):|$)"
    matches = re.findall(pattern, text)

    messages = []
    role_map = {"Human": "user", "Assistant": "assistant"}

    for role, content in matches:
        clean_role = role_map.get(role, role)
        clean_content = content.strip()

        # Check if we have messages and if the last one is from the SAME role
        if messages and messages[-1]["role"] == clean_role:
            # Merge the new content into the previous message
            # We use a newline separator to keep the text distinct
            messages[-1]["content"] += "\n" + clean_content
        else:
            # Otherwise, start a new message turn
            messages.append({"role": clean_role, "content": clean_content})

    return {"messages": messages}


def transform_stanfordnlp_shp(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms stanfordnlp/SHP examples for SFT.
    Selects the 'chosen' response based on the score label
    (label 1 means A is better, 0 means B is better).
    """
    # 1. Identify the preferred response
    if example["labels"] == 1:
        chosen_response = example["human_ref_A"]
    else:
        chosen_response = example["human_ref_B"]

    # 2. Construct the conversation
    # 'history' contains the post/instruction
    return {
        "messages": [
            {"role": "user", "content": example["history"]},
            {"role": "assistant", "content": chosen_response},
        ]
    }


def transform_berkeley_nest_nectar(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms berkeley-nest/Nectar examples for SFT.

    1. Cleans the 'prompt' field to remove specific 'Human:'/'Assistant:' artifacts.
    2. Selects the answer with 'rank': 1 from the 'answers' list.
    """
    import re

    # 1. Clean the prompt
    # The prompt usually looks like: "Human: [Query]\n\nAssistant:"
    # We remove the leading "Human: " and trailing "\n\nAssistant:"
    raw_prompt = example["prompt"]

    # Remove leading "Human: " (case insensitive just in case)
    # ^\s*Human:\s* matches start of string, optional whitespace, "Human:", optional whitespace
    clean_prompt = re.sub(r"^\s*Human:\s*", "", raw_prompt, flags=re.IGNORECASE)

    # Remove trailing "\n\nAssistant:" (and any trailing spaces)
    # \s*Assistant:\s*$ matches "Assistant:", optional whitespace, at end of string
    clean_prompt = re.sub(r"\s*Assistant:\s*$", "", clean_prompt, flags=re.IGNORECASE)

    # 2. Find the best answer (Rank 1)
    best_answer = ""
    # Default to the first one just in case rank 1 isn't found
    if example["answers"]:
        best_answer = example["answers"][0]["answer"]

        # Search specifically for rank 1
        for ans_obj in example["answers"]:
            if ans_obj["rank"] == 1:
                best_answer = ans_obj["answer"]
                break

    return {
        "messages": [
            {"role": "user", "content": clean_prompt},
            {"role": "assistant", "content": best_answer},
        ]
    }


def transform_arena_preference(example) -> dict[str, list[dict[str, str]]]:
    """
    Transforms lmarena-ai/arena-human-preference-55k examples.

    1. Checks 'winner_model_a', 'winner_model_b', and 'winner_tie'
    2. Selects the winning response; ties are included in the dataset.
    """

    # Check winners (usually boolean or 1/0)
    if example["winner_model_a"] or example["winner_tie"]:
        chosen_content = example["response_a"]
    else:
        chosen_content = example["response_b"]

    # The 'prompt' column contains the user input
    # Note: Sometimes the prompt is a list of strings (multi-turn).
    # For simplicity, we assume single-turn or take the first prompt string
    # if it's a list, but usually it's a string in this specific dataset version.
    prompt_content = example["prompt"]

    # If prompt is a list (rare but possible in arena data), join it or take first
    if isinstance(prompt_content, list):
        prompt_content = "\n".join(prompt_content)

    return {
        "messages": [
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": chosen_content},
        ]
    }


def transform_comparia_votes(example):
    """
    Transforms ministere-culture/comparia-votes examples.

    1. Identifies the winner (model A or B).
    2. Checks quality flags (incorrect/useful).
    3. Prepends the system prompt if available.
    4. Merges consecutive turns from the same role.
    """
    # 1. Identify the winner
    if example["both_equal"]:
        chosen_model = example["model_a_name"]
    elif example["chosen_model_name"] is not None:
        chosen_model = example["chosen_model_name"]
    else:
        return {"messages": []}

    model_a = example["model_a_name"]
    is_a_winner = chosen_model == model_a

    # 2. Select data based on winner
    model = "a" if is_a_winner else "b"

    # Quality check: Drop if incorrect OR not useful
    if example.get(f"conv_incorrect_{model}") is True:
        return {"messages": []}
    if example.get(f"conv_useful_{model}") is False:
        return {"messages": []}

    selected_conversation = example[f"conversation_{model}"]
    selected_system_prompt = example[f"system_prompt_{model}"]

    # 3. Handle System Prompt
    # We create a new list to avoid mutating the original dataset cache
    messages = list(selected_conversation)

    if selected_system_prompt and isinstance(selected_system_prompt, str):
        cleaned_prompt = selected_system_prompt.strip()
        if cleaned_prompt:
            # Prepend the system prompt
            messages.insert(0, {"role": "system", "content": cleaned_prompt})

    # 4. Merges consecutive turns from the same role.
    # Create a new list to avoid removing from the list while iterating
    merged_messages = [messages[0]]
    for message in messages[1:]:
        if message["role"] == merged_messages[-1]["role"]:
            merged_messages[-1]["content"] += "\n" + message["content"]
        else:
            merged_messages.append(message)
    messages = merged_messages

    # Map over messages to keep only the `content` and `role` keys
    messages = [
        {"content": message["content"], "role": message["role"]} for message in messages
    ]

    return {"messages": messages}


def transform_ultrafeedback(example):
    """
    Transforms argilla/ultrafeedback-binarized-preferences-cleaned samples.

    The 'chosen' column already contains the preferred conversation in the
    standard messages format.
    """
    return {"messages": example["chosen"]}


def transform_aegis_safety(example):
    """
    Transforms nvidia/Aegis-AI-Content-Safety-Dataset-2.0.

    1. Checks if the response is labeled 'safe'.
    2. If "REDACTED" returns empty messages (skipping the sample).
    3. If unsafe, returns empty messages (skipping the sample).
    4. If safe, returns the conversation.
    """
    if example["prompt"] == "REDACTED":
        return {"messages": []}

    if example["response_label"] == "unsafe":
        return {"messages": []}

    if example["response_label"] == "safe":
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }

    return {"messages": []}


def transform_helpsteer2(example):
    """
    Transforms nvidia/HelpSteer2.

    Picks the response with >= preference strength.
    """
    if example["preference_strength"] >= 0:
        chosen_response = example["response_1"]
    else:
        chosen_response = example["response_2"]

    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen_response},
        ]
    }


def transform_intel_orca_dpo(example):
    """
    Transforms argilla/distilabel-intel-orca-dpo-pairs.
    Uses 'input' as user prompt and 'chosen' as assistant response.
    Includes 'system' prompt if present.
    """
    messages = []

    # Add system prompt if it exists
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})

    messages.append({"role": "user", "content": example["input"]})
    messages.append({"role": "assistant", "content": example["chosen"]})

    return {"messages": messages}


def transform_human_like_dpo(example):
    """
    Transforms HumanLLMs/Human-Like-DPO-Dataset.
    Uses 'prompt' and 'chosen' columns.
    """
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
    }


def transform_mt_bench_judgments(example):
    """
    Transforms lmsys/mt_bench_human_judgments.
    Selects conversation_a or conversation_b based on the 'winner' field.
    """
    winner = example["winner"]

    if winner == "model_b":
        return {"messages": example["conversation_b"]}
    else:
        return {"messages": []}


def transform_math_preference(example):
    """
    Transforms argilla/distilabel-math-preference-dpo.
    Uses 'instruction' and 'chosen_response'.
    """
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["chosen_response"]},
        ]
    }


def transform_truthy_dpo(example):
    """
    Transforms jondurbin/truthy-dpo-v0.1.
    Includes 'system' prompt if available.
    """
    messages = []

    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})

    messages.append({"role": "user", "content": example["prompt"]})
    messages.append({"role": "assistant", "content": example["chosen"]})

    return {"messages": messages}


def get_bad_conversation_indices(dataset, msg_key="messages"):
    """
    Returns a list of (index, messages) tuples for examples where
    conversation roles do not strictly alternate.
    """
    bad_examples = []

    for i, example in enumerate(dataset):
        messages = example.get(msg_key, [])
        if not messages:
            continue

        # 1. Determine where the actual conversation loop starts
        # If the first message is 'system', the loop starts at index 1
        start_idx = 1 if messages[0]["role"] == "system" else 0
        conversation = messages[start_idx:]

        # 2. Check strict alternation
        # After system prompt (if any), 0 must be user, 1 assistant, 2 user...
        for loop_idx, msg in enumerate(conversation):
            is_user = msg["role"] == "user"
            should_be_user = loop_idx % 2 == 0

            if is_user != should_be_user:
                bad_examples.append((i, messages))
                break

    return bad_examples


################################################################################
# Data mixtures
################################################################################

DATA_MIXTURES: List[Dict[str, Any]] = [
    {
        "name": "nvidia-Nemotron-Post-Training-Dataset-v2",
        "datasets": [
            {
                "id": "nvidia/Nemotron-Post-Training-Dataset-v2",
                "config": "default",
                "split": "stem+chat+math+code+multilingual_ja+multilingual_de+multilingual_it+multilingual_es+multilingual_fr",
                "columns": ["messages"],
            },
        ],
    },
    {
        "name": "HuggingFaceTB-smoltalk2",
        "datasets": [
            {
                "id": "HuggingFaceTB/smoltalk2",
                "config": "SFT",
                "split": "LongAlign_64k_context_lang_annotated_lang_6_no_think+Mixture_of_Thoughts_science_no_think+OpenHermes_2.5_no_think+OpenThoughts3_1.2M_no_think_no_think+hermes_function_calling_v1_no_think+smoltalk_multilingual_8languages_lang_5_no_think+smoltalk_smollm3_everyday_conversations_no_think+smoltalk_smollm3_explore_instruct_rewriting_no_think+smoltalk_smollm3_smol_magpie_ultra_no_think+smoltalk_smollm3_smol_rewrite_no_think+smoltalk_smollm3_smol_summarize_no_think+smoltalk_smollm3_systemchats_30k_no_think+table_gpt_no_think+tulu_3_sft_personas_instruction_following_no_think+xlam_traces_no_think",
                "columns": ["messages"],
                "transform_code": get_function_source(transform_smoltalk2),
                "final_columns": ["messages"],
            },
            {
                "id": "HuggingFaceTB/smoltalk2",
                "config": "Preference",
                "split": "llama_3.1_tulu_3_8b_preference_mixture_no_think",
                "columns": ["chosen"],
                "rename_columns": {
                    "chosen": "messages",
                },
                "transform_code": get_function_source(transform_smoltalk2),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "mlabonne-open-perfectblend",
        "datasets": [
            {
                "id": "mlabonne/open-perfectblend",
                "config": "default",
                "split": "train",
                "columns": ["conversations"],
                "transform_code": get_function_source(transform_perfectblend),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "microsoft-orca-agentinstruct-1M-v1",
        "datasets": [
            {
                "id": "microsoft/orca-agentinstruct-1M-v1",
                "config": "default",
                "split": "creative_content+text_modification+struct2text_flow+rc+rag+text_extraction+mcq+follow_up+analytical_reasoning+fermi+fs_cot_flow+code_+brain_teaser+text_classification+open_domain_qa",
                "columns": ["messages"],
                "transform_code": get_function_source(transform_orca_agentinstruct_1m),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "lmsys-lmsys-chat-1m",
        "datasets": [
            {
                "id": "lmsys/lmsys-chat-1m",
                "config": "default",
                "split": "train",
                "columns": ["conversation"],
                "rename_columns": {
                    "conversation": "messages",
                },
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "Anthropic-hh-rlhf",
        "datasets": [
            {
                "id": "Anthropic/hh-rlhf",
                "config": "default",
                "split": "train+test",
                "columns": ["chosen"],
                "transform_code": get_function_source(transform_anthropic_hh_rlhf),
            },
        ],
    },
    {
        "name": "stanfordnlp-SHP",
        "datasets": [
            {
                "id": "stanfordnlp/SHP",
                "config": "default",
                "split": "train+validation+test",
                "columns": ["labels", "history", "human_ref_A", "human_ref_B"],
                "transform_code": get_function_source(transform_stanfordnlp_shp),
            },
        ],
    },
    {
        "name": "berkeley-nest-Nectar",
        "datasets": [
            {
                "id": "berkeley-nest/Nectar",
                "config": "default",
                "split": "train",
                "columns": ["prompt", "answers"],
                "transform_code": get_function_source(transform_berkeley_nest_nectar),
            },
        ],
    },
    {
        "name": "lmarena-ai-arena-human-preference-55k",
        "datasets": [
            {
                "id": "lmarena-ai/arena-human-preference-55k",
                "config": "default",
                "split": "train",
                "columns": [
                    "prompt",
                    "response_a",
                    "response_b",
                    "winner_model_a",
                    "winner_model_b",
                    "winner_tie",
                ],
                "transform_code": get_function_source(transform_arena_preference),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "ministere-culture-comparia-votes",
        "datasets": [
            {
                "id": "ministere-culture/comparia-votes",
                "config": "default",
                "split": "train",
                "columns": [
                    "chosen_model_name",
                    "model_a_name",
                    "conversation_a",
                    "conversation_b",
                    "both_equal",
                    "conv_incorrect_a",
                    "conv_useful_a",
                    "conv_incorrect_b",
                    "conv_useful_b",
                    "system_prompt_a",
                    "system_prompt_b",
                ],
                "transform_code": get_function_source(transform_comparia_votes),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "argilla-ultrafeedback-binarized-preferences-cleaned",
        "datasets": [
            {
                "id": "argilla/ultrafeedback-binarized-preferences-cleaned",
                "config": "default",
                "split": "train",
                "columns": ["chosen"],
                "transform_code": get_function_source(transform_ultrafeedback),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "nvidia-Aegis-AI-Content-Safety-Dataset-2.0",
        "datasets": [
            {
                "id": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                "config": "default",
                "split": "train+validation+test",
                "columns": ["prompt", "response", "response_label"],
                "transform_code": get_function_source(transform_aegis_safety),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "nvidia-HelpSteer2",
        "datasets": [
            {
                "id": "nvidia/HelpSteer2",
                "config": "default",
                "data_dir": "preference",
                "split": "train",
                "columns": [
                    "prompt",
                    "response_1",
                    "response_2",
                    "preference_strength",
                ],
                "transform_code": get_function_source(transform_helpsteer2),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "argilla-distilabel-intel-orca-dpo-pairs",
        "datasets": [
            {
                "id": "argilla/distilabel-intel-orca-dpo-pairs",
                "config": "default",
                "split": "train",
                "columns": ["system", "input", "chosen"],
                "transform_code": get_function_source(transform_intel_orca_dpo),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "HumanLLMs-Human-Like-DPO-Dataset",
        "datasets": [
            {
                "id": "HumanLLMs/Human-Like-DPO-Dataset",
                "config": "default",
                "split": "train",
                "columns": ["prompt", "chosen"],
                "transform_code": get_function_source(transform_human_like_dpo),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "argilla-distilabel-capybara-dpo-7k-binarized",
        "datasets": [
            {
                "id": "argilla/distilabel-capybara-dpo-7k-binarized",
                "config": "default",
                "split": "train",
                "columns": ["chosen"],
                "rename_columns": {
                    "chosen": "messages",
                },
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "lmsys-mt_bench_human_judgments",
        "datasets": [
            {
                "id": "lmsys/mt_bench_human_judgments",
                "config": "default",
                "split": "human",
                "columns": ["winner", "conversation_a", "conversation_b"],
                "transform_code": get_function_source(transform_mt_bench_judgments),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "argilla-distilabel-math-preference-dpo",
        "datasets": [
            {
                "id": "argilla/distilabel-math-preference-dpo",
                "config": "default",
                "split": "train",
                "columns": ["instruction", "chosen_response"],
                "transform_code": get_function_source(transform_math_preference),
                "final_columns": ["messages"],
            },
        ],
    },
    {
        "name": "jondurbin-truthy-dpo-v0.1",
        "datasets": [
            {
                "id": "jondurbin/truthy-dpo-v0.1",
                "config": "default",
                "split": "train",
                "columns": ["system", "prompt", "chosen"],
                "transform_code": get_function_source(transform_truthy_dpo),
                "final_columns": ["messages"],
            },
        ],
    },
]


################################################################################
# Models
################################################################################

MODELS: List[Dict[str, Any]] = [
    {"name": "allenai/Olmo-3-1025-7B"},
]

################################################################################
# Generate recipes
################################################################################


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "-")


def create_recipe(model: Dict[str, Any], mixture: Dict[str, Any]) -> dict:
    config = deepcopy(make_base_config())
    config["model_name_or_path"] = model["name"]
    config["dataset_mixture"]["datasets"] = mixture["datasets"]

    model_slug = sanitize_model_name(model["name"])
    config["output_dir"] = f"results/data_selection/sft/{model_slug}_{mixture['name']}"

    return config


def write_recipe(
    recipe: dict,
    mixture_name: str,
    model_name: str,
    target_dir: Path,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sanitize_model_name(model_name)}_{mixture_name}.yaml"
    output_path = target_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(recipe, handle, sort_keys=False)
    return output_path


def write_slurm_script(
    recipe_path: Path, mixture_name: str, model_name: str, target_dir: Path
) -> Path:
    """Generate a SLURM script for running this SFT recipe."""
    script_name = f"{sanitize_model_name(model_name)}_{mixture_name}.sh"
    script_path = target_dir / script_name

    # TODO: MODIFY TO EXPERIMENT TARGET
    devices = 1

    # You can modify settings here as appropriate for your environment!
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={sanitize_model_name(model_name)}_{mixture_name}_sft
#SBATCH --output=slurm_logs/data_selection/sft/{sanitize_model_name(model_name)}_{mixture_name}/%j.%x.%N.out
#SBATCH --error=slurm_logs/data_selection/sft/{sanitize_model_name(model_name)}_{mixture_name}/%j.%x.%N.err
#SBATCH --time=00-04:00:00
#SBATCH --partition=accelerated-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{devices}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alielganzory@hotmail.com

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

source handbook/bin/activate

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes={devices} \
    scripts/sft.py --config {recipe_path}
"""

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w", encoding="utf-8") as fh:
        fh.write(slurm_content)
    script_path.chmod(0o770)
    return script_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate HuggingFace TRL recipes and Slurm scripts for each model/data mixture combination."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to place generated YAML and SLURM files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for mixture in DATA_MIXTURES:
        for model in MODELS:
            recipe = create_recipe(model, mixture)
            recipe_path = write_recipe(
                recipe,
                mixture["name"],
                model["name"],
                args.output_dir,
            )
            rel_recipe_path = recipe_path.relative_to(Path(__file__).parent)
            print(
                f"✓ Model: {model['name']} | Mixture: {mixture['name']} | Recipe Path: {rel_recipe_path}"
            )
            slurm_path = write_slurm_script(
                recipe_path=recipe_path,
                mixture_name=mixture["name"],
                model_name=model["name"],
                target_dir=args.output_dir,
            )
            print(f"  SLURM script written to: {slurm_path}")


if __name__ == "__main__":
    main()
