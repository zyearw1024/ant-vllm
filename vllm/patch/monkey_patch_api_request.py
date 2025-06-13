import sys
import json
import base64
import logging
import re

from typing import Optional
from vllm import __version__ as vllm_version
from packaging import version
from fastapi.responses import JSONResponse
from functools import cache
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import StreamOptions
from vllm.entrypoints.openai import cli_args
from vllm.entrypoints.openai.cli_context_args import cliContext
from typing import Union, List

# Try importing VLLM components with version-specific handling
try:
    from vllm.entrypoints.openai.serving_engine import OpenAIServing
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    OpenAIServing = None
    OpenAIServingChat = None
    FlexibleArgumentParser = None


logger = logging.getLogger(__name__)


# Compile the regex for checking /think or /nothink at the end of the string
THINKING_TAG_REGEX = re.compile(r"/(think|nothink||no_think)\s*$")


class VllmVersionMagic:
    """
    VLLM version comparison utility class.
    Handles version comparison and provides convenient version check methods.
    """

    # Version constants
    V0_2_7 = "0.2.7"
    V0_6_2 = "0.6.2"

    def __init__(self, current_version: str):
        """
        Initialize with current VLLM version
        Args:
            current_version: Current VLLM version string
        """
        self.version = version.parse(current_version)

    def less_than(self, target_version: str) -> bool:
        """Check if current version is less than target version"""
        return self.version < version.parse(target_version)

    def greater_than(self, target_version: str) -> bool:
        """Check if current version is greater than target version"""
        return self.version > version.parse(target_version)

    def greater_than_or_equal(self, target_version: str) -> bool:
        """Check if current version is greater than or equal to target version"""
        return self.version >= version.parse(target_version)

    def less_than_0_2_7(self) -> bool:
        """Check if version < 0.2.7"""
        return self.less_than(self.V0_2_7)

    def less_than_0_6_2(self) -> bool:
        """Check if version < 0.6.2"""
        return self.less_than(self.V0_6_2)

    def greater_than_or_equal_0_6_2(self) -> bool:
        """Check if version >= 0.6.2"""
        return self.greater_than_or_equal(self.V0_6_2)


def compare_version(
    current_version: str, target_version: str, ignore_dev: bool = False
) -> bool:
    """
    Compare version numbers with optional dev version handling

    Args:
        current_version: Current version string (e.g., '0.6.4.dev653+g1b5b0be7.d20241125')
        target_version: Target version string (e.g., '0.6.5')
        ignore_dev: Whether to ignore development version number in comparison
    Returns:
        bool: True if current_version > target_version
    """
    try:
        if ignore_dev:
            current = version.parse(".".join(current_version.split(".")[:3]))
            target = version.parse(target_version)
        else:
            current = version.parse(current_version)
            target = version.parse(target_version)
        return current > target
    except version.InvalidVersion:
        return False


# Initialize version magic singleton
vllm_version_magic = VllmVersionMagic(vllm_version)

# Store original make_arg_parser for later use
origin_make_arg_parser = cli_args.make_arg_parser


@cache
def get_default_model_template() -> Optional[str]:
    """Get default model template from CLI arguments"""
    args = cliContext.args
    if not args:
        return None
    default_model_template = getattr(args, "default_model_template", None)
    logger.debug("default_model_template: %s", default_model_template)
    return default_model_template


@cache
def get_stream_include_usage_status() -> bool:
    """Get stream include usage status from CLI arguments"""
    args = cliContext.args
    if not args:
        return False
    return getattr(args, "enable_stream_include_usage", False)


@cache
def get_continuous_usage_stats_status() -> bool:
    """Get continuous usage stats status from CLI arguments"""
    args = cliContext.args
    if not args:
        return False
    return getattr(args, "disable_continuous_usage_stats", False)


@cache
def get_qwen3_prompt_suffix_thinking_status() -> bool:
    """Get qwen3_enable_prompt_suffix_thinking status from CLI arguments"""
    args = cliContext.args
    if not args:
        return False
    return getattr(args, "qwen3_enable_prompt_suffix_thinking", False)


@cache
def get_qwen3_chat_template_thinking_status() -> bool:
    """Get qwen3_enable_chat_template_thinking status from CLI arguments"""
    args = cliContext.args
    if not args:
        return False
    return getattr(args, "qwen3_enable_chat_template_thinking", False)


@cache
def get_qwen3_chat_template_thinking_status() -> bool:
    """Get qwen3_enable_chat_template_thinking status from CLI arguments"""
    args = cliContext.args
    if not args:
        return False
    return getattr(args, "qwen3_enable_chat_template_thinking", False)


def _handle_qwen3_chat_template_thinking(request) -> None:
    """Handle Qwen3 chat template thinking logic (hard switch)."""
    if not hasattr(request, "chat_template_kwargs"):
        return
    
    if not hasattr(request.chat_template_kwargs, "enable_thinking"):
        request.chat_template_kwargs.enable_thinking = True
        logger.debug("Applied Qwen3 chat template thinking (hard switch)")


def handle_qwen3_thinking_modes(request) -> None:
    """
    Handle all Qwen3 thinking mode logic (both hard and soft switches).
    Enforces mutual exclusion between different thinking modes.
    """
    qwen3_chat_template_thinking = get_qwen3_chat_template_thinking_status()
    qwen3_prompt_suffix_thinking = get_qwen3_prompt_suffix_thinking_status()
    
    if qwen3_chat_template_thinking and qwen3_prompt_suffix_thinking:
        logger.warning("Both Qwen3 thinking modes enabled, using hard switch only")
        qwen3_prompt_suffix_thinking = False

    if qwen3_chat_template_thinking:
        _handle_qwen3_chat_template_thinking(request)
    elif qwen3_prompt_suffix_thinking:
        _handle_qwen3_prompt_suffix_thinking(request, qwen3_prompt_suffix_thinking)


@cache
def get_default_stop() -> Optional[Union[str, List[str]]]:
    """
    Get default stop from CLI arguments.
    Supports both string and JSON formats.
    Returns base64 decoded stop if provided, otherwise returns plain stop.
    """
    args = cliContext.args
    if not args:
        return None

    # Try base64 first
    if getattr(args, "default_stop_base64", None):
        try:
            decoded = base64.b64decode(args.default_stop_base64).decode("utf-8")
            return parse_stop_string(decoded)
        except:
            pass

    # Try plain stop
    if getattr(args, "default_stop", None):
        return parse_stop_string(args.default_stop)

    return None


@cache
def get_default_stop_token_ids() -> Optional[list]:
    """
    Get default stop token ids from CLI arguments.
    Converts comma separated string to list of integers.
    """
    args = cliContext.args
    if not args:
        return None
    stop_token_ids = getattr(args, "default_stop_token_ids", None)
    if stop_token_ids:
        try:
            return [int(x.strip()) for x in stop_token_ids.split(",")]
        except:
            pass
    return None


def _handle_qwen3_prompt_suffix_thinking(request, qwen3_enable_prompt_suffix_thinking):
    """Handle Qwen3 thinking logic based on model name and enable flag."""
    try:
        # If qwen3 prompt suffix thinking is not enabled, return
        if not qwen3_enable_prompt_suffix_thinking:
            return

        # Ensure messages exist and is a list
        if not hasattr(request, "messages") or not request.messages or not isinstance(request.messages, list):
            return

        last_message = request.messages[-1]

        # Ensure the last message is a dictionary and has content
        if not isinstance(last_message, dict) or "content" not in last_message:
            return

        content = last_message["content"]
        model_name = request.model
        _enable_think = True if model_name.endswith("-think") else False
        if _enable_think:
            #   # Patch SubCliServe
            request.model = model_name.rstrip("-think")
        # Check if content already ends with /think or /nothink followed by optional whitespace
        if THINKING_TAG_REGEX.search(content):
            return

        # Determine the tag to add based on model name
        if _enable_think:
            last_message["content"] += " /think"
        else:
            last_message["content"] += " /no_think"

    except Exception as e:
        logger.error(f"Error processing qwen3 prompt suffix thinking: {e}")


async def patch_check_model(request) -> Optional[JSONResponse]:
    """Patch for check_model in versions < 0.6.2"""
    if version.parse(vllm_version) < version.parse("0.6.2"):
        from vllm.entrypoints.openai.api_server import origin_check_model

    reset_default_request(request=request)
    ret = await origin_check_model(request)
    return ret


async def patch_check_model_v2(self, request):
    """Patch for check_model in versions >= 0.6.2"""
    reset_default_request(request=request)
    ret = await self.origin_check_model(request)
    return ret


async def origin_serving_self_check_model(self, request):
    """Original serving model check implementation"""
    if request.model in self.served_model_names:
        return None
    if request.model in [lora.lora_name for lora in self.lora_requests]:
        return None
    if request.model in [
        prompt_adapter.prompt_adapter_name
        for prompt_adapter in self.prompt_adapter_requests
    ]:
        return None
    return self.create_error_response(
        message=f"The model `{request.model}` does not exist.",
        err_type="NotFoundError",
        status_code=HTTPStatus.NOT_FOUND,
    )


async def patch_serving_self_check_model(self, request):
    """Patch for serving model check"""
    reset_default_request(request=request)
    ret = await self.origin_serving_check_model(request)
    return ret


def parse_stop_string(stop_str: str) -> Union[str, List[str]]:
    """
    Parse stop string input which could be either:
    1. A JSON format string (single string or array)
    2. A plain string if JSON parsing fails

    Args:
        stop_str: Input stop string
    Returns:
        Either a single string or list of strings
    """
    try:
        parsed = json.loads(stop_str)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, str):
            return parsed
        return stop_str
    except:
        return stop_str


def patch_make_arg_parser(*args):
    """Add custom arguments to the argument parser"""
    logger.info("Patching make_arg_parser to add custom arguments")
    parser = origin_make_arg_parser(*args)

    parser.add_argument(
        "--default-model-template", type=str, default=None, help="model template"
    )

    parser.add_argument(
        "--enable-stream-include-usage",
        action="store_true",
        help="Enable the inclusion of stream usage data in the output",
    )

    parser.add_argument(
        "--disable-continuous-usage-stats",
        action="store_true",
        help="Disable continuous collection of usage statistics",
    )

    parser.add_argument(
        "--default-stop",
        type=str,
        default=None,
        help='Default stop string or JSON format (e.g. "<|im_end|>" or \'["<|im_end|>", "<|endoftext|>"]\')',
    )

    parser.add_argument(
        "--default-stop-base64",
        type=str,
        default=None,
        help="Base64 encoded stop string or JSON format",
    )

    parser.add_argument(
        "--default-stop-token-ids",
        type=str,
        default=None,
        help="Comma separated stop token ids (e.g. '151643,151645')",
    )

    parser.add_argument(
        "--qwen3-enable-prompt-suffix-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 thinking mode by appending `/think` or `/nothink` suffix to the prompt (soft switch). "
             "When enabled, models ending with '-think' will automatically append '/think', "
             "while other models will append '/no_think'. Default: off."
    )

    parser.add_argument(
        "--qwen3-enable-chat-template-thinking",
        action="store_true",
        default=False,
        help="Enable Qwen3 chat template thinking mode (hard switch). "
             "When enabled, it will modify the system prompt template directly "
             "if chat_template_kwargs.enable_thinking is not specified in the request. "
             "This provides stronger constraints than the soft switch. "
             "Mutually exclusive with --qwen3-enable-prompt-suffix-thinking. Default: off."
    )

    return parser


def _patch_parse_args(self, *args, **kwargs):
    """Patch for argument parsing with context storage"""
    logger.info("Entering _patch_parse_args")
    # The problematic call that resets args is `parser.parse_args([])`.
    # In this case, the `args` tuple will be `([],)`.
    # We want to avoid storing the result of this specific call.
    # The main entrypoint calls `parse_args()` with no arguments,
    # which means `args` will be an empty tuple `()`.
    is_default_check_parse = args and isinstance(args[0], list) and not args[0]

    parsed_args = self._origin_parse_args(*args, **kwargs)
    logger.info(f"_patch_parse_args Parsed Arguments: {parsed_args}")
    from vllm.entrypoints.openai.cli_context_args import cliContext

    if not is_default_check_parse:
        cliContext.args = parsed_args
        logger.info(f"cliContext.args updated to: {cliContext.args}")
    else:
        logger.info("Skipping update of cliContext.args for default check parse.")

    return parsed_args


def reset_default_request(request) -> None:
    """
    Reset request defaults and apply configuration settings

    Args:
        request: The request object to modify
    """
    include_usage_status = get_stream_include_usage_status()
    if include_usage_status:
        if get_continuous_usage_stats_status():
            continuous_usage_stats = False
        else:
            continuous_usage_stats = True

        if request.stream_options is None:
            request.stream_options = StreamOptions(
                include_usage=True, continuous_usage_stats=continuous_usage_stats
            )

    # Handle Qwen3 thinking modes
    handle_qwen3_thinking_modes(request)

    default_model_template = get_default_model_template()
    if default_model_template:
        request.model = default_model_template

    # Handle default stop
    default_stop = get_default_stop()
    # logger.debug("default_stop: %s", default_stop)
    if default_stop:
        if not hasattr(request, "stop") or not request.stop:
            request.stop = (
                default_stop if isinstance(default_stop, list) else [default_stop]
            )
        else:
            current_stops = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
            new_stops = (
                default_stop if isinstance(default_stop, list) else [default_stop]
            )
            request.stop = current_stops + new_stops

    # Handle default stop token ids
    default_stop_token_ids = get_default_stop_token_ids()
    # logger.debug("default_stop_token_ids: %s", default_stop_token_ids)
    if default_stop_token_ids:
        if not hasattr(request, "stop_token_ids") or not request.stop_token_ids:
            request.stop_token_ids = default_stop_token_ids
        elif isinstance(request.stop_token_ids, list):
            request.stop_token_ids.extend(default_stop_token_ids)


def patch_api_server() -> None:
    """Apply patches to VLLM API server components based on version"""

    logger.info("Loading monkey_patch_api_request_v4")

    try:
        if vllm_version_magic.less_than_0_6_2():
            from vllm.entrypoints.openai.api_server import (
                check_model as origin_check_model,
            )

            OpenAIServingChat = None
        else:
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

            origin_check_model = OpenAIServingChat._check_model
            OpenAIServingChat.origin_check_model = origin_check_model
    except ImportError:
        origin_check_model = None

    if origin_check_model and vllm_version_magic.less_than_0_6_2():
        from vllm.entrypoints.openai import api_server
        api_server.origin_check_model = origin_check_model
        api_server.check_model = patch_check_model

    if vllm_version_magic.greater_than_or_equal_0_6_2() and OpenAIServingChat:
        OpenAIServingChat._check_model = patch_check_model_v2

    if vllm_version_magic.less_than_0_2_7():
        OpenAIServing.origin_serving_check_model = OpenAIServing._check_model
        OpenAIServing._check_model = patch_serving_self_check_model

    if FlexibleArgumentParser:
        FlexibleArgumentParser._origin_parse_args = FlexibleArgumentParser.parse_args
        FlexibleArgumentParser.parse_args = _patch_parse_args

    cli_args.make_arg_parser = patch_make_arg_parser
    for name, module in sys.modules.items():
        if name.startswith("vllm") and hasattr(module, "make_arg_parser"):
        # if name.startswith("vllm.entrypoints.openai.cli_args") and hasattr(module, "make_arg_parser"):
            setattr(module, "make_arg_parser", patch_make_arg_parser)


def examples():
    """Example usage of version comparison"""
    vllm_version = "0.6.4.dev653+g1b5b0be7.d20241125"

    if vllm_version_magic.less_than_0_6_2():
        logger.info("Using legacy import")
    else:
        logger.info("Using new import")

    if vllm_version_magic.less_than("0.7.0"):
        logger.info("Version is less than 0.7.0")


if __name__ == "__main__":
    examples()
