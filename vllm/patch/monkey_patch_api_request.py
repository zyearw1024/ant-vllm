import sys
import json
import base64
import logging
import re
import threading
from typing import Optional, Union, List, Any

from vllm import __version__ as vllm_version
from packaging import version
from fastapi.responses import JSONResponse
from functools import cache
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import StreamOptions
from vllm.entrypoints.openai import cli_args
from vllm.entrypoints.openai.cli_context_args import cliContext

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
THINKING_TAG_REGEX = re.compile(r"/(think|nothink|no_think)\s*$")


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

# Thread lock for cache invalidation safety
_cache_lock = threading.RLock()


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




def _handle_qwen3_chat_template_thinking(request: Any) -> None:
    """Handle Qwen3 chat template thinking logic (hard switch)."""
    try:
        # Initialize enable_thinking flag
        _enable_think = False
        
        # Handle model name compatibility: support -think suffix like soft switch
        if hasattr(request, "model") and request.model:
            model_name = request.model
            # Check if model name ends with '-think' (enable thinking)
            _enable_think = model_name.endswith("-think")
            
            if _enable_think:
                # Remove '-think' suffix from model name for actual model lookup
                request.model = model_name.rstrip("-think")
                logger.debug(f"Model name changed from '{model_name}' to '{request.model}' (thinking enabled via -think suffix)")
        
        # Check if request has chat_template_kwargs attribute
        if not hasattr(request, "chat_template_kwargs"):
            # Create chat_template_kwargs as a dictionary if it doesn't exist
            request.chat_template_kwargs = {}
        
        # Handle case where chat_template_kwargs is None
        if request.chat_template_kwargs is None:
            request.chat_template_kwargs = {}
        
        # Handle case where chat_template_kwargs is a SimpleNamespace (convert to dict)
        if hasattr(request.chat_template_kwargs, '__dict__') and not isinstance(request.chat_template_kwargs, dict):
            # Convert SimpleNamespace to dict
            request.chat_template_kwargs = vars(request.chat_template_kwargs)
        
        # Ensure it's a dictionary
        if not isinstance(request.chat_template_kwargs, dict):
            request.chat_template_kwargs = {}
        
        # Check if enable_thinking is already set
        if "enable_thinking" not in request.chat_template_kwargs:
            # Set enable_thinking based on model name suffix
            # Only enable thinking if model name ends with '-think'
            request.chat_template_kwargs["enable_thinking"] = _enable_think
            logger.debug(f"Applied Qwen3 chat template thinking (hard switch): enable_thinking={_enable_think}")
        else:
            logger.debug(f"Qwen3 chat template thinking already set: {request.chat_template_kwargs['enable_thinking']}")
            
    except Exception as e:
        logger.debug(f"Error processing qwen3 chat template thinking: {e}")


def handle_qwen3_thinking_modes(request: Any) -> None:
    """
    Handle all Qwen3 thinking mode logic (both hard and soft switches).
    Enforces mutual exclusion between different thinking modes.
    Default behavior: no thinking mode enabled unless explicitly requested.
    
    Args:
        request: The request object to modify
    """
    try:
        qwen3_chat_template_thinking = get_qwen3_chat_template_thinking_status()
        qwen3_prompt_suffix_thinking = get_qwen3_prompt_suffix_thinking_status()
        
        # Mutual exclusion: hard switch takes priority over soft switch
        if qwen3_chat_template_thinking and qwen3_prompt_suffix_thinking:
            logger.debug("Both Qwen3 thinking modes enabled, using hard switch only")
            qwen3_prompt_suffix_thinking = False

        # Process thinking modes in order of priority
        if qwen3_chat_template_thinking:
            _handle_qwen3_chat_template_thinking(request)
        elif qwen3_prompt_suffix_thinking:
            _handle_qwen3_prompt_suffix_thinking(request, qwen3_prompt_suffix_thinking)
        else:
            # Default behavior: no thinking mode enabled
            logger.debug("No Qwen3 thinking modes enabled, using default behavior")
            
    except Exception as e:
        logger.debug(f"Error handling qwen3 thinking modes: {e}")


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
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.debug(f"Failed to decode base64 stop string: {e}")
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
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse stop token ids: {e}")
            pass
    return None


def _handle_qwen3_prompt_suffix_thinking(request: Any, qwen3_enable_prompt_suffix_thinking: bool) -> None:
    """
    Handle Qwen3 thinking logic based on model name and enable flag (soft switch).
    Default behavior: no thinking mode unless model name ends with '-think'.
    
    Args:
        request: The request object to modify
        qwen3_enable_prompt_suffix_thinking: Whether prompt suffix thinking is enabled
    """
    try:
        # If qwen3 prompt suffix thinking is not enabled, return early
        if not qwen3_enable_prompt_suffix_thinking:
            logger.debug("Qwen3 prompt suffix thinking disabled, skipping")
            return

        # Ensure request has required attributes
        if not hasattr(request, "messages") or not request.messages or not isinstance(request.messages, list):
            logger.debug("Request missing valid messages, skipping qwen3 prompt suffix thinking")
            return

        # Ensure request has model attribute
        if not hasattr(request, "model") or not request.model:
            logger.debug("Request missing model name, skipping qwen3 prompt suffix thinking")
            return

        last_message = request.messages[-1]

        # Ensure the last message is valid
        if not isinstance(last_message, dict) or "content" not in last_message:
            logger.debug("Last message invalid, skipping qwen3 prompt suffix thinking")
            return

        content = last_message["content"]
        model_name = request.model
        
        # Check if model name ends with '-think' (enable thinking)
        _enable_think = model_name.endswith("-think")
        
        if _enable_think:
            # Remove '-think' suffix from model name for actual model lookup
            request.model = model_name.rstrip("-think")
            logger.debug(f"Model name changed from '{model_name}' to '{request.model}' (thinking enabled)")
        
        # Check if content already has thinking tags to avoid duplication
        if THINKING_TAG_REGEX.search(content):
            logger.debug("Content already contains thinking tags, skipping modification")
            return

        # Add appropriate thinking tag based on model name
        if _enable_think:
            last_message["content"] += " /think"
            logger.debug("Added '/think' suffix to prompt (thinking enabled)")
        else:
            # Default behavior: explicitly disable thinking for non-think models
            last_message["content"] += " /no_think"
            logger.debug("Added '/no_think' suffix to prompt (thinking disabled by default)")

    except Exception as e:
        logger.debug(f"Error processing qwen3 prompt suffix thinking: {e}")


async def patch_check_model(request: Any) -> Optional[JSONResponse]:
    """
    Patch for check_model in versions < 0.6.2
    
    Args:
        request: The request object to check and modify
        
    Returns:
        Optional JSONResponse if model check fails
    """
    if version.parse(vllm_version) < version.parse("0.6.2"):
        from vllm.entrypoints.openai.api_server import origin_check_model

    reset_default_request(request=request)
    ret = await origin_check_model(request)
    return ret


async def patch_check_model_v2(self: Any, request: Any) -> Optional[JSONResponse]:
    """
    Patch for check_model in versions >= 0.6.2
    
    Args:
        self: The serving instance
        request: The request object to check and modify
        
    Returns:
        Optional JSONResponse if model check fails
    """
    reset_default_request(request=request)
    ret = await self.origin_check_model(request)
    return ret


async def origin_serving_self_check_model(self: Any, request: Any) -> Optional[JSONResponse]:
    """
    Original serving model check implementation
    
    Args:
        self: The serving instance
        request: The request object to check
        
    Returns:
        Optional JSONResponse if model check fails
    """
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


async def patch_serving_self_check_model(self: Any, request: Any) -> Optional[JSONResponse]:
    """
    Patch for serving model check
    
    Args:
        self: The serving instance
        request: The request object to check and modify
        
    Returns:
        Optional JSONResponse if model check fails
    """
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
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse stop string as JSON: {e}")
        return stop_str


def patch_make_arg_parser(*args: Any) -> Any:
    """
    Add custom arguments to the argument parser
    
    Args:
        *args: Arguments passed to the original make_arg_parser
        
    Returns:
        The modified argument parser
    """
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


def _patch_parse_args(self: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Patch for argument parsing with context storage
    
    Args:
        self: The parser instance
        *args: Arguments passed to parse_args
        **kwargs: Keyword arguments passed to parse_args
        
    Returns:
        Parsed arguments
    """
    logger.info("Entering _patch_parse_args")
    # The problematic call that resets args is `parser.parse_args([])`.
    # In this case, the `args` tuple will be `([],)`.
    # We want to avoid storing the result of this specific call.
    # The main entrypoint calls `parse_args()` with no arguments,
    # which means `args` will be an empty tuple `()`.
    is_default_check_parse = args and isinstance(args[0], list) and not args[0]

    parsed_args = self._origin_parse_args(*args, **kwargs)
    logger.info(f"_patch_parse_args Parsed Arguments: {parsed_args}")

    if not is_default_check_parse:
        with _cache_lock:
            cliContext.args = parsed_args
        logger.info(f"cliContext.args updated to: {cliContext.args}")
    else:
        logger.info("Skipping update of cliContext.args for default check parse.")

    return parsed_args


def _apply_default_list_value(request: Any, attr_name: str, default_value: Union[str, List[str]]) -> None:
    """
    Apply default value to request attribute if not present or empty
    
    Args:
        request: The request object to modify
        attr_name: The attribute name to set
        default_value: The default value to apply
    """
    if not hasattr(request, attr_name) or not getattr(request, attr_name):
        setattr(request, attr_name, default_value if isinstance(default_value, list) else [default_value])
    else:
        current_values = getattr(request, attr_name)
        current_list = current_values if isinstance(current_values, list) else [current_values]
        new_list = default_value if isinstance(default_value, list) else [default_value]
        setattr(request, attr_name, current_list + new_list)


def reset_default_request(request: Any) -> None:
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
    logger.debug("default_stop: %s", default_stop)
    if default_stop:
        _apply_default_list_value(request, "stop", default_stop)

    # Handle default stop token ids
    default_stop_token_ids = get_default_stop_token_ids()
    logger.debug("default_stop_token_ids: %s", default_stop_token_ids)
    if default_stop_token_ids:
        if not hasattr(request, "stop_token_ids") or not request.stop_token_ids:
            request.stop_token_ids = default_stop_token_ids
        elif isinstance(request.stop_token_ids, list):
            request.stop_token_ids.extend(default_stop_token_ids)


def patch_api_server() -> None:
    """
    Apply patches to VLLM API server components based on version.
    This function modifies various VLLM components to add custom functionality
    including argument parsing, model checking, and request processing.
    """
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


def _examples() -> None:
    """Example usage of version comparison - for testing purposes only"""
    test_version = "0.6.4.dev653+g1b5b0be7.d20241125"

    if vllm_version_magic.less_than_0_6_2():
        logger.info("Using legacy import")
    else:
        logger.info("Using new import")

    if vllm_version_magic.less_than("0.7.0"):
        logger.info("Version is less than 0.7.0")


if __name__ == "__main__":
    _examples()
