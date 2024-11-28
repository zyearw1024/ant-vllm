from vllm.patch.monkey_patch_api_request import patch_api_server;patch_api_server()
import asyncio

from vllm.entrypoints.openai.api_server import (
    make_arg_parser,
    run_server,
    FlexibleArgumentParser
)

if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))