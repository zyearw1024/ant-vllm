from vllm.patch.monkey_patch_api_request import patch_api_server;patch_api_server()

from vllm.entrypoints.openai.api_server import (
    make_arg_parser,
    run_server,
    cli_env_setup,
    validate_parsed_serve_args,
    FlexibleArgumentParser,
    uvloop,
)

if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
