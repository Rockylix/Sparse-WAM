"""Public CLI. Config inspection and source audit do not load model weights."""

import argparse
import json

from .config import BACKBONES, SparseWAMConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sparse-WAM policy inference")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("config", help="Print the effective Sparse-WAM configuration")
    show.add_argument("--backbone", choices=BACKBONES, required=True)
    show.add_argument("--execution", choices=("eager", "compile", "compile-graph"), default="eager")
    show.add_argument("--no-nano-packing-cache", action="store_true")
    server = commands.add_parser("serve", help="Start the RoboLab/openpi policy server")
    server.add_argument("--backbone", choices=("edge", "nano"), required=True)
    server.add_argument("--execution", choices=("eager", "compile", "compile-graph"), default="eager")
    server.add_argument("--no-nano-packing-cache", action="store_true")
    server.add_argument("--mode", choices=("dense", "sparse"), default="sparse")
    server.add_argument("--checkpoint-path", dest="checkpoint")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--seed", type=int, default=0)
    server.add_argument("--output-dir", default="outputs/server")
    server.add_argument("--native-config", help="JSON overrides for native checkpoint/domain/assets settings")
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    config = SparseWAMConfig(
        options.pop("backbone"),
        options.pop("execution"),
        nano_packing_cache=not options.pop("no_nano_packing_cache"),
    )
    if command == "config":
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
    else:
        from .service import serve

        serve(config, **options)


if __name__ == "__main__":
    main()
