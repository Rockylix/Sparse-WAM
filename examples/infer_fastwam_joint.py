"""Run FastWAM-Joint on a recorded LIBERO observation and save one action chunk."""

if __package__:
    from .infer_edge import main as run_policy
else:
    from infer_edge import main as run_policy


def main(argv=None):
    run_policy(argv, backbone="fastwam-joint")


if __name__ == "__main__":
    main()
