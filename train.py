from src.parse_config import ConfigParser
from src.trainer import Trainer
import argparse


def main(configp: ConfigParser):
    trainer = Trainer(configp)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser('train')
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
        required=True
    )
    config = ConfigParser(args.parse_args().config)
    main(config)
