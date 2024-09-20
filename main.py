import yaml
import argparse
from omegaconf import OmegaConf

from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    
    return args



if __name__ == '__main__':
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    config = OmegaConf.create(config)
    trainer = Trainer(config)

    trainer.run()
    
