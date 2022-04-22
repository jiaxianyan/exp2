import argparse
import os
import commons, dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/default_config.yaml',help='path of dataset')
    args = parser.parse_args()

    config = commons.get_config_easydict(args.config_path)

    g = dataset.get_meetup_biparticle_graph(config)

    print(g)
