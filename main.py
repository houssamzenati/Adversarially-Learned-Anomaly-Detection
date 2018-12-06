#! python3

import argparse
import importlib
import logging
import os
import shutil
import urllib3
import zipfile
# import data

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("AnomalyDetection")


def run(args):
    print("""
 ______   _____       _____       ____   
|_     `.|_   _|     / ___ `.   .'    '. 
  | | `. \ | |      |_/___) |  |  .--.  |
  | |  | | | |   _   .'____.'  | |    | |
 _| |_.' /_| |__/ | / /_____  _|  `--'  |
|______.'|________| |_______|(_)'.____.' 
                                         
""")

    has_effect = False

    if args.model and args.dataset and args.split:
        try:

            mod_name = "{}.{}".format(args.model, args.split)

            logger.info("Running script at {}".format(mod_name))

            mod = importlib.import_module(mod_name)

            mod.run(args)

        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error("Script halted without any effect. To run code, use command:\npython3 main.py <example name> {train, test}")

def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomaly Detector.')
    parser.add_argument('model', nargs="?", type=path, help='the folder name of the example you want to run e.g gan or bigan')
    parser.add_argument('dataset', nargs="?", choices=['kdd','cifar10', 'svhn', 'arrhythmia'], help='the name of the dataset you want to run the experiments on')

    parser.add_argument('split', nargs="?", choices=['run'])
    parser.add_argument('--nb_epochs', nargs="?", type=int, default=0, help='number of epochs you want to train the dataset on')
    parser.add_argument('--gpu', nargs="?", type=int, default=0, help='which gpu to use')
    #anomaly
    parser.add_argument('--label', nargs="?", type=int, default=0, help='anomalous label for the experiment')
    parser.add_argument('--m', nargs="?", default='fm',  choices=['cross-e', 'fm'],
                        help='mode/method for discriminator loss')
    parser.add_argument('--w', nargs="?", type=float, default=0.1, help='weight for AnoGAN')
    parser.add_argument('--d', nargs="?", type=int, default=1, help='degree for the L norm')
    parser.add_argument('--rd', nargs="?", type=int, default=42,  help='random_seed')
    parser.add_argument('--enable_sm', action='store_true',  help='enable TF summaries')
    parser.add_argument('--enable_dzz', action='store_true', help='enable dzz discriminator')
    parser.add_argument('--enable_early_stop', action='store_true', help='enable early_stopping')
    parser.add_argument('--sn', action='store_true', help='enable spectral_norm')
    # args for dagmm
    parser.add_argument('--K', nargs="?", type=float, default=-1, help='number of mixtures in GMM')
    parser.add_argument('--l1', nargs="?", type=float, default=-1, help='weight of the energy in DAGMM')
    parser.add_argument('--l2', nargs="?", type=float, default=-1, help='weight of the penalty of diag term in DAGMM')

    run(parser.parse_args())
