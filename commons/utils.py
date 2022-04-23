import logging
import os
import yaml
from easydict import EasyDict

def get_config_easydict(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)

def get_logger(run_dir):
    """
    Set the logger
    """
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    logfile_name = os.path.join(run_dir, 'log.txt')

    fmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(fmt, filedatefmt)
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(fmt, sdatefmt)

    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(sformatter)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    return logging.getLogger()

def output_results_sequence(matrics_sequence):
    print('===========result sequence==============')
    for index,matrics in enumerate(matrics_sequence):
        print('No.{}, AUC : {:.4f}, acc : {:.4f}, recall : {:.4f}, F1 : {:.4f}'.format(index+1,matrics[0],matrics[1],matrics[2],matrics[3]))
