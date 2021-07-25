import logging
import sys
from utils.config import Config

sys.path.append("..")


class Msg:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    @staticmethod
    # print the log of train config
    def training_conf(conf: Config):
        logging.info(f'''Starting training:
            Epochs:          {conf.epochs}
            Batch size:      {conf.batch_size}
            Learning rate:   {conf.lr}
            Device:          {conf.device.type}
            network:         {conf.network}
            optimizer:       {conf.optimizer}
            sec_network:     {conf.second_network}
            
        ''')

    @staticmethod
    def num_dataset(n_train, n_val):
        logging.info(f'''
            Training size:   {n_train}
            Validation size: {n_val}
        ''')

    @staticmethod
    def end():
        logging.info(f'''end''')

    @staticmethod
    def norm(note, msg):
        logging.info(f"{note}:{msg}")