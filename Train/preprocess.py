import h5py
import os
from config import ArgModel
import logging
from collections import Counter
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self):
        self.init_token = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>', '<SPLIT>']
        self.t2i = {v: i for i, v in enumerate(self.init_token)}
        self.i2t = {i: v for i, v in enumerate(self.init_token)}
        self.raw_file = 'Train/raw_folder/raw.txt'
        self.hdf_file = 'Train/processed.hdf'

    def build_vocab(self):
        logger.info('start build vocab')
        counter = Counter()
        counter.update(self.init_token)
        with open(self.raw_file) as reader:
            for raw_line in tqdm(reader,desc='collecting tokens'):
                line = raw_line.split(' ')
                counter.update(line)
        rare_word = []
        for i, v in counter.items():
            if i not in self.init_token and v>=5:
                index = len(self.t2i)
                self.t2i[i] = index
                self.i2t[index] = i
            else:
                rare_word.append(i)
        logger.info(f'vocab built. num token:{len(self.t2i)}, rare token num:{len(rare_word)}')

    def build_hdf5(self):
        logger.info('start build hdf5 file')
        with h5py.File(self.hdf_file, 'w') as hdf:
            hdf.create_dataset('tokens',)

def build_vocab():
    pass


with open('Train/raw_folder/raw.txt') as reader:
    for raw_line in reader:
        line = raw_line.split(' ')


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.build_vocab()
    preprocess.build_hdf5()
