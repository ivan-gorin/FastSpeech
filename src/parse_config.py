from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import os
import json
from datetime import datetime
from pathlib import Path
from itertools import repeat

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from .dataset import LJSpeechDataset
from .collator import LJSpeechCollator
from .vocoder import Vocoder
from .melspectrogram import MelSpectrogram
from .aligner import GraphemeAligner
from .model import FastSpeech
from .loss import FastSpeechLoss
from .logger import WanDBWriter, TensorboardWriter
from torch.utils.data import DataLoader
from typing import Tuple
import logging
import logging.config
from .utils import ROOT_PATH, read_json


def setup_logging(
        save_dir, log_config=None, default_level=logging.INFO
):
    """
    Setup logging configuration
    """
    if log_config is None:
        log_config = str(ROOT_PATH / "src" / "logger_config.json")
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)


class ConfigParser:
    def __init__(self, config_file):
        with open(config_file, 'rt') as file:
            self.config = json.load(file)

        self.run_id = datetime.now().strftime(r"%m%d_%H_%M_%S_%f")[:-3]
        self.log_dir = Path('log') / self.run_id
        self.save_dir = Path('saved') / self.run_id
        self.tensorboard_dir = Path('.tensorboard') / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.name = self.config['name']
        self.seed = self.config['random_seed']

        self.device = 'cpu'
        if self.config['device'] == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')

        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        if 'resume' in self.config:
            self.resume = Path(self.config['resume'])
            # self.resume_cfg = self.resume.parent / "config.json"
        else:
            self.resume = None

    def __getitem__(self, item: str):
        return self.config[item]

    def get_device(self):
        return self.device

    def get_optimizer(self, model):
        return AdamW(model.parameters(), **self.config['optimizer']['args'])

    def get_scheduler(self, optimizer):
        return OneCycleLR(optimizer, **self.config['lr_scheduler']['args'])

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        dataset = LJSpeechDataset(self.config['data']['root'])
        train_split = int(len(dataset) * self.config['data']['split'])
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_split, len(dataset) - train_split],
                                                                    generator=torch.Generator().manual_seed(self.seed))

        train_dataloader = DataLoader(dataset=train_dataset, collate_fn=LJSpeechCollator(),
                                      **self.config['data']['train'])
        val_dataloader = DataLoader(dataset=test_dataset, collate_fn=LJSpeechCollator(),
                                    **self.config['data']['val'])
        if self.config['trainer']['overfit']:
            train_dataloader = iter(train_dataloader)
            batch = next(train_dataloader)
            train_dataloader = repeat(batch)
        return train_dataloader, val_dataloader

    def get_vocoder(self):
        if not os.path.exists('waveglow_256channels_universal_v5.pt'):
            gdd.download_file_from_google_drive(
                file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
                dest_path='./waveglow_256channels_universal_v5.pt'
            )
        return Vocoder().to(self.device)

    def get_melspectrogram(self):
        return MelSpectrogram(self.config['melspectrogram'], 1.).to(self.device)

    def get_aligner(self):
        sr = self.config['melspectrogram']['sample_rate']
        return GraphemeAligner(sr).to(self.device)

    def get_model(self):
        model = FastSpeech(**self.config['model']).to(self.device)
        return model

    def get_criterion(self):
        return FastSpeechLoss()

    def get_writer(self):
        if self.config['trainer']['visualize'] == 'wandb':
            return WanDBWriter(self.config)
        elif self.config['trainer']['visualize'] == 'tensorboard':
            return TensorboardWriter(self.tensorboard_dir, True)
        else:
            raise NotImplementedError()

    def get_logger(self, name, verbosity=1):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
