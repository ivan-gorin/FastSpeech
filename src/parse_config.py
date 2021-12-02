from google_drive_downloader import GoogleDriveDownloader as gdd
import torch
import os
import json

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


class ConfigParser:
    def __init__(self, config_file):
        with open(config_file, 'rt') as file:
            self.config = json.load(file)

        self.config = self.config
        self.name = self.config['name']
        self.seed = self.config['random_seed']

        self.device = 'cpu'
        if self.config['device'] == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')

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

    def get_logger(self):
        if self.config['trainer']['visualize'] == 'wandb':
            return WanDBWriter(self.config)
        elif self.config['trainer']['visualize'] == 'tensorboard':
            return TensorboardWriter("./log", True)
        else:
            raise NotImplementedError()