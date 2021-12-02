from typing import Tuple, Dict, Optional, List, Union
from itertools import islice
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return {"waveform": waveform, "waveform_length": waveform_length, "transcript": transcript, "tokens": tokens,
                "token_lengths": token_lengths}
