import torch

from src.parse_config import ConfigParser
import argparse
from tqdm import tqdm


def main(config: ConfigParser):
    print('_' * 50, 'TRAIN', '_' * 50)
    train_dataloader, val_dataloader = config.get_dataloaders()
    Logger = config.get_logger()
    Model = config.get_model()
    Optimizer = config.get_optimizer(Model)
    Scheduler = config.get_scheduler(Optimizer)
    MelSpectrogram = config.get_melspectrogram()
    Aligner = config.get_aligner()
    Criterion = config.get_criterion()
    print(Model)
    print('Number of parameters', sum(p.numel() for p in Model.parameters()))
    n_epoch = config['trainer']['n_epoch']
    device = config.get_device()

    for epoch in range(n_epoch):
        # train epoch
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            # {"waveform": waveform, "waveform_length": waveform_length, "transcript": transcript, "tokens": tokens,
            #                 "token_lengths": token_lengths}
            Logger.set_step(Logger.step + 1, 'train')
            waveform = batch['waveform'].to(device)
            waveform_length = batch['waveform_length']
            transcript = batch['transcript']
            tokens = batch['tokens'].to(device)
            token_lengths = batch['token_lengths']
            specs = MelSpectrogram(waveform)
            with torch.no_grad():
                durations = Aligner(waveform, waveform_length, transcript)
            spec_length = (waveform_length // config['melspectrogram']['hop_length'] + 1).unsqueeze(-1)
            durations = (durations * spec_length).to(device)

            Optimizer.zero_grad()
            pred_specs, pred_durations = Model(tokens, durations)
            duration_loss, spec_loss = Criterion(durations, pred_durations, specs, pred_specs)
            loss = duration_loss + spec_loss
            Logger.add_scalar("Duration Loss", duration_loss)
            Logger.add_scalar("Spec Loss", spec_loss)
            loss.backward()
            Optimizer.step()
            # break

        Scheduler.step()


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
