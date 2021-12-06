from src.parse_config import ConfigParser
import torch
import torchaudio
import argparse


@torch.no_grad()
def main(config: ConfigParser, resume_path: str):
    logger = config.get_logger('Test')
    model = config.get_model()
    device = config.device
    vocoder = config.get_vocoder()
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path, device)

    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("Checkpoint Loaded.")
    model.eval()
    test_sents = config['test_sents']
    tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    translate_table = str.maketrans('èêéâàü', 'eeeaau', '“’]"”[')
    for i in range(len(test_sents)):
        test_sents[i] = test_sents[i].translate(translate_table)

    test_tokens, _ = tokenizer(test_sents)
    test_tokens = test_tokens.to(device)

    logger.info("Synthesizing audio...")
    pred_specs, _ = model(test_tokens)
    audios = vocoder.inference(pred_specs).cpu()
    logger.info("Audio synthesized. Saving...")
    for i in range(audios.shape[0]):
        path = config.save_dir / f'SynthesizedAudio{i}.wav'
        sr = config['melspectrogram']['sample_rate']
        torchaudio.save(path, audios[i].unsqueeze(0), sr)
    logger.info("Saved in {}".format(config.save_dir))


if __name__ == '__main__':
    args = argparse.ArgumentParser('test')
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
        required=True
    )
    args.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="model checkpoint file path (default: None)",
        required=True
    )
    parsed = args.parse_args()
    configparser = ConfigParser(parsed.config)
    main(configparser, parsed.model)
