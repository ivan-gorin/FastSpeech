# FastSpeech

Tested on python 3.7.12

Setup instructions:
```
pip install -r requirements.txt
chmod +x setup.sh
./setup.sh
```

Run train with 
```
python train.py --config config.json
```

Choose `wandb` or `tensorboard` visualizer in `config.json`
