if ! [[ -d "./third_party/waveglow" ]]; then
    git clone https://github.com/NVIDIA/waveglow.git ./third_party/waveglow
fi
mkdir data
cd data
if ! [[ -d "LJSpeech-1.1" ]]; then
    if ! [[ -f "LJSpeech-1.1.tar.bz2" ]]; then
        wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    fi
    echo "Unpacking archive."
    tar -xjf LJSpeech-1.1.tar.bz2
fi