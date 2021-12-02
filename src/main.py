from dataset import LJSpeechDataset


def main():
    ds = LJSpeechDataset('../data')
    test = ds[0]
    print(test)


if __name__ == '__main__':
    main()
