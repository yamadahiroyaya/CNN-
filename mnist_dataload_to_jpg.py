import os
import torch
from torchvision import datasets

def main():
    """
    MNISTデータをjpgで保存するプログラム
    """
    print("start")

    # 保存先フォルダ設定
    rootdir = "./MNIST"
    traindir = os.path.join(rootdir, "trainImgs")
    testdir = os.path.join(rootdir, "testImgs")

    # MNIST データセット読み込み
    train_dataset = datasets.MNIST(root=rootdir, train=True, download=True)
    test_dataset = datasets.MNIST(root=rootdir, train=False, download=True)

    # 画像保存 train
    count = 0
    for img, label in train_dataset:
        os.makedirs(traindir, exist_ok=True)
        img_name = "data" + str(count).zfill(5) + "_ans" + str(label) + ".jpg"
        savepath = os.path.join(traindir, img_name)
        img.save(savepath)
        count += 1
        print(savepath)

    # 画像保存 test
    count = 0
    for img, label in test_dataset:
        os.makedirs(testdir, exist_ok=True)
        img_name = "data" + str(count).zfill(5) + "_ans" + str(label) + ".jpg"
        savepath = os.path.join(testdir, img_name)
        img.save(savepath)
        count += 1
        print(savepath)

    print("finished")

if __name__ == "__main__":
    main()
