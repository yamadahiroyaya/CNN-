import os
import argparse
import glob

import cv2
import torch
import torchvision
import torch.nn.functional as f
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

from model import MyNet # このあと自分で定義するmodel.pyからのネットワーククラス

def Main(args):
    # 計算環境が、CUDA(GPU)か、CPUか
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ' + device)

    # 学習・テスト結果の保存用辞書
    history_train = {
        'train_loss': [],   # 損失関数の値
        'train_acc': [],    # 正解率
    }

    history_test = {
        'test_loss': [],    # 損失関数の値
        'test_acc': [],     # 正解率
    }

    # ネットワークを構築（ : torch.nn.Module は型アノテーション）
    # 変数netに構築するMyNet()は「ネットワークの実装」で定義します
    net : torch.nn.Module = MyNet()
    net = net.to(device) # GPUあるいはCPUに合わせて再構成

    # データローダー・データ数を取得
    #（load_dataは「学習データ・テストデータのロードの実装（データローダー）」の章で定義します）
    train_dir = args.trainDir # 学習用画像があるディレクトリパス
    test_dir = args.testDir   # テスト用画像があるディレクトリパス
    train_loaders, train_data_size = load_data(args.trainDir)
    test_loaders, test_data_size = load_data(args.testDir)
    
    # オプティマイザを設定
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    
    # エポック数（学習回数）
    epoch = args.epoch

    # 学習・テストを実行
    for e in range(epoch):
        # 以下2つの関数は「学習の実装」「テストの実装」の章で定義します
        cnn_train(net, device, train_loaders, train_data_size, optimizer, e, history_train)
        cnn_test(net, device, test_loaders, test_data_size, e, epoch, history_test)

    # 学習済みパラメータを保存
    torch.save(net.state_dict(), 'params_cnn.pth')

    # 結果を出力（「結果出力の実装」の章で定義します）
    output_graph(epoch, history_train, history_test)

if __name__ == '__main__':
    """
    学習を行うプログラム。

    trainDir : 学習用画像があるディレクトリパス
    testDir  : テスト用画像があるディレクトリパス
    epoch    : エポック数
    """

    # 起動引数設定
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--trainDir", type=str, default="./MNIST/trainImgs")
    parser.add_argument("-ts", "--testDir", type=str, default="./MNIST/testImgs")
    parser.add_argument("-ep", "--epoch", type=int, default=10)
    args = parser.parse_args()

    # メイン関数
    Main(args)
