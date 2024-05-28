import torch
import torch.nn.functional as f

class MyNet(torch.nn.Module):
    """
    CNNを用いた、オライリーP.229 7-5章と同じモデル。
    バッチの画像群に対し、
    入力(batch_size, 3, 28, 28) - conv2d - ReLU - Pooling - Affine - ReLU - Affine - Softmax
    を計算する。

    Returns:
        float: 10個の要素に対する確率を格納した配列。
    """
    
    def __init__(self, input_dim=(3, 28, 28)):
        """
        ネットワークで使用する関数の定義。
        """
        # モジュールの継承
        super(MyNet, self).__init__()
        
        filter_num = 30 # フィルター数
        filter_size = 5 # フィルター（カーネル）サイズ

        # conv2d定義
        self.conv1 = torch.nn.Conv2d(in_channels=input_dim[0], out_channels=filter_num, kernel_size=filter_size)
    
        # ReLU定義
        self.relu = torch.nn.ReLU()

        # Pooling定義
        pooling_size = 2
        self.pool = torch.nn.MaxPool2d(pooling_size, stride=pooling_size)

        # Affine（全結合）レイヤ定義
        fc1_size = (input_dim[1] - filter_size + 1) // pooling_size # self.pool終了時の縦横サイズ = 12
        self.fc1 = torch.nn.Linear(filter_num * fc1_size * fc1_size, 100)   # ノード100個に出力
        self.fc2 = torch.nn.Linear(100, 10)                         # ノード10個に出力
    
    def forward(self, x):
        """
        順伝播の定義。
        """
        # conv2d - ReLU - Poolingまで
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 全結合層に入れるため、バッチ内の各画像データをそれぞれ一列にする
        x = x.view(x.size()[0], -1) 
        
        # Affine - ReLU - Affine
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # ソフトマックスにより、各ノードの確率を結果として返す
        return f.softmax(x, dim=1) # dim=1はバッチ内の各画像データごとにsoftmaxをするため


