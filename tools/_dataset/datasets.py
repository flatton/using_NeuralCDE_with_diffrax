import csv
import glob
import math
from typing import Tuple, Sequence

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from ..config import MNISTStrokeDatasetConfig

class BaseDataset():
    def __init__(self):
        pass

    def make_dataset(self) -> Tuple[Sequence[Float], Sequence[Array], Sequence[Array], Sequence[Float], int]:
        dataset_size = 256
        sequence_length = 100
        channels = 2
        ts = jnp.zeros((dataset_size, sequence_length))
        ys = jnp.zeros((dataset_size, sequence_length, channels+1))
        coeffs = [jnp.zeros((dataset_size, sequence_length-1, channels+1)) for _ in jnp.arange(4)]
        labels = jnp.zeros((dataset_size,))
        in_size = ys[0][0]
        return ts, ys, coeffs, labels, in_size
    
class SpiralDataset(BaseDataset):
    def __init__(self, dataset_size: int, length: int = 100, add_noise: bool = False, *, key: PRNGKeyArray):
        super().__init__()
        self.dataset_size = dataset_size
        self.length = length
        self.add_noise = add_noise
        self.key = key

    def make_dataset(self) -> Tuple[Sequence[Float], Sequence[Array], Sequence[Array], Sequence[Float], int]:
        """
            二次元の時計回り/反時計回り渦のデータを作成. ラベルは時計回りか反時計回りかの二値ラベル.
        """
        # データセットサイズを検証
        assert self.dataset_size > 0, "The dataset_size must be a positive value."
        dataset_size = self.dataset_size
        
        # y0 を作成
        ## 乱数生成キーを分割し, 独立した新たなキーを作成
        theta_key, noise_key, drop_key = jr.split(self.key, 3) 
        ## 一様分布の乱数を作成. 0~2πの間で dataset_size 次元(入力チャネル数)の乱数を作成している.
        theta = jr.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi) # -> (256,)
        # ランダムな系列 theta を cos, sin に入力して三角関数をランダムサンプリング.
        y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1) # -> y0: (256, 2)
    
        # ts を作成
        ## タイムスタンプ系列を作成. `broadcast_to` で "入力チャネル数の次元を追加(`reshape`)して, `dataset_size` に合うように `tile` "している.
        ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, self.length), (dataset_size, self.length)) # -> (256, 100)
    
        # ys を作成
        ## y(0) = y0, d/dt y(t) = -Ay の初期値問題の解 y(t) = y(0) exp(-At) で考えると, A = -matrix となっている. 式の形状は対数螺旋になっている.
        matrix = jnp.array([[-0.3, 2], [-2, -0.3]])
        ## `jsp.linalg.expm` は行列指数関数, `jax.vmap` は並列処理用の関数
        ys = jax.vmap(
            lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
        )(y0, ts) # -> (256, 100, 2)
        ## チャネルにタイムスタンプのチャネルを追加
        ys = jnp.concatenate([ts[:, :, None], ys], axis=-1) # -> (256, 100, 3)
        ## データセットの半量のデータについて, 渦の y 軸の値を正負反転させ, 反時計のデータから時計周りのデータを作成.
        ys = ys.at[: dataset_size // 2, :, 1].multiply(-1)
        ## データに白色ノイズを付与
        if self.add_noise:
            ys = ys + jr.normal(noise_key, ys.shape) * 0.01 
            
        # 欠損値ありのデータ用の補間関数: Hermite cubic splines with backward differences
        ## 欠損値は NaN で表現. 出力は元々のタイムスタンプの間を補間したものになり, 系列長 T のデータを入力すると T-1 のデータが出力される.
        ## 3次スプライン補間の係数 (batchsize (d, c, b, a)) = coeffs　が生成される. これは3次スプライン補間 CubicInterpolation でパスを形成する前に使用する必要がある.
        coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys) # -> (256, 256*99*3, 256*99*3, 256*99*3)
    
        # 0/1の2値ラベルの作成
        labels = jnp.zeros((dataset_size,))
        labels = labels.at[: dataset_size // 2].set(1.0)
    
        # データのチャネル数を取得
        _, _, in_size = ys.shape
        
        return ts, ys, coeffs, labels, in_size

class MNISTStrokeDataset(BaseDataset):
    def __init__(self, dataset_size: int, mode_train: bool):
        super().__init__()
        self.dataset_size = dataset_size
        self.mode_train = mode_train
        self.config = MNISTStrokeDatasetConfig()

    def make_dataset(self) -> Tuple[Sequence[Float], Sequence[Array], Sequence[Array], Sequence[Float], int]:
        # データセットの読み込み
        if self.mode_train:
            path_list_data_input_sequence = glob.glob(self.config.train_data_input_sequence, recursive=True)
            path_label = self.config.train_label
        else: 
            path_list_data_input_sequence = glob.glob(self.config.test_data_input_sequence, recursive=True)
            path_label = self.config.test_label
    
        # データセットサイズを規定. 保有するデータ量よりもサイズが大きい or 負の値を指定したときは, データセットの全量を使用するとみなす.
        dataset_size = len(path_list_data_input_sequence) if (self.dataset_size < 0 or self.dataset_size > len(path_list_data_input_sequence)) else self.dataset_size
    
        # ts, ys の作成
        ys = []
        ts = []
        for filepath in tqdm(path_list_data_input_sequence[:dataset_size]):
            with open(filepath) as f:
                lines = [[jnp.int32(x) for x in row] for row in csv.reader(f, delimiter=' ')]
                ts.append(jnp.arange(len(lines)))
                _ys = jnp.asarray(lines)
                ys.append(jnp.concatenate([ts[-1][:, None], _ys], axis=-1))
    
        # 欠損値ありのデータ用の補間関数: Hermite cubic splines with backward differences
        coeffs = []
        for _ts, _ys in zip(ts, ys):
            coeffs.append(diffrax.backward_hermite_coefficients(_ts, _ys))
    
        # ラベルデータの読み込み
        with open(path_label) as f:
            labels = [[jnp.int32(x) for x in row] for row in csv.reader(f, delimiter=' ')]
    
        # データのチャネル数を取得
        in_size = len(ys[0][0])
        
        return ts, ys, coeffs, labels, in_size