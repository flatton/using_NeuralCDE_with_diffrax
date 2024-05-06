import re
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
from natsort import natsorted

from ..config import MNISTStrokeDatasetConfig

class BaseDataset():
    """Base class of the class for creating or loading records.
    データセットの作成や読み込みを行うクラスで継承されるクラス.
    """
    def __init__(self):
        pass

    def make_dataset(self) -> Tuple[Float[Array, "dataset sequence"], Float[Array, "dataset sequence channels+1"], Tuple[Float[Array, "dataset sequence-1 channels+1"]], Float[Array, "dataset None"], int]:
        """Creat or load records.
        データセットの作成や読み込みを行う関数.

        **Returns:**
        - ts: タイムスタンプ.
        - ys: レコード.
        - coeffs: 補間で使用される係数. [Handling missing data](https://docs.kidger.site/diffrax/api/interpolation/#handling-missing-data)
        - labels: 正解ラベル.
        - in_size: 入力データのチャネル数. ただし, タイムスタンプを含めたチャネル数である. モデルの入力チャネル数を決定するために使用される.
        """
        dataset_size = 256
        sequence_length = 100
        channels = 2
        ts = jnp.zeros((dataset_size, sequence_length))
        ys = jnp.zeros((dataset_size, sequence_length, channels+1))
        coeffs = tuple([jnp.zeros((dataset_size, sequence_length-1, channels+1)) for _ in jnp.arange(4)])
        labels = jnp.zeros((dataset_size,))
        _, _, in_size = ys.shape
        return ts, ys, coeffs, labels, in_size
    
class SpiralDataset(BaseDataset):
    """Class for creating 2D clockwise/counterclockwise vortices data.
    二次元の時計回り/反時計回り渦のデータを作成. ラベルは時計回りか反時計回りかの二値ラベル.
    """
    dataset_size: int
    length: int
    add_noise: bool
    interpolation: str
    key: PRNGKeyArray
    
    def __init__(self, dataset_size: int, length: int = 100, add_noise: bool = False, interpolation: str = 'cubic', *, key: PRNGKeyArray):
        """**Arguments:**
        - dataset_size: サンプル数.
        - length: 作成する時系列データの系列長.
        - add_noise: ノイズを付与するか否かのフラグ.
        - interpolation: 補間形式の指定. [Interpolations](https://docs.kidger.site/diffrax/api/interpolation/#interpolations)
        - key: a pseudo-random number generator (;PRNG) key. `key = jax.random.PRNGKey(seed)`
        """
        super().__init__()
        self.dataset_size = dataset_size
        self.length = length
        self.add_noise = add_noise
        self.key = key
        self.interpolation = interpolation

    def make_dataset(self) -> Tuple[Float[Array, "dataset sequence"], Float[Array, "dataset sequence channels+1"], Tuple[Float[Array, "dataset sequence-1 channels+1"]], Float[Array, "dataset None"], int]:
        """Creat or load records.
        二次元の時計回り/反時計回り渦のデータを作成する関数.
        
        **Returns:**
        - ts: タイムスタンプ.
        - ys: レコード. 2次元の渦の軌跡とタイムスタンプからなる, 3チャネルの時系列データ.
        - coeffs: 補間で使用される係数. [Handling missing data](https://docs.kidger.site/diffrax/api/interpolation/#handling-missing-data)
        - labels: 正解ラベル. 時計回りか反時計回りかの二値ラベル.
        - in_size: 入力データのチャネル数. ただし, タイムスタンプを含めたチャネル数である. モデルの入力チャネル数を決定するために使用される.
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
            
        # 欠損値ありのデータ用の補間
        """**NOTE:**
        欠損値ありのデータ用の補間. 欠損値は NaN で表現. 出力は元々のタイムスタンプの間を補間したものになり, 
        系列長 $$T$$ のデータを入力すると $$T - 1$$ のデータが出力される.
            
        backward_hermite_coefficients (Hermite cubic splines with backward differences) では, 
        3次スプライン補間の係数 `d, c, b, a = coeffs` が生成される.
        これは3次スプライン補間 `CubicInterpolation` でパスを形成する前に使用する必要がある.
        """
        if self.interpolation == 'linear':
            func_interpolation = diffrax.linear_interpolation
        elif self.interpolation == 'cubic':
            func_interpolation = diffrax.backward_hermite_coefficients
        coeffs = jax.vmap(func_interpolation)(ts, ys) # -> (25699*3, 256*99*3, 256*99*3, 256*99*3)
        
        # 0/1の2値ラベルの作成
        labels = jnp.zeros((dataset_size,))
        labels = labels.at[: dataset_size // 2].set(1.0)
    
        # データのチャネル数を取得
        _, _, in_size = ys.shape
        
        return ts, ys, coeffs, labels, in_size

class MNISTStrokeDataset(BaseDataset):
    """**Citation:**
    https://edwin-de-jong.github.io/blog/mnist-sequence-data/
    
    ```
        The MNIST handwritten digit data set is widely used as a benchmark dataset for regular supervised learning.
        While a 2-D image of a digit does not look complex to a human being, it is a highly inefficient way for a 
        computer to represent a handwritten digit; only a fraction of the pixels are used.
        Furthermore, there is a lot of regularity in digits that is not exploited by a 2-D image encoding;
        all digits consist of continuous line segments, produced by pen strokes.
    ```
    """
    dataset_size: int
    mode_train: bool
    input_format: str
    noise_ratio: float
    interpolation: str
    key: PRNGKeyArray
    
    def __init__(self, *, dataset_size: int = -1, mode_train: bool = True, input_format: str = 'point_sequence', noise_ratio: float = 0.0, interpolation: str = 'cubic', key: PRNGKeyArray):
        """**Arguments:**
        - dataset_size: サンプル数. `-1` の場合は全サンプルが使用される.
        - mode_train: 読み込むデータが学習用か評価用かを指定するフラグ.
        - input_format: 入力データの形式. [mnist-sequence-data](https://edwin-de-jong.github.io/blog/mnist-sequence-data/) には複数の形式の系列データがあり, そのどれを使用するか指定する. `'point_sequence'` の場合はピクセル座標の系列データ, それ以外の場合(`'input_sequence'`)は画像の左上を(0, 0)とした二次元の座標変位, ペンが紙面から離れたタイミングか否か, および文字の書き終わりか否かの4チャネルの系列データが読み込まれる.
        - noise_ratio: データの一部をランダムに `nan` に置換する際の置換されるセルの割合. `0.` の場合は読み込んだデータそのままになる.
        - interpolation: 補間形式の指定. [Interpolations](https://docs.kidger.site/diffrax/api/interpolation/#interpolations)
        - key: a pseudo-random number generator (;PRNG) key. `key = jax.random.PRNGKey(seed)`
        """
        super().__init__()
        self.dataset_size = dataset_size
        self.mode_train = mode_train
        self.config = MNISTStrokeDatasetConfig()
        self.input_format = input_format
        self.noise_ratio = noise_ratio
        self.interpolation = interpolation
        self.key = key

    def make_dataset(self) -> Tuple[Sequence[Float[Array, "sequence"]], Sequence[Float[Array, "sequence channels+1"]], Tuple[Sequence[Float[Array, "sequence-1 channels+1"]]], Sequence[Float[Array, "sequence None"]], int]:
        """Creat or load records.
        手書き数字データセット MNIST の画像を解析して作成された, ピクセルの位置からなる書き順の系列データを読み込む関数.
        
        **Returns:**
        - ts: タイムスタンプ.
        - ys: レコード. 読み込まれた書き順の系列データと, それをもとづいて作成したタイムスタンプからなる系列データ.
        - coeffs: 補間で使用される係数. [Handling missing data](https://docs.kidger.site/diffrax/api/interpolation/#handling-missing-data)
        - labels: 正解ラベル. 0~9の整数値の正解ラベル.
        - in_size: 入力データのチャネル数. ただし, タイムスタンプを含めたチャネル数である. モデルの入力チャネル数を決定するために使用される.
        """
        # データセットの読み込み
        if self.mode_train:
            if self.input_format == 'point_sequence':
                path_list_input_sequence = glob.glob(self.config.train_data_point_sequence, recursive=True)
            elif self.input_format == 'input_sequence':
                path_list_input_sequence = glob.glob(self.config.train_data_input_sequence, recursive=True)
            path_label = self.config.train_label
        else:
            if self.input_format == 'point_sequence':
                path_list_input_sequence = glob.glob(self.config.test_data_point_sequence, recursive=True)
            elif self.input_format == 'input_sequence':
                path_list_input_sequence = glob.glob(self.config.test_data_input_sequence, recursive=True)
            path_label = self.config.test_label

        # sort on filename
        path_list_input_sequence = natsorted(path_list_input_sequence)
        #print(path_list_input_sequence[:50])
        #print(path_list_input_sequence[-50:])
    
        # データセットサイズを規定. 保有するデータ量よりもサイズが大きい or 負の値を指定したときは, データセットの全量を使用するとみなす.
        if (self.dataset_size < 0) or (self.dataset_size > len(path_list_input_sequence)):
            dataset_size = len(path_list_input_sequence)
        else:
            dataset_size = self.dataset_size
    
        # ts, ys の作成
        ys = []
        ts = []
        for filepath in tqdm(path_list_input_sequence[:dataset_size]):
            with open(filepath) as f:
                if self.input_format == 'point_sequence':
                    reader = csv.reader(f, delimiter=',')
                    next(reader)
                    lines = [[jnp.float32(x) for x in row] for row in reader]
                elif self.input_format == 'input_sequence':
                    lines = [[jnp.float32(x) for x in row] for row in csv.reader(f, delimiter=' ')]
                
                #ts.append(jnp.arange(len(lines)))
                _ys = jnp.asarray(lines)
                ts.append(self._make_timestamp((_ys[:, 0], _ys[:, 1])))
                _ys = self._random_mask_nan(_ys, self.key, p=self.noise_ratio)
                ys.append(jnp.concatenate([ts[-1][:, None], _ys], axis=-1))
    
        # 欠損値ありのデータ用の補間関数: Hermite cubic splines with backward differences
        if self.interpolation == 'linear':
            func_interpolation = diffrax.linear_interpolation
        elif self.interpolation == 'cubic':
            func_interpolation = diffrax.backward_hermite_coefficients
        
        coeffs = [[], [], [], []]
        dim_ys = ys[0].shape[-1]
        for _ts, _ys in zip(ts, ys):
            _coeffs = func_interpolation(_ts, _ys, fill_forward_nans_at_end=True, replace_nans_at_start=jnp.zeros(dim_ys))
            for i in jnp.arange(4):
                coeffs[i].append(_coeffs[i])
        coeffs = tuple(coeffs)
    
        # ラベルデータの読み込み
        with open(path_label) as f:
            labels = [[jnp.float32(x) for x in row] for row in csv.reader(f, delimiter=' ')]
            labels = labels[:dataset_size]
        labels = jnp.array(labels)
    
        # データのチャネル数を取得
        in_size = len(ys[0][0])
        
        return ts, ys, coeffs, labels, in_size

    def _make_timestamp(self, points: Tuple[Float[Array, "sequence"], Float[Array, "sequence"]]) -> Float[Array, "sequence"]:
        """Create timestamps based on the amount of shift in two-dimensional coordinates.
        位置の変化量に応じて時刻の間隔が大きくなるようにタイムスタンプを作成

        **Arguments:**
        - points: x座標の系列とy座標の系列のタプル
        
        **Returns:**
        - ts: タイムスタンプ
        """
        ts = [0.,]
        point_old = jnp.zeros(2)
        for point_curr in zip(*points):
            point_curr = jnp.array(point_curr)
            delta = jnp.linalg.norm(point_curr - point_old) * 0.001
            ts.append(ts[-1] + delta)
            point_old = point_curr
        return jnp.array(ts[1:])

    def _random_mask_nan(self, input_sequence: Float[Array, "sequence channels"], key: PRNGKeyArray, p: float = 0.0) -> Float[Array, "sequence channels"]:
        """Randomly convert part of the data to nan
        非一定サンプリングレート, 欠損値あり, 非同期のデータにするために, 後処理でデータの一部をランダムに `nan` に変換

        **Arguments:**
        - input_sequence: タイムスタンプを含まない時系列データ.
        
        **Returns:**
        - input_sequence: データの一部をランダムに `nan` に変換された `input_sequence`
        """
        m, n = input_sequence.shape
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1.")
        ratio = int(m * n * p)
        
        mask = jnp.zeros(m * n, dtype=bool)
        mask = mask.at[:ratio].set(True)
        mask = jr.permutation(key, mask)
        mask = mask.reshape(m, n)
        
        input_sequence = input_sequence.at[mask].set(jnp.nan)
        return input_sequence