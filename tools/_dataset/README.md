# データセットの準備

!!! danger "Time is a channel"

    データのチャンネルとして時間が含まれていることに注意！
    細かいポイントなのでよくうっかり見逃されがちなところ。時間を含めることで Neural CDE モデルは十分な情報を持つことになり、理論的に普遍近似器となる。逆に、これを忘れてしまうとモデルはおそらくあまりうまく機能しなくなる...。もし Neural CDE モデルが上手く学習できない時は、まず"チャネルとして時間を含めたか"確認するといい。

## タイムスタンプの間隔がバラバラなデータを作成
欠損値がないデータを作成した後に、ランダムに `nan` に置き換えることで作成している。

```
def _random_mask_nan(self, input_sequence: Array, key: PRNGKeyArray, p: float = 0.0) -> Array:
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
```

## 実用時にチャネル間でサンプリングレートの同期が取れていない場合の対応
単位・表記が統一されたタイムスランプのカラム `ts_uniformed` をチャネル毎に作成し、 `pandas.DataFrame` に対して `pandas.merge` (`on='ts_uniformed'`, `how='outer'`) を使って結合することで、 Neural CDE に入力できる形式のデータとなる。