import matplotlib
import matplotlib.pyplot as plt
import jax.random as jr

from .._dataset import datasets

def test_module(seed: int = 5678, dataset_size:int = 256, length:int = 100, add_noise: bool = True) -> None:
    key = jr.PRNGKey(seed)
    ckey, nkey=  jr.split(key, 2)

    ts, ys, coeffs, labels, in_size = datasets.make_2dspiral_dataset(dataset_size=dataset_size, length=length, add_noise=add_noise, key=ckey)
    spiral_dataset = (ts, ys, labels)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(projection="3d")

    ax.plot(spiral_dataset[1][0, :, 1], spiral_dataset[1][0, :, 2], spiral_dataset[1][0, :, 0], c="dodgerblue", label="Spiral")
    ax.plot(spiral_dataset[1][-1, :, 1], spiral_dataset[1][-1, :, 2], spiral_dataset[1][-1, :, 0], c="crimson", label="Reverse")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    plt.tight_layout()
    plt.savefig("./figures/sample_of_2D_spiral.png")
    plt.show()

# %% 実行部
if __name__ == '__main__':
    test_module()
