import os
import numpy as np
import matplotlib.pyplot as plt


def bent_identity(x: np.ndarray) -> np.ndarray:
    return x + (np.sqrt(x * x + 1.0) - 1.0) / 2.0


def bent_identity_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 + x / (2.0 * np.sqrt(x * x + 1.0))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def main() -> None:
    x = np.linspace(-6.0, 6.0, 1200)

    y_bent = bent_identity(x)
    y_relu = relu(x)
    dy_bent = bent_identity_grad(x)
    dy_relu = relu_grad(x)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)

    axes[0].plot(x, y_bent, label="Bent Identity", linewidth=2.2)
    axes[0].plot(x, y_relu, label="ReLU", linewidth=2.2, linestyle="--")
    axes[0].set_title("Activation Function")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, dy_bent, label="Bent Identity grad", linewidth=2.2)
    axes[1].plot(x, dy_relu, label="ReLU grad", linewidth=2.2, linestyle="--")
    axes[1].set_title("Derivative")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f'(x)")
    axes[1].set_ylim(-0.05, 1.55)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Bent Identity vs ReLU")
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bent_identity_vs_relu.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
