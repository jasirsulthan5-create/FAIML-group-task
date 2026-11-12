"""Module 4 - Backpropagation walkthrough on the 2-2-1 sigmoid network.
Reproduces Tables 4.1-4.3 and Figure 4.2 of the report."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def init_params():
    return dict(w1=0.4, w2=0.3, w3=0.2, w4=0.6, w5=0.5, w6=0.7, b1=0.1, b2=0.2)


X1, X2, T = 0.5, 0.8, 1.0


def forward(p):
    z1 = p["w1"] * X1 + p["w2"] * X2 + p["b1"]
    z2 = p["w3"] * X1 + p["w4"] * X2 + p["b1"]
    h1, h2 = sigmoid(z1), sigmoid(z2)
    z3 = p["w5"] * h1 + p["w6"] * h2 + p["b2"]
    o = sigmoid(z3)
    return h1, h2, o, 0.5 * (T - o) ** 2


def backward(p):
    h1, h2, o, _ = forward(p)
    delta_o = (o - T) * o * (1 - o)
    delta_h1 = delta_o * p["w5"] * h1 * (1 - h1)
    delta_h2 = delta_o * p["w6"] * h2 * (1 - h2)
    return dict(w1=delta_h1 * X1, w2=delta_h1 * X2,
                w3=delta_h2 * X1, w4=delta_h2 * X2,
                w5=delta_o * h1, w6=delta_o * h2,
                b1=delta_h1 + delta_h2, b2=delta_o)


def step(p, lr):
    g = backward(p)
    return {k: p[k] - lr * g[k] for k in p}, g


def train(lr, iters):
    p, losses = init_params(), []
    for _ in range(iters):
        losses.append(forward(p)[3])
        p, _ = step(p, lr)
    losses.append(forward(p)[3])
    return losses


if __name__ == "__main__":
    p = init_params()
    h1, h2, o, E = forward(p)
    print(f"Forward : h1={h1:.4f} h2={h2:.4f} o={o:.4f} E={E:.4f}")

    p2, g = step(p, 0.5)
    print("\nTable 4.1 - gradients and updated parameters")
    for k in p:
        print(f"{k}: old={p[k]:.1f}  grad={g[k]:+.5f}  new={p2[k]:.4f}")

    h1, h2, o, E = forward(p2)
    print(f"\nTable 4.2 - after update: h1={h1:.4f} h2={h2:.4f} "
          f"o={o:.4f} E={E:.4f}")

    print("\nTable 4.3 - effect of learning rate")
    for lr in (0.1, 0.5, 1.0, 2.0):
        L = train(lr, 300)
        print(f"lr={lr}: loss@20={L[20]:.4f}  loss@300={L[300]:.4f}")

    L = train(0.5, 300)
    plt.plot(L, color="darkred")
    plt.xlabel("Training iteration")
    plt.ylabel("Loss E")
    plt.title("Loss falls as backpropagation repeats (lr = 0.5)")
    plt.tight_layout()
    plt.savefig("module4_loss_curve.png", dpi=150)
    print("\nSaved module4_loss_curve.png")
