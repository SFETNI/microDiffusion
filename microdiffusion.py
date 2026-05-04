"""The most atomic way to train and sample a diffusion model in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency. @yourname"""

import math
import random
import sys

# -----------------------------------------------------------------------------
# scalar autograd: tiny engines make bright little fires

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = math.exp(self.data)
        out = Value(x, (self,), "exp")

        def _backward():
            self.grad += x * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


# -----------------------------------------------------------------------------
# data: sixteen little cave paintings, each flattened to 64 floats


def grid(s):
    rows = [r.strip() for r in s.strip().split("/")]
    return [1.0 if c != "." else 0.0 for r in rows for c in r]


DATA = [
    ("plus", grid("...##.../...##.../...##.../########/########/...##.../...##.../...##...")),
    ("ring", grid("..####../.#....#./#......#/#......#/#......#/#......#/.#....#./..####..")),
    ("box", grid("########/#......#/#......#/#......#/#......#/#......#/#......#/########")),
    ("diag", grid("#......./.#....../..#...../...#..../....#.../.....#../......#./.......#")),
    ("up", grid("...##.../..####../.######./...##.../...##.../...##.../...##.../...##...")),
    ("right", grid("...#..../....#.../.....#../########/########/.....#../....#.../...#....")),
    ("smile", grid(".######./#......#/#.##.##./#......#/#.#..#.#/#..##..#/#......#/.######.")),
    ("heart", grid(".##..##./########/########/.######./..####../...##.../......../........")),
    ("a", grid("..####../.##..##./##....##/########/##....##/##....##/##....##/........")),
    ("z", grid("########/......##/.....##./....##../...##.../..##..../.##...../########")),
    ("check", grid("#.#.#.#./.#.#.#.#/#.#.#.#./.#.#.#.#/#.#.#.#./.#.#.#.#/#.#.#.#./.#.#.#.#")),
    ("diamond", grid("...##.../..####../.######./########/########/.######./..####../...##...")),
    ("x", grid("##....##/.##..##./..####../...##.../...##.../..####../.##..##./##....##")),
    ("bars", grid("########/......../########/......../########/......../########/........")),
    ("three", grid(".######./##....##/......##/..####../......##/......##/##....##/.######.")),
    ("moon", grid("..#####./.##...../##....../##....../##....../##....../.##...../..#####.")),
]


def show(x, title=None):
    chars = " .:-=+*#%@"
    if title:
        print(title)
    for y in range(8):
        row = ""
        for v in x[y * 8 : (y + 1) * 8]:
            i = max(0, min(len(chars) - 1, int(v * (len(chars) - 1) + 0.5)))
            row += chars[i] * 2
        print(row)
    print()


# -----------------------------------------------------------------------------
# network: noisy pixels enter, guessed noise leaves


def timestep_embedding(t, dim=16):
    emb = []
    for d in range(dim):
        scale = 10000 ** ((d - d % 2) / dim)
        emb.append(math.sin(t / scale) if d % 2 == 0 else math.cos(t / scale))
    return emb


class MLP:
    def __init__(self, sizes):
        self.layers = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            s = math.sqrt(2 / fan_in)
            w = [[Value(random.uniform(-s, s)) for _ in range(fan_in)] for _ in range(fan_out)]
            b = [Value(0.0) for _ in range(fan_out)]
            self.layers.append((w, b))

    def __call__(self, xs):
        xs = [x if isinstance(x, Value) else Value(x) for x in xs]
        for li, (w, b) in enumerate(self.layers):
            out = []
            for row, bias in zip(w, b):
                z = bias
                for wi, xi in zip(row, xs):
                    z = z + wi * xi
                out.append(z.relu() if li < len(self.layers) - 1 else z)
            xs = out
        return xs

    def parameters(self):
        return [p for layer in self.layers for part in layer for row in part for p in (row if isinstance(row, list) else [row])]


def adam(params, step, lr=0.003):
    b1, b2, eps = 0.9, 0.999, 1e-8
    if not hasattr(adam, "m") or len(adam.m) != len(params):
        adam.m = [0.0 for _ in params]
        adam.v = [0.0 for _ in params]
    lr *= min(1.0, step / 100) * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * step / adam.total)))
    for i, p in enumerate(params):
        g = max(-5.0, min(5.0, p.grad))
        adam.m[i] = b1 * adam.m[i] + (1 - b1) * g
        adam.v[i] = b2 * adam.v[i] + (1 - b2) * g * g
        mh = adam.m[i] / (1 - b1**step)
        vh = adam.v[i] / (1 - b2**step)
        p.data -= lr * mh / (math.sqrt(vh) + eps)
        p.grad = 0.0


# -----------------------------------------------------------------------------
# diffusion: let there be noise; then ask the net to remember what was added


T = 20
betas = [0.0] + [0.01 + (0.28 - 0.01) * i / (T - 1) for i in range(T)]
alphas = [1.0] + [1 - b for b in betas[1:]]
alpha_bars = [1.0]
for a in alphas[1:]:
    alpha_bars.append(alpha_bars[-1] * a)


def q_sample(x0, t, noise):
    a = alpha_bars[t]
    return [math.sqrt(a) * x + math.sqrt(1 - a) * e for x, e in zip(x0, noise)]


def pixels(x): return [max(0, min(1, 0.5 * v + 0.5)) for v in x]


def train(net, steps, verbose=True, lr=0.003, prediction="x0"):
    params = net.parameters()
    adam.total = steps
    losses = []
    for step in range(1, steps + 1):
        _, img = random.choice(DATA)
        x0 = [2 * v - 1 for v in img]
        t = 1 + int((T - 1) * (random.random() ** 0.65)) if prediction == "eps" else random.randint(1, T)
        eps = [random.gauss(0, 1) for _ in range(64)]
        xt = q_sample(x0, t, eps)
        pred = net(xt + timestep_embedding(t))
        target = eps if prediction == "eps" else x0
        loss = sum((p - y) ** 2 for p, y in zip(pred, target)) / 64
        loss.backward()
        adam(params, step, lr)
        losses.append(loss.data)
        if verbose and (step == 1 or step % max(1, steps // 20) == 0):
            tail = sum(losses[-50:]) / len(losses[-50:])
            bar = "#" * min(40, int(40 / (1 + tail)))
            print(f"step {step:5d}/{steps}  loss {tail:7.4f}  {bar}")
    return losses


def sample(net, frames=False, return_history=False, prediction="x0"):
    x = [random.gauss(0, 1) for _ in range(64)]
    history = [(T, x[:])]
    for t in range(T, 0, -1):
        pred = [v.data for v in net(x + timestep_embedding(t))]
        z = [random.gauss(0, 1) if t > 1 else 0.0 for _ in range(64)]
        b, a, ab = betas[t], alphas[t], alpha_bars[t]
        if prediction == "x0":
            pred = [max(-1, min(1, v)) for v in pred]
            eps = [(xi - math.sqrt(ab) * x0i) / math.sqrt(1 - ab) for xi, x0i in zip(x, pred)]
        else:
            eps = pred
        x = [(xi - b / math.sqrt(1 - ab) * ei) / math.sqrt(a) + math.sqrt(b) * zi for xi, ei, zi in zip(x, eps, z)]
        if frames and t in (20, 15, 10, 5, 1):
            history.append((t - 1, x[:]))
    if frames:
        print("one dream waking up:")
        for t, img in history:
            show(pixels(img), f"t={t}")
    history = [(t, pixels(img)) for t, img in history]
    return (pixels(x), history) if return_history else pixels(x)


def main():
    random.seed(1337)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    dreams = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    prediction = sys.argv[3] if len(sys.argv) > 3 else "x0"

    print("\nmicroDiffusion: learning 8x8 worlds from scratch\n")
    for name, img in DATA[:6]:
        show(img, f"training glyph: {name}")

    net = MLP([80, 32, 32, 64])
    print(f"parameters: {len(net.parameters())} | timesteps: {T} | target: {prediction} | training steps: {steps}\n")
    train(net, steps, prediction=prediction)

    print("\nhere is what the model dreams up:\n")
    sample(net, frames=True, prediction=prediction)
    for i in range(dreams):
        show(sample(net, prediction=prediction), f"dream {i + 1}")
    print("May your loss be low, and your noise remember structure.")


if __name__ == "__main__":
    main()
