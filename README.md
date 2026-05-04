# microDiffusion

`microdiffusion.py` is a tiny diffusion model written as one plain Python file.

It trains from scratch on a small embedded set of 8x8 pixel-art glyphs, then starts from pure Gaussian noise and denoises its way back into new little images. There are no data files, no downloads, no NumPy, no PyTorch, no hidden machinery. Just `math`, `random`, scalar autograd, a small MLP, and the DDPM loop.

The point is not speed or state-of-the-art image quality. The point is to make the whole algorithm visible.

## What Is Inside

- A hardcoded 8x8 pixel-art dataset
- A scalar `Value` autograd engine
- A tiny MLP denoiser
- A 20-step diffusion noise schedule
- Training and sampling in the same file
- ASCII output in the terminal
- Optional notebook views for the dataset, noise schedule, loss curve, denoising steps, and generated samples

## Run It

```bash
python microdiffusion.py
```

Shorter run:

```bash
python microdiffusion.py 800 4
```

The first argument is training steps. The second argument is the number of generated samples.

By default the model trains with `x0` prediction, which is easier for this tiny network. The original epsilon-prediction mode is still available:

```bash
python microdiffusion.py 800 4 eps
```

## Explore It

Open:

```text
microdiffusion_explorer.ipynb

## Notes on Notebooks and Repo Size

The repository may include executed Jupyter notebooks. Notebook outputs (embedded images/base64) can significantly increase repository size and make cloning slower for contributors.

Recommendations:

- Do not commit large notebook outputs. Use `nbstripout` or the provided `pre-commit` hook to automatically strip outputs before committing.
- If you're preparing a release archive and want to exclude notebooks, `.gitattributes` contains an `export-ignore` entry for `*.ipynb`.

See `CONTRIBUTING.md` for setup instructions.
```

The notebook shows the embedded glyph dataset, the forward noising process, the training curve, reverse denoising frames, and a gallery of generated samples.

## Philosophy

The same spirit as Karpathy's `microgpt.py`: the most atomic, dependency-free, single-file implementation of a diffusion model. Training and inference. This file is the complete algorithm. Everything else is just efficiency.

Reference: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
