# (Modified) gen_images.py - StyleGAN3 sampling with orig-folder based imbalanced sampling
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# (Original header kept)
import os
import re
import math
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

# ---------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        p = p.strip()
        if not p:
            continue
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

# ---------------------------------------------------------------------------

def count_images_in_class_folder(path: str) -> int:
    """Count image-like files in a folder (non-recursive)."""
    if not os.path.isdir(path):
        return 0
    files = os.listdir(path)
    # count files that look like images (common extensions), skip hidden files
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    c = 0
    for fn in files:
        if fn.startswith('.'):
            continue
        _, e = os.path.splitext(fn)
        if e.lower() in exts or e == '':
            # treat extensionless too, in case
            full = os.path.join(path, fn)
            if os.path.isfile(full):
                c += 1
    return c

def compute_generation_counts_from_folder(orig_folder: str,
                                          num_classes: int,
                                          multiple: float,
                                          balance_alpha: float,
                                          include_empty: bool) -> List[int]:
    """
    Compute per-class generation counts by interpolating between
    original imbalance and full class-balance.

    balance_alpha ∈ [0,1]:
      1.0 → match original imbalance
      0.0 → fully class-balanced
    """
    orig_counts = []
    for cls in range(num_classes):
        cls_path = os.path.join(orig_folder, str(cls))
        orig_counts.append(count_images_in_class_folder(cls_path))
    orig_counts = np.array(orig_counts, dtype=np.float64)

    print(f'Original counts per class (from {orig_folder}): {orig_counts.tolist()}')

    nonzero = orig_counts[orig_counts > 0]
    if nonzero.size == 0:
        print('Warning: no files found under orig_folder.')
        return [0] * num_classes

    min_c = nonzero.min()

    # Interpolate between imbalanced and balanced
    alpha = float(balance_alpha)
    interpolated = alpha * orig_counts + (1.0 - alpha) * min_c

    # Handle empty classes
    if not include_empty:
        interpolated = np.where(orig_counts == 0, 0.0, interpolated)
    else:
        interpolated = np.where(orig_counts == 0, min_c, interpolated)

    # Apply multiple
    final = np.round(interpolated * float(multiple)).astype(np.int64)

    print(f'Computed generation counts per class (multiple={multiple}, balance_alpha={balance_alpha}): {final.tolist()}')
    return final.tolist()

# ---------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help="List of random seeds (e.g., '0,1,4-6'). Optional when using --classes.", required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Single class label (kept for compatibility)')
@click.option('--classes', type=parse_range, help="Iterable of class indices (e.g. '0,2,5-7'). When provided with a conditional model this is required.", required=False)
@click.option('--n-per-class', 'n_per_class', type=int, help='Number of images to generate per class when using --classes (used only if --orig-folder not provided)', default=1, show_default=True)
@click.option('--base-seed', 'base_seed', type=int, help='Base seed used to derive per-image seeds (deterministic)', default=0, show_default=True)
@click.option('--trunc-seed-offset', 'seed_offset', type=int, help='Optional offset added to derived seeds (useful to avoid overlap with other runs)', default=0, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# new options for orig-folder based imbalanced sampling
@click.option('--orig-folder', help='(Optional) ImageFolder-style folder with original images to derive class counts from (subfolders named 0..K-1)', type=str, required=False)
@click.option('--multiple', help='Multiply original counts by this factor to produce generation counts (float)', type=float, default=1.0, show_default=True)
@click.option('--balance-alpha', help='Interpolation factor between original imbalance (1.0) and full balance (0.0)',
              type=float, default=1.0, show_default=True)
@click.option('--include-empty-classes/--no-include-empty-classes', help='Whether to generate images for classes that have zero originals (default False)', default=False)
def generate_images(
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    class_idx: Optional[int],
    classes: Optional[List[int]],
    n_per_class: int,
    base_seed: int,
    seed_offset: int,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    orig_folder: Optional[str],
    multiple: float,
    balance_alpha: float,
    include_empty_classes: bool
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Decide mode: conditional / unconditional and input sources
    is_conditional = (G.c_dim != 0)

    # Validate and prepare class list(s)
    if is_conditional:
        if classes is not None:
            target_classes = classes
        elif class_idx is not None:
            target_classes = [class_idx]
        else:
            raise click.ClickException('Conditional network detected: please provide --classes or --class.')

        for c in target_classes:
            if c < 0 or c >= G.c_dim:
                raise click.ClickException(f'class index {c} is out of range for this network (valid: 0..{G.c_dim-1}).')
    else:
        if classes is not None or class_idx is not None:
            print('warn: --classes/--class ignored when running on an unconditional network')
        target_classes = [None]

    # Helper: deterministic seed per generated image
    def derived_seed(cls_val: Optional[int], idx_within_cls: int) -> int:
        cls_part = 0 if cls_val is None else int(cls_val) + 1
        raw = int(base_seed) ^ (cls_part * 0x9e3779b9) ^ (idx_within_cls + seed_offset)
        return raw % (2**32)

    # If orig_folder provided, compute per-class generation counts
    gen_counts_by_class = None
    if orig_folder is not None:
        if not is_conditional:
            print('Warning: --orig-folder provided but model is unconditional; --orig-folder will be ignored.')
        else:
            print(f'Inspecting orig-folder: {orig_folder}')
            gen_counts_by_class = compute_generation_counts_from_folder(
                                            orig_folder, G.c_dim, multiple, balance_alpha, include_empty_classes
                                        )

    # Generate images (conditional path)
    if classes is not None and is_conditional:
        # When gen_counts_by_class is available, derive per-class n; else use n_per_class constant.
        for cls in target_classes:
            class_outdir = os.path.join(outdir, str(cls))
            os.makedirs(class_outdir, exist_ok=True)

            # number to generate for this class
            if gen_counts_by_class is not None:
                n_generate = int(gen_counts_by_class[cls])
            else:
                n_generate = int(n_per_class)

            if n_generate <= 0:
                print(f'Skipping generation for class {cls} (n_generate={n_generate})')
                continue

            # Prepare label tensor for this class
            label = torch.zeros([1, G.c_dim], device=device)
            label[:, cls] = 1

            for i in range(n_generate):
                seed = derived_seed(cls, i)
                print(f'Generating class {cls} image {i+1}/{n_generate} using seed {seed} -> {class_outdir}')
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

                # Transform
                if hasattr(G.synthesis, 'input'):
                    m = make_transform(translate, rotate)
                    m = np.linalg.inv(m)
                    G.synthesis.input.transform.copy_(torch.from_numpy(m))

                img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                filename = f'img{i:05d}_seed{seed}.png'
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(class_outdir, filename))

    else:
        # Unconditional / seeds usage
        if seeds is None:
            raise click.ClickException('No seeds provided. For unconditional networks provide --seeds. For conditional networks provide --classes (and optionally --n-per-class or --orig-folder).')

        label = torch.zeros([1, G.c_dim], device=device) if is_conditional else None
        if is_conditional and class_idx is not None:
            label[:, class_idx] = 1
            class_outdir = os.path.join(outdir, str(class_idx))
            os.makedirs(class_outdir, exist_ok=True)
        else:
            class_outdir = outdir

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            # Transform
            if hasattr(G.synthesis, 'input'):
                m = make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            filename = f'seed{seed:05d}.png'
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(class_outdir, filename))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
