# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle.

Supports class-conditional sampling by providing --classes (e.g. "0,1,4-6")
and --n-per-class to generate multiple images per class. Saves images in
ImageFolder layout under outdir/<class_idx>/.
"""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
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

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.'''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help="List of random seeds (e.g., '0,1,4-6'). Optional when using --classes.", required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Single class label (kept for compatibility)')
@click.option('--classes', type=parse_range, help="Iterable of class indices (e.g. '0,2,5-7'). When provided with a conditional model this is required.", required=False)
@click.option('--n-per-class', 'n_per_class', type=int, help='Number of images to generate per class when using --classes', default=1, show_default=True)
@click.option('--seed', 'base_seed', type=int, help='Base seed used to derive per-image seeds (deterministic)', default=0, show_default=True)
@click.option('--trunc-seed-offset', 'seed_offset', type=int, help='Optional offset added to derived seeds (useful to avoid overlap with other runs)', default=0, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
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
    rotate: float
):
    """Generate images using pretrained network pickle.

    New usage for class-conditional sampling (recommended):
        python gen_images.py --network=... --outdir=out --classes=0,1,2 --n-per-class=5 --trunc=1 --seed=42

    Backwards-compatible usage (single class or unconditional):
        python gen_images.py --network=... --outdir=out --class=2 --seeds=0,1,2
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Decide mode: conditional / unconditional and input sources
    is_conditional = (G.c_dim != 0)

    # Validate and prepare class list(s)
    if is_conditional:
        # If classes provided, use them (preferred).
        if classes is not None:
            target_classes = classes
        elif class_idx is not None:
            target_classes = [class_idx]
        else:
            raise click.ClickException('Conditional network detected: please provide --classes or --class.')
        # Validate indices in range
        for c in target_classes:
            if c < 0 or c >= G.c_dim:
                raise click.ClickException(f'class index {c} is out of range for this network (valid: 0..{G.c_dim-1}).')
    else:
        # Unconditional network - ignore classes, require seeds.
        if classes is not None or class_idx is not None:
            print('warn: --classes/--class ignored when running on an unconditional network')
        target_classes = [None]

    # Helper: produce a deterministic seed for each generated image
    def derived_seed(cls_val: Optional[int], idx_within_cls: int) -> int:
        cls_part = 0 if cls_val is None else int(cls_val) + 1
        raw = int(base_seed) ^ (cls_part * 0x9e3779b9) ^ (idx_within_cls + seed_offset)
        return raw % (2**32)

    # Generate images
    if classes is not None and is_conditional:
        # Use classes + n_per_class
        for cls in target_classes:
            class_outdir = os.path.join(outdir, str(cls))
            os.makedirs(class_outdir, exist_ok=True)

            # Prepare label tensor for this class
            label = torch.zeros([1, G.c_dim], device=device)
            label[:, cls] = 1

            for i in range(n_per_class):
                seed = derived_seed(cls, i)
                print(f'Generating class {cls} image {i+1}/{n_per_class} using seed {seed} -> {class_outdir}')
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
        # Fall back to seeds usage (old behavior / unconditional networks / single-class via --class)
        if seeds is None:
            raise click.ClickException('No seeds provided. For unconditional networks provide --seeds. For conditional networks provide --classes (and optionally --n-per-class).')

        # single class label (or empty for unconditional)
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


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
