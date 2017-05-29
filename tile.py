#!/bin/env python3

"""Image tiler for the SpatialEye ST App"""

import argparse as ap

import logging

from math import ceil, log2

import os

import numpy as np

from scipy.misc import imread, imsave


DEF_TILE_SIZE = [256, 256]


logging.basicConfig(format='[%(asctime)s]  %(message)s')
LOG = logging.getLogger()


def zoom(img):
    """
    Zoom image by a factor of 2.

    Parameters
    ----------
    img : numpy.ndarray
        The input image.

    Returns
    -------
    zoomed : numpy.ndarray
        The zoomed image.
    """
    ret = np.zeros(
        [ceil(d / 2) for d in img.shape[:2]] + list(img.shape[2:]),
        dtype=np.uint8,
    )
    for (row, col) in (
            (r, c)
            for r in range(ceil(img.shape[0] / 2))
            for c in range(ceil(img.shape[1] / 2))
    ):
        ret[row, col] = np.round(np.sum(
            img[
                slice(2 * row, 2 * (row + 1)),
                slice(2 * col, 2 * (col + 1))
            ],
            axis=(0, 1),
        ) / 4)
    return ret

def gettiles(tile_shape, img):
    """
    Tile image.

    Parameters
    ----------
    tile_shape : array-like
        The shape of the tiles.
    img : numpy.ndarray
        The input image.

    Yields
    ------
    tile : numpy.ndarray
        The output tile.
    row : tuple
        A 2-tuple `(x, y)` where `x` is the current row and `y` is the total
        number of rows in the tiled image.
    column : tuple
        A 2-tuple `(x, y)` where `x` is the current column and `y` is the total
        number of columns in the tiled image.
    """
    rows, cols = [ceil(s / t) for (s, t) in zip(img.shape, tile_shape)]
    for (row, col) in ((r, c) for r in range(rows) for c in range(cols)):
        yield tuple((
            img[
                slice(tile_shape[0] * row, tile_shape[0] * (row + 1)),
                slice(tile_shape[1] * col, tile_shape[1] * (col + 1))
            ],
            tuple((row, rows)),
            tuple((col, cols)),
        ))

def tile(tile_shape, img, nlevels=None):
    """
    Tile image over different zoom levels.

    Parameters
    ----------
    tile_shape : array-like
        The shape of the tiles.
    img : numpy.ndarray
        The input image.
    nlevels : int
        The maximum zoom level.

    Yields
    ------
    tiles : list
        List of the tiles at a certain `level`
    level : int
        The zoom level of the `tiles`
    """
    if nlevels is None:
        nlevels = 1 + max([
            ceil(log2(s / t))
            for (s, t) in zip(img.shape, tile_shape)
        ])
    for level in reversed(range(nlevels)):
        for curtile, row, col in gettiles(tile_shape, img):
            yield curtile, row, col, tuple((level, nlevels))
        if level > 0:
            img = zoom(img)

def save(output_dir, row, col, lvl, image_tile):
    """
    Save tile according to the format required by the SpatialEye ST app.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    row : tuple
        A 2-tuple `(x, y)` where `x` is the current row and `y` is the total
        number of rows in the tiled image.
    col : tuple
        A 2-tuple `(x, y)` where `x` is the current column and `y` is the total
        number of columns in the tiled image.
    lvl : tuple
        A 2-tuple `(x, y)` where `x` is the current level and `y` is the total
        number of levels in the tiled image.
    image_tile : numpy.ndarray
        The tile to save.
    """
    currow, maxrows = row
    curcol, _ = col
    curlvl, _ = lvl
    dirname = os.path.join(
        output_dir,
        str(curlvl),
        str(curcol),
    )
    basename = '{:d}.png'.format(maxrows - currow - 1)
    filename = os.path.join(dirname, basename)
    LOG.info('Writing tile lxrxc={:d}x{:d}x{:d} to {:s}.'.format(
        curlvl, currow, curcol, filename,
    ))
    os.makedirs(dirname, exist_ok=True)
    imsave(filename, image_tile)

def climain():
    """Entry point for cli"""
    opt = ap.ArgumentParser()
    opt.add_argument('-i', '--input', required=True, help='path to input image')
    opt.add_argument('-o', '--output', required=True, help='output directory')
    opt.add_argument('--shape', type=list, default=DEF_TILE_SIZE,
                     help='tile dimensions. '
                          'default = {:d} {:d}.'.format(*DEF_TILE_SIZE))
    opt.add_argument('--levels', type=int, default=None,
                     help='max zoom level. if set to n, 2^n px in the input '
                          'image will correspond to 1 px in the tiled image on '
                          'the lowest (0:th) zoom level. by default, this '
                          'will be set to the smallest value needed to fit the '
                          'entire image on a single tile at the lowest zoom '
                          'level.')
    opt.add_argument('--silent', action='store_true',
                     help='suppress output to stdout. note: output to stderr '
                          'is not suppressed.')
    opt = opt.parse_args()

    if opt.silent:
        LOG.setLevel(logging.WARN)
    else:
        LOG.setLevel(logging.INFO)

    for curtile, row, col, lvl in tile(
            opt.shape,
            imread(opt.input),
            opt.levels,
    ):
        save(opt.output, row, col, lvl, curtile)

if __name__ == "__main__":
    climain()
