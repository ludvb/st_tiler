#!/bin/env python3

"""Image tiler for the SpatialEye ST App"""

import argparse as ap

from enum import Enum

from functools import partial

import logging

from math import ceil, log2

import os

import queue

import threading

import numpy as np

from scipy.misc import imread, imsave, imresize


class Corner(Enum):
    """
    Enum of the ''corner'' of the tilemap (i.e., where the (0, 0) coordinate
    is).
    """
    SouthWest = 0  # TMS default
    NorthWest = 1
    SouthEast = 2
    NorthEast = 3

class Order(Enum):
    """
    Enum of the coordinate order of the tilemap. If stored in `Order.ColMajor`
    order, coordinates are given as `(c, r)`, where `c` is the column and `r` is
    the row of the tilemap. Conversely, if set to `Order.RowMajor`, coordinates
    will be given as `(r, c)`.
    """
    ColMajor = 0
    RowMajor = 1


DEF_TILE_SIZE = [256, 256]
DEF_ORDER = Order.ColMajor
DEF_CORNER = Corner.SouthWest

DEF_THREADS = 8


logging.basicConfig(format='[%(asctime)s]  %(message)s')
LOG = logging.getLogger()


class Threader(object):
    """
    Helper class for multithreading tasks.

    Parameters
    ----------
    nthreads : int
        The number of threads to use.
    """
    def __init__(self, nthreads):
        self.queue = queue.Queue()
        self.nthreads = nthreads
        self.threads = []

    def add(self, item):
        """
        Add task to queue.

        Parameters
        ----------
        item : callable
            The job to be executed.

        Returns
        -------
        self : Threader
            The current instance.
        """
        self.queue.put(item)
        return self

    def await(self):
        """
        Wait for all queued tasks to complete.

        Returns
        -------
        self : Threader
            The current instance.
        """
        self.queue.join()
        return self

    def start(self):
        """
        Start task consumption.

        Returns
        -------
        self : Threader
            The current instance.
        """
        self.threads.clear()
        for _ in range(self.nthreads):
            thread = threading.Thread(target=self._worker)
            self.threads.append(thread)
            thread.start()
        return self

    def stop(self):
        """
        Wait for all queued tasks to complete and stop all threads.

        Returns
        -------
        self : Threader
            The current instance.
        """
        self.await()
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()
        return self

    def _worker(self):
        """Worker event loop"""
        while True:
            job = self.queue.get()
            if job is None:
                break
            job()
            self.queue.task_done()

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
    return imresize(img, 50)

def pad(corner, shape, img, fill=None):
    """
    Pad image to shape by expanding the image array from a given corner.

    Parameters
    ----------
    corner : Corner
        The corner to expand from.
    shape : array-like
        Target shape. Must be a 2-tuple in which each coordinate is greater than
        or equal to `img[:2]`.
    img : numpy.ndarray
        The input image.
    fill : numpy.ndarray, optional
        The value to fill the padded pixels with.

    Returns
    -------
    padded_img : numpy.ndarray
        The padded image.
    """
    new_shape = list(shape) + list(img.shape[2:])
    top, left = [r - s for (r, s) in zip(new_shape[:2], img.shape[:2])]
    if top < 0 or left < 0:
        raise ValueError('Target shape is smaller than the shape of the image.')
    bottom, right = 0, 0
    if corner == Corner.SouthWest or corner == Corner.NorthWest:
        left, right = right, left
    if corner == Corner.NorthWest or corner == Corner.NorthEast:
        top, bottom = bottom, top
    bottom, right = [s - x for (s, x) in zip(new_shape[:2], (bottom, right))]
    padded_img = np.zeros(new_shape)
    padded_img[top:bottom, left:right] = img
    if fill is not None:
        if fill.shape != img.shape[2:]:
            raise ValueError('Fill value has invalid shape.')
        padded_img[:top, :] = \
            padded_img[bottom:, :] = \
            padded_img[:, :left] = \
            padded_img[:, right:] = \
            fill
    return padded_img

def tilecoordinates(order, corner, shape, coordinates):
    """
    Translate tile coordinates given in ''image coordinates'' to another basis.
    If `order == Order.RowMajor` and `corner == Corner.NorthWest`, this is the
    identity function.

    Parameters
    ----------
    order : Order
        See `Order`.
    corner : Corner
        See `Corner`.
    shape : array_like
        The shape of the tilemap.
    coordinates : array_like
        The coordinates of the tile in ''image coordinates'', i.e., `(row,
        column)`, where `(0, 0)` represents the north west corner of the image.

    Returns
    -------
    new_coordinates : tuple
        A 2-tuple `(r, c)` where `r` is the current row and `c` is the current
        column of the tile.
    """
    rows, cols = shape
    row, col = coordinates
    if corner == Corner.SouthEast or corner == Corner.NorthEast:
        col = cols - col - 1
    if corner == Corner.SouthWest or corner == Corner.SouthEast:
        row = rows - row - 1
    return (col, row) if order == Order.ColMajor else (row, col)

def gettiles(tile_shape, img, corner=Corner.SouthWest, order=Order.ColMajor):
    """
    Tile image.

    Parameters
    ----------
    tile_shape : array-like
        The shape of the tiles.
    img : numpy.ndarray
        The input image.
    corner : Corner, optional
        See `Corner`. Defaults to `Corner.SouthWest`.
    order : Order, optional
        See `Order`. Defaults to `Order.ColMajor`.

    Yields
    ------
    tile : numpy.ndarray
        The output tile.
    coordinate : tuple
        A 2-tuple `(r, c)` where `r` is the current row and `c` is the current
        column.
    """
    rows, cols = [ceil(s / t) for (s, t) in zip(img.shape, tile_shape)]
    tilecoordinates_ = partial(tilecoordinates, order, corner, (rows, cols))
    img = pad(
        corner,
        [s * n for (s, n) in zip(tile_shape, (rows, cols))],
        img,
    )
    for (row, col) in ((r, c) for r in range(rows) for c in range(cols)):
        yield (
            img[
                slice(tile_shape[0] * row, tile_shape[0] * (row + 1)),
                slice(tile_shape[1] * col, tile_shape[1] * (col + 1))
            ],
            tilecoordinates_((row, col)),
        )

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
        for curtile, (row, col) in gettiles(tile_shape, img):
            yield curtile, row, col, level
        if level > 0:
            img = zoom(img)

def save(output_dir, major, minor, level, image_tile):
    """
    Save tile according to the format required by the SpatialEye ST app.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    major : int
        The major order coordinate of the tile. For example, if the tiles are
        given in `Order.RowMajor` order, this is the row of the tile.
    minor : int
        The minor order coordinate of the tile.
    level : int
        The zoom level of the tile.
    image_tile : numpy.ndarray
        The tile to save.
    """
    dirname = os.path.join(
        output_dir,
        str(level),
        str(major),
    )
    basename = '{:d}.png'.format(minor)
    filename = os.path.join(dirname, basename)
    LOG.info('Writing tile lxrxc={:d}x{:d}x{:d} to {:s}.'.format(
        level,
        major,
        minor,
        filename,
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
    opt.add_argument('--threads', type=int, default=DEF_THREADS,
                     help='number of threads to use for file I/O. '
                          'default = {:d}'.format(DEF_THREADS))
    opt = opt.parse_args()

    if opt.silent:
        LOG.setLevel(logging.WARN)
    else:
        LOG.setLevel(logging.INFO)

    threader = Threader(opt.threads).start()
    for curtile, row, col, lvl in tile(
            opt.shape,
            imread(opt.input, mode='RGBA'),
            opt.levels
    ):
        threader.add(partial(save, opt.output, row, col, lvl, curtile))
    threader.stop()

if __name__ == "__main__":
    climain()
