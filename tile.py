#!/bin/env python3

"""Image tiler for the SpatialEye ST App"""

import argparse as ap

from functools import partial

import logging

from math import ceil, log2

import os

import queue

import threading

from scipy.misc import imread, imsave, imresize


DEF_TILE_SIZE = [256, 256]

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
            imread(opt.input),
            opt.levels
    ):
        threader.add(partial(save, opt.output, row, col, lvl, curtile))
    threader.stop()

if __name__ == "__main__":
    climain()
