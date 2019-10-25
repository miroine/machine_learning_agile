#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import random
from collections import namedtuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_file_info(fname):
    """
    Get various bits of info from the full path of a file.

    Example
    >>> get_file_info('../data/prod/train/trilobites/0263.jpg')
    File_info(base='0263.jpg', cohort='train', cname='trilobites', stem='0263', ext='.jpg')
    """
    _path, base = os.path.split(fname)    # base: the name of the file
    _path, cname = os.path.split(_path)   # cname: the class name
    _path, cohort = os.path.split(_path)  # cohort: train or val
    stem, ext = os.path.splitext(base)    # stem, ext: file stem and extension

    File_info = namedtuple('File_info', ['base', 'cohort', 'cname', 'stem', 'ext'])
    return File_info(base, cohort, cname, stem, ext)


def make_train_test(path, include=None, skip=None):
    """
    Take a POSIX path, with wildcards, and turn the image files into arrays.

    Example
    >>> path = '../data/prod/*/*/*.jpg'
    >>> X_train, X_val, y_train, y_val = make_train_test(path)
    >>> X_train.shape, y_train.shape
    ((528, 4096), (528,))
    """

    X_train, X_val, y_train, y_val = [], [], [], []

    for fname in glob.glob(path, recursive=True):
        base, cohort, cname, stem, ext = get_file_info(fname)

        if skip is None:
            skip = []

        if cname in skip:
            continue

        if (include is not None) and (cname not in include):
            continue

        im = Image.open(fname)
        img_i = np.asarray(im, dtype=np.float) / 255

        if cohort == 'train':
            X_train.append(img_i.ravel())
            y_train.append(cname)
        elif cohort == 'val':
            X_val.append(img_i.ravel())
            y_val.append(cname)

    return (np.array(X_train), np.array(X_val),
            np.array(y_train), np.array(y_val)
            )


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    From https://scikit-learn.org/
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def visualize(X_val, y_val, y_prob, cutoff=0.5, ncols=6, nrows=3, figsize=(12, 8), classes=None, shape=None):
    """
    Visualize some random samples from the prediction results.
    Colours: green for a good prediction, red for a wrong one. If the
    probability was less than some cutoff (default 0.5), we'll mute the colour.

    Args:
        X_val (ndarray): The validation features, n_samples x n_features.
        y_val (ndarray): The validation labels, n_samples x 1.
        y_prob (ndarray): The predicted probabilities, n_samples x n_classes.
        cutoff (float): the cutoff for 'uncertain'.
        ncols (int): how many plots across the grid.
        nrows (int): how many plots down the grid.
        figsize (tuple): tuple of ints.
        classes (array-like): the classes, in order. Will be inferred if None.
        shape (tuple): Shape of each instance, if it needs reshaping.
    """
    idx = random.sample(range(X_val.shape[0]), ncols*nrows)
    sample = X_val[idx]

    if classes is None:
        classes = np.unique(y_val)
    else:
        y_val = np.asarray(classes)[y_val]

    fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
    axs = axs.ravel()

    for ax, img, actual, probs in zip(axs, sample, y_val[idx], y_prob[idx]):

        pred = classes[np.argmax(probs)]
        prob = np.max(probs)
        if shape is not None:
            img = img.reshape(shape)

        ax.imshow(np.squeeze(img), cmap='gray')
        ax.set_title(f"{pred} - {prob:.3f}\n[{actual}]")
        ax.set_xticks([])
        ax.set_yticks([])

        if prob > cutoff:
            c = 'limegreen' if (actual == pred) else 'red'
        else:
            c = 'y' if (actual == pred) else 'lightsalmon'

        for spine in ax.spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(4)

    return


def preprocess_images(path,
                      target,
                      size=None,
                      prop=0.25,
                      grey=True,
                      verbose=True
                      ):
    """
    Prepare the training and validation data.

    Args:
        path (str): The POSIX path, globbable.
        target (str): The name of the directory in which to put everything.
        size (tuple): Tuple of ints, the size of the output images.
        prop (float): Proportion of images to send to val (rest go to train).
        grey (bool): Whether to send to greyscale.
        verbose (bool): Whether to report out what the function did.

    Returns:
        None
    """
    trn = 0
    count = 0

    if size is None:
        size = (32, 32)

    for i, fname in enumerate(glob.glob(path)):

        # Read various bits of the path.
        path, base = os.path.split(fname)
        _, folder = os.path.split(path)
        name, ext = os.path.splitext(base)

        # Try to open the file.
        try:
            with Image.open(fname) as im:

                count += 1
                # Send 25% to val folder.
                cohort = 'train' if i % round(1/prop, 0) else 'val'

                # Form an output filename.
                outfile = os.path.join(target, cohort, folder, f'{i:04d}.jpg')

                # Resize, remove alpha, and save.
                im = im.resize(size, Image.ANTIALIAS)

                if grey:
                    im = im.convert('L')

                im.save(outfile)

                # Flip and rotate.
                if cohort == 'train':
                    trn += 1
                    outfile = os.path.join(target,
                                           cohort,
                                           folder,
                                           f'{i:04d}r.jpg')
                    im.transpose(Image.ROTATE_90).save(outfile)
                    outfile = os.path.join(target,
                                           cohort,
                                           folder,
                                           f'{i:04d}f.jpg')
                    im.transpose(Image.FLIP_LEFT_RIGHT).save(outfile)

        except OSError as e:
            print(f'{folder}/{base} rejected:', e)
            continue

    if verbose:
        print(f"Wrote {trn} files to {os.path.join(target, 'train')}")
        print(f"Wrote {count-trn} files to {os.path.join(target, 'val')}")

    return None


def plot_result(*args,
                labels=None,
                colours=None,
                lws=None,
                styles=None,
                samples=100
                ):
    """
    Plot the results of a prediction on the Zoeppritz dataset.
    """
    labels = labels or [f'Data {i}' for i in range(len(args))]
    colours = colours or [None for _ in args]
    lws = lws or [None for _ in args]
    styles = styles or [None for _ in args]

    rmss = np.sqrt(np.mean(np.square(np.subtract(args[0], args)), axis=-1))

    fig, ax = plt.subplots(figsize=(15, 4))

    for i, (data, rms, label, c, lw, ls) in enumerate(zip(args,
                                                          rmss,
                                                          labels,
                                                          colours,
                                                          lws,
                                                          styles
                                                          )):
        if i >= 1:
            label = label + f' [RMS: {rms:.3e}]'
        ax.plot(data[:samples], label=label, c=c, lw=lw, ls=ls)

    ax.set_title(f'Neural network prediction')
    ax.set_xlabel('Data instance number')
    ax.set_ylabel('y (Target)')
    ax.tick_params(axis='both', which='major')
    ax.legend()
    ax.grid(color='k', alpha=0.2)

    plt.tight_layout()
    plt.show()

    return


def plot_zoeppritz(theta, r, r_hat, min_theta=None, max_theta=None):
    """
    Plot the result of the Zoeppritz comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(theta, r, label="Ground truth", lw=2)
    ax.plot(theta, r_hat, label="Network output", lw=2)

    if (min_theta is not None) and (max_theta is not None):
        ax.axvspan(min_theta, max_theta, facecolor='k', alpha=0.1, lw=0)
        ax.axhline(0, lw=0.75, color='k')
        ax.text(np.mean([min_theta, max_theta]), plt.gca().get_ylim()[1]-0.005,
                s="ϑ TRAINING DOMAIN",
                fontsize=14, color=(0.4, 0.4, 0.4),
                va='top', ha='center')

    ax.grid(color='k', alpha=0.15)
    ax.set_xlabel('Incidence angle, theta [deg]', size=14)
    ax.set_ylabel('Amplitude', size=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()
    return


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
