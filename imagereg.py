import csv

import cupy as np
import numpy
import pandas as pd
import skimage.draw
import skimage.io


def align_images(image, shift, pad_width=None):
    """Align an image, given a pixel shift.

    Parameters
    ----------
    image : ndarray
        Input image
    shift : listlike
        Pixel shift in x, y format. Integers only.
    pad_width : {sequence, array_like, int}, optional
        Padding widths in the form expected by numpy.pad, by default None.
        ((before_1, after_1), … (before_N, after_N))

    Returns
    -------
    image
        The ahifted image.
    """
    if pad_width:
        image = np.pad(image, pad_wdith, mode='constant', constant_values=0)
    image = np.roll(image, shift[0], axis=1)
    image = np.roll(image, shift[1], axis=0)
    return image


def register_translation(src_image, target_image):
    """Calculate pixel shift between two input images.

    This function runs with numpy or cupy for GPU acceleration.

    Parameters
    ----------
    src_image : array
        Reference image.
    target_image : array
        Image to register.  Must be same dimensionality as ``src_image``.

    Returns
    -------
    shifts : ndarray
        Pixel shift in x, y order between target and source image.

    References
    ----------
    scikit-image register_translation function in the skimage.feature module.
    """
    src_freq = np.fft.fftn(src_image)
    target_freq = np.fft.fftn(target_image)
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)
    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([float(np.fix(axis_size / 2)) for axis_size in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    shifts = np.flip(shifts, axis=0)  # x, y order
    return shifts.astype(np.int)


def bandpass_mask(image_shape, outer_radius, inner_radius=0):
    """Create a fourier bandpass mask.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    outer_radius : int
        Outer radius for bandpass filter array.
    inner_radius : int, optional
        Inner radius for bandpass filter array, by default 0

    Returns
    -------
    bandpass_mask : ndarray
        The bandpass image mask.
    """
    bandpass_mask = numpy.zeros(image_shape)
    r, c = numpy.array(image_shape) / 2
    inner_circle_rr, inner_circle_cc = skimage.draw.circle(
        r, c, inner_radius, shape=image_shape)
    outer_circle_rr, outer_circle_cc = skimage.draw.circle(
        r, c, outer_radius, shape=image_shape)
    bandpass_mask[outer_circle_rr, outer_circle_cc] = 1.0
    bandpass_mask[inner_circle_rr, inner_circle_cc] = 0.0
    bandpass_mask = np.array(bandpass_mask)
    return bandpass_mask


def calculate_relative_shifts(filenames):
    """Calculate the relative (pair-wise) shifts between images in a sequence.

    Parameters
    ----------
    filenames : listlike, str
        Ordered list of filenames
    """
    my_generator = read_files(filenames)
    for (img1, img2) in my_generator:
        shift = register_translation(img1, img2)
        yield shift


def calculate_cumulative_shifts(relative_shift_df):
    """Calculates cumulative shifts from a pandas dataframe of relative shifts.

    Parameters
    ----------
    calculate_cumulative_shifts : DataFrame
        Dataframe containing two columns named 'x_shift', and 'y_shift'.
        The values in the columns must be the relative shift between
        the current image frame and the one immediately preceeding it.

    Returns
    -------
    cumulative_df
        Dataframe containing two columns named 'x_shift', and 'y_shift'.
        Shifts are now all relative to the first image frame.
    """
    cumulative_df = pd.DataFrame()
    cumulative_df['x_shift'] = relative_shift_df.x_shift.cumsum()
    cumulative_df['y_shift'] = relative_shift_df.y_shift.cumsum()
    return cumulative_df


def calculate_padding(relative_shift_df):
    """Calculates the total amount of padding to place around a shifted image.

    Parameters
    ----------
    relative_shift_df : DataFrame
        Dataframe containing two columns named 'x_shift', and 'y_shift'.
        The values in the columns must be the relative shift between
        the current image frame and the one immediately preceeding it.

    Returns
    -------
    pad_width : {sequence, array_like, int}, optional
        Padding widths in the form expected by numpy.pad, by default None.
        ((before_1, after_1), … (before_N, after_N))
    """
    x_pad_left = min(relative_shift_df['x_shift'])
    x_pad_right = max(relative_shift_df['x_shift'])
    y_pad_top = min(relative_shift_df['y_shift'])
    y_pad_bottom = max(relative_shift_df['y_shift'])
    # Clip to allowable range and make sure values are positive
    x_pad_left = abs(int(np.minimum(x_pad_left, 0)))
    x_pad_right = abs(int(np.maximum(x_pad_right, 0)))
    y_pad_top = abs(int(np.minimum(y_pad_top, 0)))
    y_pad_bottom = abs(int(np.maximum(y_pad_bottom, 0)))
    # Pack into the form numpy/cupy likes
    pad_width = ((y_pad_top, y_pad_bottom), (x_pad_left, x_pad_right))
    return pad_width


def find_filenames(filename_pattern):
    """Return ordered list of filenames matching a regex pattern.

    Parameters
    ----------
    filename_pattern : str
        Regex pattern matching files of interest.

    Returns
    -------
    filenames
        Ordered list of filenames (alphabetical ordering).
    """
    image_collection = skimage.io.imread_collection(
        load_pattern=filename_pattern)
    filenames = image_collection.files
    return filenames


def read_files(filenames):
    """Generator returning paired image frames from a list of filenames.

    Parameters
    ----------
    filenames : listlike, str
        Ordered list of filenames corresponding to images on disk.
    """
    for i, _ in enumerate(filenames[:-1]):
        filename_1 = filenames[i]
        filename_2 = filenames[i + 1]
        image_1 = skimage.io.imread(filename_1)
        image_2 = skimage.io.imread(filename_2)
        # This is not redundant if you are overriding numpy with cupy
        image_1 = np.array(image_1)
        image_2 = np.array(image_2)
        yield image_1, image_2


def align_and_save_images(filenames, cumulative_shift_df, pad_width=None,
                          gpu=False):
    """

    Parameters
    ----------
    filenames : listlike, str
        Ordered list of filenames (alphabetical ordering).
    cumulative_shift_df : DataFrame
        Dataframe containing two columns named 'x_shift', and 'y_shift'.
        The values in the columns must be the cumulative shift,
        so the shift is relative to the very first image frame in the sequence.
    pad_width : {sequence, array_like, int}, optional
        Padding widths in the form expected by numpy.pad, by default None.
        ((before_1, after_1), … (before_N, after_N))
    """
    for filename, (idx, row) in zip(filenames, cumulative_shift_df.iterrows()):
        image = skimage.io.imread(filename)
        image = np.array(image)  # not redundant, if you import cupy as np
        shift = [int(row['x_shift']), int(row['y_shift'])]
        aligned_image = align_images(image, shift, pad_width=pad_width)
        # this could be handled in a nicer way, by splitting the filename
        output_filename = 'My_Aligned_' + filename
        if gpu:
            skimage.io.imsave(output_filename, np.asnumpy(aligned_image))
        else:
            skimage.io.imsave(output_filename, aligned_image)
        print('Saved: {}'.format(output_filename))
        yield filename, aligned_image


def main():
    regex_pattern = 'SEM Image 2-[0-9][0-9][0-9]_000-000.tif'
    filenames = find_filenames(regex_pattern)

    output_relative_shifts = 'relative_shifts.csv'
    output_cumulative_shifts = 'cumulative_shifts.csv'
    with open(output_relative_shifts, "w") as f:
        writes = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
        writes.writerows([['x_shift', 'y_shift']])
        writes.writerows([[0, 0]])   # the first frame is the anchor, no shift
        writes.writerows(calculate_relative_shifts(filenames))

    relative_shift_df = pd.read_csv(output_relative_shifts)
    cumulative_shift_df = calculate_cumulative_shifts(relative_shift_df)
    relative_shift_df.to_csv(output_cumulative_shifts)
    pad_wdith = calculate_padding(relative_shift_df)
    mygenerator = align_and_save_images(filenames, cumulative_shift_df,
                                        pad_width=pad_width, gpu=True)
    for ff, aa in mygenerator:
        print(ff)


if __name__ == '__main__':
    main()
