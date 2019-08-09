import csv
import logging
import os

try:
    import cupy as np
    GPU_AVAILABLE = True
except ModuleNotFoundError:
    logging.warning('cupy not installed, '
                    'falling back to cpu only calculations with numpy')
    import numpy as np
    GPU_AVAILABLE = False

import click
import numpy  # this line is not redundant
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
        image = np.pad(image, pad_width, mode='constant', constant_values=0)
    image = np.roll(image, shift[0], axis=1)
    image = np.roll(image, shift[1], axis=0)
    return image


def register_translation(src_image, target_image,
                         max_shift_mask=None, bandpass_mask=None):
    """Calculate pixel shift between two input images.

    This function runs with numpy or cupy for GPU acceleration.

    Parameters
    ----------
    src_image : array
        Reference image.
    target_image : array
        Image to register.  Must be same dimensionality as ``src_image``.
    max_shift_mask : array
        The fourier mask restricting the maximum allowable pixel shift.
    bandpass_mask : array
        Fourier mask image array, by default None.

    Returns
    -------
    shifts : ndarray
        Pixel shift in x, y order between target and source image.

    References
    ----------
    scikit-image register_translation function in the skimage.feature module.
    """
    src_image = np.array(src_image)
    target_image = np.array(target_image)
    src_freq = np.fft.fftn(src_image)
    target_freq = np.fft.fftn(target_image)
    # Fourier bandpass filtering
    if bandpass_mask:
        src_freq = src_freq * bandpass_mask
        target_freq = target_freq * bandpass_mask
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)
    # Limit maximum allowable shift
    if max_shift_mask:
        cross_correlation = cross_correlation * max_shift_mask
    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([float(np.fix(axis_size / 2)) for axis_size in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    shifts = np.flip(shifts, axis=0).astype(np.int)  # x, y order
    return shifts


def normalize_image(image):
    """Ensure the image mean is zero and the standard deviation is one.

    Parameters
    ----------
    image : ndarray
        The input image array.

    Returns
    -------
    image
        The normalized image.
        The mean intensity is equal to zero and standard deviation equals one.
    """
    image = image - np.mean(image)
    image = image / np.std(image)
    return image


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
    # fourier space origin should be in the corner
    bandpass_mask = np.roll(bandpass_mask,
                            (np.array(image_shape) / 2).astype(int),
                            axis=(0, 1))
    return bandpass_mask


def max_shift_mask(image_shape, max_allowable_shift):
    """Create a fourier mask to restrict image registration shift values.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    max_allowable_shift : int
        Maximum allowable pixel shift for image registration.

    Returns
    -------
    fourier_mask : ndarray
        The fourier mask for restricting image registration shifts.
    """
    fourier_mask = bandpass_mask(image_shape, outer_radius=max_allowable_shift)
    return fourier_mask


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


def read_files(filenames, cropping_coordinates=None, normalize=True):
    """Generator returning paired image frames from a list of filenames.

    Parameters
    ----------
    filenames : listlike, str
        Ordered list of filenames corresponding to images on disk.
    cropping_coordinates : tuple
        Coordinates for cropping images, (topleft_x, topleft_y, width, height)
    """
    for i, _ in enumerate(filenames[:-1]):
        filename_1 = filenames[i]
        filename_2 = filenames[i + 1]
        image_1 = skimage.io.imread(filename_1)
        image_2 = skimage.io.imread(filename_2)
        if cropping_coordinates:
            topleft_x, topleft_y, width, height = cropping_coordinates
            slicer = (slice(topleft_y, topleft_y + height),
                      slice(topleft_x, topleft_x + width))
            image_1 = np.array(image_1[slicer])
            image_2 = np.array(image_2[slicer])
        else:
            # This is not redundant if you are overriding numpy with cupy
            image_1 = np.array(image_1)
            image_2 = np.array(image_2)
        if normalize:
            image_1 = normalize_image(image_1)
            image_2 = normalize_image(image_2)
        yield image_1, image_2


def align_and_save_images(filenames, output_directory, cumulative_shift_df,
                          pad_width=None, gpu=GPU_AVAILABLE):
    """
    Aligns and saves images.

    Parameters
    ----------
    filenames : listlike, str
        Ordered list of filenames (alphabetical ordering).
    output_directory : str
        Directory location to save the output images.
    cumulative_shift_df : DataFrame
        Dataframe containing two columns named 'x_shift', and 'y_shift'.
        The values in the columns must be the cumulative shift,
        so the shift is relative to the very first image frame in the sequence.
    pad_width : {sequence, array_like, int}, optional
        Padding widths in the form expected by numpy.pad, by default None.
        ((before_1, after_1), … (before_N, after_N))

    Returns
    -------
    output_filename, aligned_image
    """
    for filename, (idx, row) in zip(filenames, cumulative_shift_df.iterrows()):
        image = skimage.io.imread(filename)
        image = np.array(image)  # not redundant, if you import cupy as np
        shift = [int(row['x_shift']), int(row['y_shift'])]
        aligned_image = align_images(image, shift, pad_width=pad_width)
        # this could be handled in a nicer way, by splitting the filename
        output_filename = os.path.join(
            output_directory, 'Aligned_' + os.path.basename(filename))
        if gpu:
            skimage.io.imsave(output_filename, np.asnumpy(aligned_image))
        else:
            skimage.io.imsave(output_filename, aligned_image)
        print('Saved: {}'.format(output_filename))
        yield output_filename, aligned_image


def check_directory(directory_path):
    # If the directory doesn't exist, create it
    # If the directory does exist, check it is empty
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        logging.info('Created new directory: {}'.format(directory_path))
    else:
        if not os.listdir(directory_path) == []:
            message_directory_is_not_empty = (
                'Output directory must be empty. '
                'Please choose another location to save output files.'
            )
            logging.error(message_directory_is_not_empty)
            raise ValueError(message_directory_is_not_empty)
        if not os.access(directory_path, os.W_OK):
            message_directory_is_not_writable = (
                'Output directory must have correct permissions to write data.'
                'Please choose another location to save output files.'
            )
            logging.error(message_directory_is_not_writable)
            raise ValueError(message_directory_is_not_writable)


@click.command()
@click.argument('input_directory',
                type=click.Path(exists=True, dir_okay=True, writable=True))
@click.argument('regex_pattern')
@click.argument('output_directory',
                type=click.Path(exists=True, dir_okay=True, writable=True))
def run_full_pipeline(input_directory, regex_pattern, output_directory):
    pipeline(input_directory, regex_pattern, output_directory)


def pipeline(input_directory, regex_pattern, output_directory):
    # Set up output file location
    check_directory(output_directory)
    output_relative_shifts = os.path.join(
        output_directory, 'relative_shifts.csv')
    output_cumulative_shifts = os.path.join(
        output_directory, 'cumulative_shifts.csv')
    # Set up inputs
    full_regex_pattern = os.path.join(input_directory, regex_pattern)
    filenames = find_filenames(full_regex_pattern)
    # Pipeline stage 1
    # Calculate relative shifts
    with open(output_relative_shifts, "w") as f:
        writes = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
        writes.writerows([['x_shift', 'y_shift']])
        writes.writerows([[0, 0]])   # the first frame is the anchor, no shift
        writes.writerows(calculate_relative_shifts(filenames))
    # Pipeline stage 2
    # Calculate the cumulative shifts
    relative_shift_df = pd.read_csv(output_relative_shifts)
    cumulative_shift_df = calculate_cumulative_shifts(relative_shift_df)
    cumulative_shift_df.to_csv(output_cumulative_shifts)
    # Pipeline stage 3
    # Aligning and saving the images
    # Must have relative_shift_df and cumulative_shift_df both in memory
    pad_width = calculate_padding(cumulative_shift_df)
    mygenerator = align_and_save_images(
        filenames, output_directory, cumulative_shift_df, pad_width=pad_width)
    for filename_out, _ in mygenerator:
        print(filename_out)


if __name__ == '__main__':
    run_full_pipeline()
