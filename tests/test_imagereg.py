import os

try:
    import cupy as np
    GPU_AVAILABLE = True
except ModuleNotFoundError:
    logging.warning('cupy not installed, '
                    'falling back to cpu only calculations with numpy')
    import numpy as np
    GPU_AVAILABLE = False

import pandas as pd
import pytest
import skimage.io

from imagereg.main import (align_images,
                           calculate_cumulative_shifts,
                           calculate_padding,
                           calculate_relative_shifts,
                           check_directory,
                           find_filenames,
                           normalize_image,
                           pipeline,
                           register_translation,
                           )


def test_pipeline(tmp_path):
    input_directory = os.path.join(os.path.dirname(__file__), 'images')
    regex_pattern = 'img[0-9].tif'
    output_directory = tmp_path / "pipeline_test_output"
    output_directory.mkdir()
    pipeline(input_directory, regex_pattern, output_directory)
    expected_filenames = ['Aligned_img1.tif',
                          'Aligned_img2.tif',
                          'Aligned_img3.tif',
                          'cumulative_shifts.csv',
                          'relative_shifts.csv']
    expected_filenames.sort()
    result_filenames = os.listdir(output_directory)
    result_filenames.sort()
    assert len(result_filenames) == 5  # 3 aligned images & 2 csv
    assert result_filenames == expected_filenames
    expected_output_dir = os.path.join(os.path.dirname(__file__),
                                       'image_output')
    for i in range(1, 4):
        output_image = skimage.io.imread(
            os.path.join(output_directory, 'Aligned_img{}.tif'.format(i)))
        expected_output = skimage.io.imread(
            os.path.join(expected_output_dir, 'Aligned_img{}.tif'.format(i)))
        assert np.allclose(output_image, expected_output)


def test_check_directory_empty(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    check_directory(d)


def test_check_directory_not_empty(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("Contents of the file...")
    assert p.read_text() == "Contents of the file..."
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError):
        check_directory(d)


def test_find_filenames():
    full_regex = os.path.join(os.path.dirname(__file__),
                              'images', 'img[0-9].tif')
    filenames = find_filenames(full_regex)
    assert len(filenames) == 3
    assert filenames[0].endswith('img1.tif')
    assert filenames[1].endswith('img2.tif')
    assert filenames[2].endswith('img3.tif')


def test_calculate_relative_shifts():
    filename_1 = os.path.join(os.path.dirname(__file__),
                              'images', 'img1.tif')
    filename_2 = os.path.join(os.path.dirname(__file__),
                              'images', 'img2.tif')
    filename_3 = os.path.join(os.path.dirname(__file__),
                              'images', 'img3.tif')
    filenames = [filename_1, filename_2, filename_3]
    generator = calculate_relative_shifts(filenames)
    result_1 = next(generator)  # relative shift between image 1 and 2
    result_2 = next(generator)  # relative shift between image 2 and 3
    assert list(result_1) == [18,  3]
    assert list(result_2) == [13, -5]


def test_calculate_cumulative_shifts():
    contents = {'x_shift': [0, 18, 13],
                'y_shift': [0, 3, -5]}
    relative_shift_df = pd.DataFrame.from_dict(contents)
    result = calculate_cumulative_shifts(relative_shift_df)
    expected_x = [0, 18, 31]
    expected_y = [0, 3, -2]
    assert result['x_shift'].to_list() == expected_x
    assert result['y_shift'].to_list() == expected_y


def test_calculate_padding():
    contents = {'x_shift': [0, 18, 13],
                'y_shift': [0, 3, -5]}
    relative_shift_df = pd.DataFrame.from_dict(contents)
    expected = ((5, 3), (0, 18))
    result = calculate_padding(relative_shift_df)
    assert result == expected


def test_register_translation():
    filename_1 = os.path.join(os.path.dirname(__file__),
                              'images', 'img1.tif')
    filename_2 = os.path.join(os.path.dirname(__file__),
                              'images', 'img2.tif')
    src_image = skimage.io.imread(filename_1)
    target_image = skimage.io.imread(filename_2)
    result = register_translation(src_image, target_image)
    assert np.allclose(result, np.array([18, 3]))


def test_normalize_image():
    image = np.random.random((10, 10))
    image = image * 10  # change the standard deviation
    image = image + 50  # change the mean intensity
    result = normalize_image(image)
    assert np.isclose(np.mean(result), 0)
    assert np.isclose(np.std(result), 1)


@pytest.mark.parametrize("fname", [('img1.tif'), ('img2.tif'), ('img3.tif')])
def test_normalize_image_real_data(fname):
    filename = os.path.join(os.path.dirname(__file__),
                            'images', fname)
    image = skimage.io.imread(filename)
    result = normalize_image(image)
    assert np.isclose(np.mean(result), 0)
    assert np.isclose(np.std(result), 1)


@pytest.mark.parametrize("shift, padding, expected", [
    ([0, 0],
     ((0, 0), (0, 0)),
     [[1, 2], [3, 4]]
     ),
    ([0, 0],
     ((1, 1), (1, 1)),
     [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
     ),
    ([0, 1],
     ((1, 1), (1, 1)),
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0]]
     ),
    ([1, 0],
     ((1, 1), (1, 1)),
     [[0, 0, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4], [0, 0, 0, 0]]
     ),
    ([1, 1],
     ((1, 1), (1, 1)),
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]]
     ),
    ([-1, -1],
     ((1, 1), (1, 1)),
     [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
     ),
    ([-1, 0],
     ((1, 1), (1, 1)),
     [[0, 0, 0, 0], [1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0]]
     ),
    ([0, -1],
     ((1, 1), (1, 1)),
     [[0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
     ),
    ([0, -1],
     ((1, 0), (0, 0)),
     [[1, 2], [3, 4], [0, 0]]
     ),
    ([1, 0],
     None,
     [[2, 1], [4, 3]]
     ),
    ([0, 1],
     None,
     [[3, 4], [1, 2]]
     ),
    ([1, 1],
     None,
     [[4, 3], [2, 1]]
     ),
    ([-1, -1],
     None,
     [[4, 3], [2, 1]]
     ),
    ([0, -1],
     None,
     [[3, 4], [1, 2]]
     ),
    ([-1, 0],
     None,
     [[2, 1], [4, 3]]
     ),
])
def test_align_images(shift, padding, expected):
    input_image = np.array([[1, 2], [3, 4]])
    expected = np.array(expected)
    result = align_images(input_image, shift, pad_width=padding)
    assert np.allclose(result, expected)
