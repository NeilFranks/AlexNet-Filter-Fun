import os
import pathlib
import shutil

from PIL import Image


import utils

def test_make_256x256_LANDSCAPE():
    path_to_image = "./sample_images/landscape.JPEG"
    original = Image.open(path_to_image)
    original_width, original_height = original.size
    assert original_width != 256
    assert original_height != 256

    result = utils.make_256x256(path_to_image)
    result_width, result_height = result.size
    
    assert result_width == 256
    assert result_height == 256
    # result.show()

def test_make_256x256_SQUARE():
    path_to_image = "./sample_images/square.JPEG"
    original = Image.open(path_to_image)
    original_width, original_height = original.size
    assert original_width != 256
    assert original_height != 256

    result = utils.make_256x256(path_to_image)
    result_width, result_height = result.size
    
    assert result_width == 256
    assert result_height == 256
    # result.show()

def test_make_256x256_PORTRAIT():
    path_to_image = "./sample_images/portrait.JPEG"
    original = Image.open(path_to_image)
    original_width, original_height = original.size
    assert original_width != 256
    assert original_height != 256

    result = utils.make_256x256(path_to_image)
    result_width, result_height = result.size
    
    assert result_width == 256
    assert result_height == 256
    # result.show()

def test_copy_folder_as_256x256():
    path_to_folder = "./sample_images"
    path_to_256 = "./sample_images_256"
    if os.path.isdir(path_to_256):
        dir256 = pathlib.Path(path_to_256)
        shutil.rmtree(dir256)

    utils.copy_folder_as_256x256(path_to_folder)
    _, _, filenames_256 = next(os.walk(path_to_256), (None, None, []))
    for filename in filenames_256:
        im = Image.open(path_to_256 + "/" + filename)
        assert im.size == (256, 256)

# test_make_256x256_LANDSCAPE()
# test_make_256x256_SQUARE()
# test_make_256x256_PORTRAIT()
test_copy_folder_as_256x256()
