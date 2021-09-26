import json
import numpy as np
import os
import pathlib
import shutil
import time

from PIL import Image
import cv2


def make_256x256(path_to_image):
    """
    From AlexNet paper:
    ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality.
    Therefore, we down-sampled the images to a fixed resolution of 256 × 256. 
    Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then
    cropped out the central 256×256 patch from the resulting image. 
    """

    im = Image.open(path_to_image)
    width, height = im.size
    if width < height:
        new_width = 256
        new_height = int(height*(new_width/width))
    else:
        new_height = 256
        new_width = int(width*(new_height/height))

    resized = im.resize((new_width, new_height))

    if new_width != 256:
        cropped = resized.crop(((new_width/2)-128, 0, (new_width/2)+128, new_height))
    elif new_height != 256:
        cropped = resized.crop((0, (new_height/2)-128, new_width, (new_height/2)+128))
    else:
        cropped = resized

    return cropped

def copy_folder_as_256x256(path_to_folder):
    parent_dir = "".join(path_to_folder.split("/")[:-1])
    folder_name = path_to_folder.split("/")[-1]
    path_to_new_folder = parent_dir+"/"+folder_name+"_256"
    os.mkdir(path_to_new_folder, mode=0o777)

    
    with open('output_%s.txt' % folder_name, 'a') as file:
            file.write("GONNA WORK ON %s\n\n\n" % (path_to_folder))

    _, _, filenames = next(os.walk(path_to_folder), (None, None, []))
    for filename in filenames:
        make_256x256(path_to_folder + "/" + filename).save(path_to_new_folder+"/"+filename)

    # subdirs = next(os.walk(path_to_folder))[1]
    # for subdir in subdirs:
    #     try:
    #         with open('output_%s.txt' % folder_name, 'a') as file:
    #             file.write("Starting %s\n" % (path_to_folder + "/" + subdir))

    #         os.mkdir(path_to_new_folder + "/" + subdir)
    #         _, _, filenames = next(os.walk(path_to_folder + "/" + subdir), (None, None, []))
    #         for filename in filenames:
    #             make_256x256(path_to_folder + "/" + subdir + "/" + filename).save(path_to_new_folder+"/"+subdir+"/"+filename)

    #         with open('output_%s.txt' % folder_name, 'a') as file:
    #             file.write("    Done with %s\n" % (path_to_folder + "/" + subdir))
    #     except Exception as e:
    #         with open('output_%s.txt' % folder_name, 'a') as file:
    #             file.write(str(e))
    
def map_folders_to_numbers_json(path_to_dir):
    m = {}

    subdirs = next(os.walk(path_to_dir))[1]
    for i in range(len(subdirs)):
        m[i] = subdirs[i]

    with open('folder_map.json', 'w') as file:
        file.write(json.dumps(m))

def put_val_into_synset_folders(path_to_val_folder, path_to_groundtruth, path_to_synset):
    number_to_index = {}
    with open(path_to_groundtruth, 'r') as file:
        i = 1
        line = file.readline()
        while line:
            line = line.replace("\n", "")
            number_to_index[i] = int(line)
            i += 1
            line = file.readline()

    index_to_synset = {}
    with open(path_to_synset, 'r') as file:
        line = file.readline()  # dump first line
        line = file.readline()
        while line:
            if line != "\n":
                index_to_synset[int(line.split(",")[0])] = line.split(",")[1]
            line = file.readline()

    path_to_synset_val_folder = "".join(path_to_val_folder.split("/")[:-1]) + "/" + "synset_" + str(path_to_val_folder.split("/")[-1])
    if os.path.isdir(path_to_synset_val_folder):
        synset_val = pathlib.Path(path_to_synset_val_folder)
        shutil.rmtree(synset_val)
    os.mkdir(path_to_synset_val_folder, mode=0o777)

    _, _, filenames = next(os.walk(path_to_val_folder), (None, None, []))
    for filename in filenames:
        number = int(filename.split("_")[-1].split(".")[0])
        index = number_to_index[number]
        synset = index_to_synset[index]

        path_to_synset_folder = path_to_synset_val_folder + "/" + synset
        if not os.path.isdir(path_to_synset_folder):
            os.mkdir(path_to_synset_folder, mode=0o777)
        
        shutil.copy(path_to_val_folder + "/" + filename, path_to_synset_folder + "/" + filename)

def find_average_pixels(path_to_folder, dimensions=(256, 256)):
    folder_name = path_to_folder.split("/")[-1]

    number_of_images_seen = 0
    average_image = np.asarray([[[0,0,0]]*dimensions[1]]*dimensions[0], dtype=np.float64)

    subdirs = next(os.walk(path_to_folder))[1]
    with open('output_average_pixels.txt', 'a') as file:
            file.write("%s START\n" % time.time())
    for subdir in subdirs:
        try:
            _, _, filenames = next(os.walk(path_to_folder + "/" + subdir), (None, None, []))
            for filename in filenames:
                pixels = np.asarray(Image.open(path_to_folder + "/" + subdir + "/" + filename), dtype=np.float64)
                if isinstance(pixels[0][0], np.uint8):
                    pixels = cv2.merge((pixels,pixels,pixels))
                assert pixels.shape == average_image.shape
                # for r in range(len(pixels)):
                #     for c in range(len(pixels[r])):
                #         for i in range(len(pixels[r][c])):  #r, g, b
                #             average_image[r][c][i] = ((number_of_images_seen*average_image[r][c][i])+pixels[r][c][i])/(number_of_images_seen+1)
                ave_weight = float(number_of_images_seen)/(number_of_images_seen+1)
                average_image = cv2.addWeighted(average_image, ave_weight, pixels, 1.0-ave_weight, 0.0)
                number_of_images_seen += 1

            with open('output_average_pixels.txt', 'a') as file:
                file.write("%s %s completed\n" % (time.time(), subdir))
            Image.fromarray(average_image.astype(np.uint8), 'RGB').save('./average_image/%s.JPEG' % number_of_images_seen)
        except Exception as e:
            with open('output_average_pixels.txt', 'a') as file:
                file.write("\t%s\n" % str(e))


