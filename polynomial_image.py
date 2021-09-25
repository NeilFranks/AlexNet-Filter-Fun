import warnings

import numpy as np

from PIL import Image

def image_as_polynomial(image, degree):
    warnings.simplefilter('ignore', np.RankWarning)
    
    pixels = np.asarray(image)
    rows = []   
    for r in pixels:
        red = []
        green = []
        blue = []
        for p in r:
            red.append(p[0])
            green.append(p[1])
            blue.append(p[2])
        rows.append([np.polyfit(range(len(r)), red, degree),
                    np.polyfit(range(len(r)), green, degree),
                    np.polyfit(range(len(r)), blue, degree)
        ])
    return rows

def image_to_sub_polynomials(image, degree):
    warnings.simplefilter('ignore', np.RankWarning)

    subrow_length = int(degree/2)  # how many pixels the sub poly covers
    
    pixels = np.asarray(image)
    rows = []   
    for r in pixels:
        red = []
        green = []
        blue = []
        for p in r:
            red.append(p[0])
            green.append(p[1])
            blue.append(p[2])

        subrows = []
        subrow_start = 0
        subrow_end = subrow_length
        while subrow_start < len(r)-1:
            if subrow_end > len(r)-1:
                subrow_end = len(r)-1
            sub_red = red[subrow_start:subrow_end]
            sub_green = green[subrow_start:subrow_end]
            sub_blue = blue[subrow_start:subrow_end]
            subrows.append([np.polyfit(range(subrow_start, subrow_end), sub_red, degree),
                    np.polyfit(range(subrow_start, subrow_end), sub_green, degree),
                    np.polyfit(range(subrow_start, subrow_end), sub_blue, degree)
            ])

            subrow_start = subrow_end
            subrow_end = subrow_start + subrow_length
        
        rows.append(subrows)
    return rows


def construct_image_from_polynomials(polynomials, dimension):
    width, _ = dimension
    pixels = []
    for p in polynomials:
        row = []
        red_poly = np.poly1d(p[0])
        green_poly = np.poly1d(p[1])
        blue_poly = np.poly1d(p[2])
        for w in range(width):
            row.append([red_poly(w), green_poly(w), blue_poly(w)])
        pixels.append(row)
    
    image = Image.fromarray(np.uint8(pixels)).convert('RGB')
    return image

def construct_image_from_sub_polynomials(polynomials, original_dimensions, desired_dimensions):
    original_width = original_dimensions[0]
    original_height = original_dimensions[1]
    desired_width = desired_dimensions[0]
    desired_height = desired_dimensions[1]

    pixels = []
    for p in polynomials:
        row = []

        for w in range(desired_width):
            sub_p = p[int(len(p)*w/desired_width)]
            red_poly = np.poly1d(sub_p[0])
            green_poly = np.poly1d(sub_p[1])
            blue_poly = np.poly1d(sub_p[2])
            row.append([red_poly(original_width*w/desired_width), green_poly(original_width*w/desired_width), blue_poly(original_width*w/desired_width)])
            
        for _ in range(int(desired_height/original_height)):
            pixels.append(row)
    
    image = Image.fromarray(np.uint8(pixels)).convert('RGB')
    return image

path_to_image = "./sample_images_256/landscape.JPEG"
original = Image.open(path_to_image)

degree = 20

# rows = image_as_polynomial(original, degree)
# image = construct_image_from_polynomials(rows, (256, 256))
rows = image_to_sub_polynomials(original, degree)
image = construct_image_from_sub_polynomials(rows, (256, 256), (2560, 2560))

image.show()


a = 21
