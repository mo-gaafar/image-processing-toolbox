import numpy as np
from modules.interpolators import bilinear_interp_pixel, nn_interp_pixel


def nasser_sinogram(image_arr, angles=None):
    """Gets sinogram of image.

    Params:
        image_arr: image array
    Returns:
        sinogram: sinogram of image
    """
    from skimage.transform import radon

    # get image dimensions
    height, width = image_arr.shape

    # # get number of angles
    # angles = np.arange(0, 180)

    # get number of projections
    projections = np.arange(0, width)

    # get sinogram
    sinogram = np.zeros((len(projections), len(angles)))
    for i in range(len(angles)):
        sinogram[:, i] = radon_projection(image_arr, angle=angles[i])

    return sinogram


def radon_projection(image_arr, angle, center=None, bilinear=True):
    # rotate image by angle and get projection line

    # fix angle bug
    angle = angle+90
    # convert angle to radians
    angle = np.deg2rad(angle)

    # get image dimensions
    height, width = image_arr.shape

    # get center of image if not provided
    if center is None:
        # get center of image
        center = np.array([width / 2, height / 2])

    center_x, center_y = center

    # get rotated image
    rotated_image = np.zeros(image_arr.shape)
    projection = np.zeros(width)
    for x in range(height):
        projection_ray_sum = 0
        for y in range(width):
            # get rotated pixel
            # get the pixel coordinates in the original image
            x1 = x - center_x
            y1 = y - center_y

            # get the pixel coordinates in the rotated image
            x2 = np.cos(angle) * x1 + np.sin(angle) * y1
            y2 = -np.sin(angle) * x1 + np.cos(angle) * y1

            # get the pixel coordinates in the original image
            x2 = x2 + center_x
            y2 = y2 + center_y
            # get pixel value using linear interpolation
            if bilinear == True:
                rotated_image[x][y] = bilinear_interp_pixel(x2, y2,
                                                            image_arr)
            else:
                rotated_image[x][y] = nn_interp_pixel(x2, y2,
                                                      image_arr)

            projection_ray_sum += rotated_image[x][y]
        projection[x] = projection_ray_sum

    # normalize projection intensities
    projection = projection / height

    # # spread projection to 2D by repeating it
    # projection_2d = np.zeros(image_arr.shape)
    # for i in range(height):
    #     projection_2d[i] = projection

    return projection
