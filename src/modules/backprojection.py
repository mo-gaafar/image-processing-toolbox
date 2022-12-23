import numpy as np
from modules.interpolators import bilinear_interp_pixel, nn_interp_pixel


def nasser_sinogram(image_arr, angles=None, projections=None):
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
    sinogram = np.zeros((len(angles), len(projections)))
    for i in range(len(angles)):
        for j in range(len(projections)):
            # get projection
            projection = radon_projection(image_arr, angle=[angles[i]])
            sinogram[i][j] = projection[0][j]

    return sinogram


def radon_projection(image_arr, angle, center=None, bilinear=True):
    # rotate image by angle and get projection line
    # using linalg and linear interpolation
    # get rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    # get image dimensions
    height, width = image_arr.shape
    if center is None:
        # get center of image
        center = np.array([width / 2, height / 2])

    # get rotated image
    rotated_image = np.zeros(image_arr.shape)
    for i in range(height):
        for j in range(width):
            # get rotated pixel
            rotated_pixel = np.dot(
                rotation_matrix, np.array([j, i]) - center) + center
            # get pixel value using linear interpolation
            if bilinear == True:
                rotated_image[i][j] = bilinear_interp_pixel(
                    image_arr, rotated_pixel[0], rotated_pixel[1])
            else:
                rotated_image[i][j] = nn_interp_pixel(
                    image_arr, rotated_pixel[0], rotated_pixel[1])

    # scan rotated image and get projection line by line
    projection = np.zeros(width)
    for i in range(width):
        # average of pixel values
        projection[i] = np.sum(rotated_image[:, i])/height

    # # spread projection to 2D by repeating it
    # projection_2d = np.zeros(image_arr.shape)
    # for i in range(height):
    #     projection_2d[i] = projection

    return projection
