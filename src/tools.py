import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

def _load_data(self, data_path):
    '''Extracts the image and header data from the provided protostar outflow path

    Args:
        uniform_path : str
            The part of the fits file `data_path` that is uniform for all protostars

            `data_path = direction_path + protostar_name + uniform_path`
    
    Returns:
        image_data : numpy.ndarray
            The outflow image data
        header : astropy.io.fits.header.Header
            The outflow header
    '''
    
    hdu_list = fits.open(data_path)

    with fits.open(data_path) as hdu_list:
        raw_data = hdu_list[0].data
        raw_data = np.array(raw_data)
        image_data = np.array(raw_data[0, 0], dtype=float)
        header = hdu_list[0].header

    return image_data, header

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class LassoTool:
    def __init__(self, outflow, image, fill_val=0, invert_selection=False, sqrt=False):
        self.image = image.copy()
        self.sqrt = sqrt
        self.fill_val = fill_val
        self.invert_selection = invert_selection
        self.fig, self.ax = plt.subplots()
        if self.sqrt:
            self.ax.imshow(np.sqrt(np.abs(self.image))*np.sign(self.image), cmap='gray', origin='upper')
        else:
            self.ax.imshow(self.image, cmap='gray', origin='upper')
        self.ax.scatter(outflow.protostar.source_choord[0], outflow.protostar.source_choord[1])
        self.mask = np.zeros_like(self.image, dtype=bool)
        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.x_coords, self.y_coords = np.meshgrid(np.arange(self.image.shape[1]), np.arange(self.image.shape[0]))

    def on_select(self, verts):
        path = Path(verts)
        points = np.vstack((self.x_coords.ravel(), self.y_coords.ravel())).T
        self.mask = path.contains_points(points).reshape(self.image.shape)

        if self.invert_selection:
            self.image[~self.mask] = self.fill_val
        else:
            self.image[self.mask] = self.fill_val

        if self.sqrt:
            self.ax.imshow(np.sqrt(np.abs(self.image))*np.sign(self.image), cmap='gray', origin='upper')
        else:
            self.ax.imshow(self.image, cmap='gray', origin='upper')
        self.fig.canvas.draw()

    def show(self):
        plt.show()
        return self.image

def fill_quadrant(image_array, quadrants, origin, fill_value=0):
    """
    Replaces the specified quadrant(s) of an image array with the given fill_value,
    based on a custom origin point.

    """
    import numpy as np
    import matplotlib.pyplot as plt

    height, width = image_array.shape[:2]
    x_orig, y_orig = np.round(origin, 0)
    x_orig, y_orig = int(x_orig), int(y_orig)
    

    modified_image = image_array.copy()

    # Ensure the origin is within the valid range
    if not (0 <= x_orig < width and 0 <= y_orig < height):
        raise ValueError("Origin point must be within image dimensions.")

    if 1 in quadrants:  # Top-Right
        modified_image[:y_orig, x_orig:] = fill_value
    if 2 in quadrants:  # Top-Left
        modified_image[:y_orig, :x_orig] = fill_value
    if 3 in quadrants:  # Bottom-Left
        modified_image[y_orig:, :x_orig] = fill_value
    if 4 in quadrants:  # Bottom-Right
        modified_image[y_orig:, x_orig:] = fill_value

    return modified_image

def _rotate_outflow(self, image, angle, reshape=False):
    """
    Rotates a binary image (numpy array with 1s and 0s) around a specified pivot point.

    Parameters:
    - image (numpy.ndarray): 2D binary array (1s and 0s).
    - angle (float): Rotation angle in degrees (counterclockwise).
    - pivot (tuple): (y, x) coordinates of the point to rotate around.
    - reshape (bool): If True, the output shape adapts to avoid cropping.
                    If False, the output keeps the original shape, potentially cropping parts.

    Returns:
    - rotated_image (numpy.ndarray): Rotated 2D binary image.
    """
    from scipy.ndimage import rotate, shift

    h, w = image.shape
    cy, cx = self.protostar.source_choord  # Pivot point (row, column)

    # Calculate translation to center pivot point at the center of the image
    shift_y, shift_x = h // 2 - cy, w // 2 - cx
    shifted_image = shift(image, shift=(shift_y, shift_x), mode='constant', cval=0)

    # Rotate the shifted image
    rotated_shifted = rotate(shifted_image, angle, reshape=reshape, order=0, mode='constant', cval=0)

    # Calculate inverse shift to move pivot back to its original location
    inv_shift_y, inv_shift_x = -shift_y, -shift_x
    self.image_rotated = shift(rotated_shifted, shift=(inv_shift_y, inv_shift_x), mode='constant', cval=0)
    self.angle_rot = angle

    return rotated_shifted

from astropy.stats import sigma_clipped_stats

def _mask(image, sigma):
    mean, median, std = sigma_clipped_stats(image)
    threshold = sigma * std
    return np.where((image < threshold), 0, 1)

def _detect_edge(image, low_threshold=0, high_threshold=4):
    from astropy.stats import sigma_clipped_stats
    from skimage import feature

    mean, median, std = sigma_clipped_stats(image)

    # Apply Canny edge detection
    edges = feature.canny(image.astype(float), sigma=std, low_threshold=low_threshold, high_threshold=high_threshold)

    return edges

def _find_max_sigma_val(self, image, starting_sigma=1, max_sigma_limit=None):
    max_sigma_found = False
    sigma = starting_sigma

    while not(max_sigma_found):
        masked_image = _mask(image, sigma)
        detected_edge = _detect_edge(masked_image)
        y, x = np.nonzero(detected_edge)
        
        mask_pass_source = np.min(x) >= self.protostar.source_choord[0] if self.color == 'blue' else np.max(x) <= self.protostar.source_choord[0]
        if mask_pass_source or sigma == max_sigma_limit:
            max_sigma_found = True
            return sigma
        else:
            sigma+=1

def _split_edge(loc, edge):
    '''Removes duplicate values retaining only min and max.

    Returns two arrays, one containing the max values (top wing) and the other containing the min values (bot_wing)
    '''

    y, x = np.nonzero(edge)
    x_unique = np.unique(x)

    indices = []

    for val in x_unique:
        all_points_at_val = np.where(x == val)[0]
        if loc == 'bot':
            indices = np.append(indices, np.min(all_points_at_val))
        elif loc == 'top':
            indices = np.append(indices, np.max(all_points_at_val))
        else:
            raise ValueError(f"Invalid location: {loc}")

    indices = indices.astype(int)

    return x[indices], y[indices]

def polynomial_fit(x, y, deg, weights=None):
    coeffs = np.polyfit(x, y, deg, w=weights)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)

    y_pred = poly(x)
    chi_squared = np.sum((y - y_pred) ** 2)

    return x_fit, y_fit, coeffs, chi_squared

def reduced_chi_squared(header, chi_squared, poly_order, fit_array):
    '''Calculates the reduced chi squared value of the polynomial fit
    '''
    pixel_size = header['CDELT2']
    obs_res = np.sqrt(header['BMAJ'] * header['BMIN'])

    # The length of the polyfit x|y array is the num_fit_pixels
    num_fit_pixels = len(fit_array)
    correction_factor = 0.5 * obs_res / pixel_size
    degree_free = num_fit_pixels / correction_factor - poly_order
    reduced_chi = chi_squared / degree_free

    return reduced_chi

def find_best_poly_order_fit(x, y, header, poly_range=(1, 10)):
    '''Finds the best polynomial fit based on the reduced chi squared value
    '''
    chi_list = []
    for poly_order in range(poly_range[0], poly_range[1]):
        x_fit, y_fit, poly_coeffs, chi_squared = polynomial_fit(x, y, poly_order)
        chi_list.append(reduced_chi_squared(header, chi_squared, poly_order, x_fit))

    return np.min(chi_list), (np.argmin(chi_list)+poly_range[0])

from scipy.optimize import curve_fit

def calc_opening_angle(right_wing, left_wing):

    right_x = right_wing[0]
    right_y = right_wing[1]
    left_x = left_wing[0]
    left_y = left_wing[1]
    vertex_x = 130
    vertex_y = 130

    # Get values respective to the source as origin
    u_x = right_x - vertex_x
    u_y = right_y - vertex_y
    v_x = left_x - vertex_x
    v_y = left_y - vertex_y

    dot_product = u_x * v_x + u_y * v_y

    magnitude_u = ((u_x)**2 + (u_y)**2)**0.5
    magnitude_v = ((v_x)**2 + (v_y)**2)**0.5
    magnitude = magnitude_u * magnitude_v

    theta = np.round(np.degrees(np.arccos(dot_product / magnitude)), 2)

    return theta

def find_idx_closest_to_length(length, wing):

    vertex_x = 130
    vertex_y = 130

    all_hypot = np.zeros(wing[0].shape)

    for idx in range(len(wing[0])):
        u_x = abs(wing[0][idx] - vertex_x)
        u_y = abs(wing[1][idx] - vertex_y)

        all_hypot[idx] = (np.hypot(u_x, u_y))

    closest_value = all_hypot[np.abs(all_hypot - length).argmin()]
    closest_idx = np.where(all_hypot == closest_value)[0]

    x = wing[0][closest_idx]
    y = wing[1][closest_idx]

    return (x, y)

def calc_opening_angle_spectrum(top_fit, bot_fit):

    angles=[]
    for distance in range(0, 100):
        top_xy = find_idx_closest_to_length(distance, top_fit)
        bot_xy = find_idx_closest_to_length(distance, bot_fit)
        opening_angle = calc_opening_angle(top_xy, bot_xy)
        angles.append(opening_angle)

    return angles

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def open_pickle(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj
