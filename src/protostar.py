import astropy.io.fits as fits
import numpy as np
from astropy.stats import sigma_clipped_stats

class Protostar:
    '''The highest level object containing the data for both the red and blue shifted lobes

    Attributes:
        `directory` : str
            The directory containing the red and blue shifted lobe data
        `name` : str
            The name of the protostar
        `blue_shifted` : Outflow
            The blue shifted outflow lobe child object
        `red_shifted` : Outflow
            The red shifted outflow lobe child object
        `source_coord` : Tuple
            A tuple containing the coordinates of the protostar in pixels
        `angle_spectrum_distance_limit` : float
            The minumum of the shorted distance from the source to the tail of each outflow wing
    '''
    def __init__(self, name, source_coord, data_directory_path):
        self.directory_path = data_directory_path
        self.name = name
        self.colors = ['red','blue']
        self.blue_shifted = Outflow(color='blue', protostar=self)
        self.red_shifted = Outflow(color='red', protostar=self)
        self.source_coord = source_coord
        self.angle_spectrum_distance_limit = None

class Outflow:
    '''A child object of protostar that contains the data of one of the protostar's outflows

    Attributes:
        `protostar` : Protostar
            The inherited protostar parent object
        `color` : str
            The color shift of the outflow (red, blue)
        `orientation` : str
            The orientation of the outflow when plotted. red shifted = left, blue shifted = right
        `image` : array
            Contains the raw image of the outflow extracted from the original fits file
        `image_cleaned` : array
            The finalized image after all pre-processing steps have been completed
        `header` : HEADERFits
            The fits header extracted from the original fits file
        `image_rotated` : array
            The rotated image
        `angle_rot` : int/float
            The inputted rotation angle used to make `image_rotated`
        `image_lasso` : array
            The lassoed image
        `min_sigma` : int
            The sigma value that results in the best loose edge detection
        `max_sigma` : int
            The sigma value that results in the best tight edge detection
        `edges` : array
            An 2D array containing all three edge detections (loose, transition, tight)
        `top_wing` : Wing
            A child object containing data of the top wing of the outflow
        `bot_wing` : Wing
            A child object containing data of the bottom wing of the outflow
        `angle_spectrum` : dict
            A dictionary storing the resulting angle spectrum of both the best and 2nd order polynomial fit
        
    '''
    def __init__(self, protostar, color:str):
        self.protostar = protostar
        self.color = color
        self.orientation = None     # Color will decide orientation: red-right, blue-left
        self.image = None
        self.image_cleaned = None
        self.header = None
        self.image_rotated = None
        self.angle_rot = None
        self.image_lasso = None
        self.min_sigma = None
        self.max_sigma = None
        self.edges = None
        self.top_wing = Wing(self, 'top')
        self.bot_wing = Wing(self, 'bot')
        self.angle_spectrum = None

class Wing:
    '''A child object of Outflow containing data of one of the outflow arms

    Attributes:
        `outflow` : Outflow
            The inherited parent outflow object
        `loc` : str
            The location of the wing (top, bot)
        `edges` : array
            The respective edge detections that has been split into top or bottom wing
        `nodes` : array
            The array of nodes used to define the start and end points of each edge detection
        `combined_edges` : array
            The resulting edge detection once combined using the node values
        `poly2` : Polyfit
            The Polyfit child object containing the 2nd degree polynomial fit
        `polyBest` : Polyfit
            The Polyfit child object containing the best polynomial fit
    '''
    def __init__(self, outflow, loc):
        self.outflow = outflow
        self.loc = loc
        self.edges = None
        self.nodes = None
        self.combined_edges = None
        self.poly2 = Polyfit(self)
        self.polyBest = Polyfit(self)

class Polyfit:
    '''The child object of `Wing` containing the polynomial fit data

    Attributes:
        `wing` : Wing
            The inherited `Wing` parent object
        `x_fit` : array
            The array containing the fitted x points
        `y_fit` : array
            The array containing the fitted y points
        `degree` : int
            The highest degree of the polynomial
        `coefficients` : array
            The array containing the coefficients of the polynomial fit function
        `weights` : array
            The array containing the weights used during the polynomial fitting
    '''
    def __init__(self, wing):
        self.wing = wing
        self.x_fit = None
        self.y_fit = None
        self.degree = None
        self.coefficients = None
        self.weights = None