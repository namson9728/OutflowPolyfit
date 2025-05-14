import astropy.io.fits as fits
import numpy as np
from astropy.stats import sigma_clipped_stats

class Protostar:
    def __init__(self, name, source_choord, data_directory_path):
        self.directory_path = data_directory_path
        self.name = name
        self.colors = ['red','blue']
        self.blue_shifted = Outflow(color='blue', protostar=self)
        self.red_shifted = Outflow(color='red', protostar=self)
        self.source_choord = source_choord

class Outflow:
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
        self.image_masked = None
        self.min_sigma = None
        self.max_sigma = None
        self.sigma_array = None
        self.edges = None
        self.top_wing = Wing(self, 'top')
        self.bot_wing = Wing(self, 'bot')
        self.angle_spectrum = None

class Wing:
    def __init__(self, outflow, loc):
        self.outflow = outflow
        self.loc = loc
        self.edges = None
        self.nodes = None
        self.combined_edges = None
        self.poly2 = Polyfit(self)
        self.polyBest = Polyfit(self)

class Polyfit:
    def __init__(self, wing):
        self.wing = wing
        self.x_fit = None
        self.y_fit = None
        self.order = None
        self.coefficients = None
        self.weights = None