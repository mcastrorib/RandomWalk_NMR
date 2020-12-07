import os
import sys
import json
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets 
from scipy import ndimage
from image_viewer import image_viewer
from setup_screen import setup_screen
from configfile_screen import configfile_screen

class setup_tab():
    def __init__(self, _parent, _widget):
        self.parent = _parent
        self.m_widget = _widget

        self.m_setup = None
        self.m_viewer = None
        self.m_config = None
        
        # setting list of widgets
        self.active_widgets = []

        # setting two initial windows
        self.active_widgets.append(QtWidgets.QWidget())
        self.active_widgets.append(QtWidgets.QWidget())
        self.active_widgets.append(QtWidgets.QWidget())

        lay = QtWidgets.QGridLayout(self.m_widget)

        lay.addWidget(self.active_widgets[0], 0, 0)
        lay.addWidget(self.active_widgets[1], 0, 1)
        

        # lay = QtWidgets.QVBoxLayout(self.active_widgets[0])
        self.m_setup = setup_screen(self.parent, self.active_widgets[0])
        # lay.addWidget(QtWidgets.QTextEdit())

        lay = QtWidgets.QGridLayout(self.active_widgets[1])
        self.active_widgets.append(QtWidgets.QWidget())
        self.active_widgets.append(QtWidgets.QWidget())
        lay.addWidget(self.active_widgets[2], 0, 0)
        lay.addWidget(self.active_widgets[3], 1, 0)

        # setting config_creation object
        self.m_config = configfile_screen(self.parent, self.active_widgets[2])

        # setting image_viewer object
        self.m_viewer = image_viewer(self.parent, self.active_widgets[3])       

    
    # method
    def addConfigTab(self, tabName):
        self.m_config.createNewTab(tabName)
        return