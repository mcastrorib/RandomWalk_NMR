"""
A simple GUI that allows to load and visualize a stack of images (slices)
from micro-computed tomography and run random-walk based NMR simulations.
Basic current capabilities:

"""
import os
import sys
import json
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets 
from scipy import ndimage
from app_rwnmr import app_rwnmr

ORGANIZATION_NAME = 'LCC-IC-UFF'
ORGANIZATION_DOMAIN = ''
APPLICATION_NAME = 'RWNMR'
SETTINGS_TRAY = 'settings/tray'

def main():
    # To ensure that every time you call QSettings not enter the data of your application, 
    # which will be the settings, you can set them globally for all applications   
    QtCore.QCoreApplication.setApplicationName(ORGANIZATION_NAME)
    QtCore.QCoreApplication.setOrganizationDomain(ORGANIZATION_DOMAIN)
    QtCore.QCoreApplication.setApplicationName(APPLICATION_NAME)

    # create pyqt5 app
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    # create the instance of our Window
    mw = app_rwnmr(APPLICATION_NAME)

    # showing all the widgets
    mw.show()

    # start the app
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()