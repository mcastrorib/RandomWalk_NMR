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

# Inherit from QDialog
class app_rwnmr(QtWidgets.QMainWindow):
    # Override the class constructor
    def __init__(self, app_name='', parent=None):
        super(app_rwnmr, self).__init__(parent)

        self.m_setup = None
        self.m_viewer = None
        self.m_config = None

        # setting title 
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(app_name) 
        
        # setting geometry and minimum size
        self.setGeometry(100, 100, 1024, 860) 
        self.setMinimumSize(QtCore.QSize(1024, 860))
        
        # setting main Widget 
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)    
        
        # setting list of widgets
        self.active_widgets = []

        # setting two initial windows
        self.active_widgets.append(QtWidgets.QWidget())
        self.active_widgets.append(QtWidgets.QWidget())
        self.active_widgets.append(QtWidgets.QWidget())

        lay = QtWidgets.QGridLayout(self.central_widget)

        lay.addWidget(self.active_widgets[0], 0, 0)
        lay.addWidget(self.active_widgets[1], 0, 1)
        

        # lay = QtWidgets.QVBoxLayout(self.active_widgets[0])
        self.m_setup = setup_screen(self, self.active_widgets[0])
        # lay.addWidget(QtWidgets.QTextEdit())

        lay = QtWidgets.QGridLayout(self.active_widgets[1])
        self.active_widgets.append(QtWidgets.QWidget())
        self.active_widgets.append(QtWidgets.QWidget())
        lay.addWidget(self.active_widgets[2], 0, 0)
        lay.addWidget(self.active_widgets[3], 1, 0)

        # lay = QtWidgets.QVBoxLayout(self.active_widgets[2])
        # lay.addWidget(QtWidgets.QTextEdit())

        # setting config_creation object
        self.m_config = configfile_screen(self, self.active_widgets[2])

        # setting image_viewer object
        self.m_viewer = image_viewer(self, self.active_widgets[3])

        # Set file menu  
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Open image', self.openImage,
                                QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        # Set help menu
        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.help_menu.addAction("About...", self.aboutDlg, QtCore.Qt.CTRL + QtCore.Qt.Key_H)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)


    
    def fileQuit(self):
        if (self.m_viewer != None and len(self.m_viewer.m_map) > 0):
            self.m_viewer.removeTempImages()
        self.close()


    # @Slot()
    def aboutDlg(self):
        sm = """pyTomoViewer\nVersion 1.0.0\n2020\nLicense GPL 3.0\n\nThe authors and the involved Institutions are not responsible for the use or bad use of the program and their results. The authors have no legal dulty or responsability for any person or company for the direct or indirect damage caused resulting from the use of any information or usage of the program available here. The user is responsible for all and any conclusion made with the program. There is no warranty for the program use. """
        msg = QtWidgets.QMessageBox()
        msg.setText(sm)
        msg.setWindowTitle("About")
        msg.exec_()
    
    # @Slot()
    def openImage(self):
        if(self.m_viewer != None):
            self.m_viewer.openImage()
        else:
            self.divideLastWidget()
            last_widget_id = len(self.active_widgets)
    
    # method
    def addConfigTab(self, tabName):
        self.m_config.createNewTab(tabName)