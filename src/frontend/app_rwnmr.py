import os
import sys
import json
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets 
from scipy import ndimage
from setup_tab import setup_tab
from image_viewer import image_viewer
from setup_screen import setup_screen
from configfile_screen import configfile_screen

# Inherit from QDialog
class app_rwnmr(QtWidgets.QMainWindow):
    # Override the class constructor
    def __init__(self, app_name='', parent=None):
        super(app_rwnmr, self).__init__(parent)

        # Set app major tabs objects as None
        self.m_setup_tab = None
        self.m_datavis_tab = None

        # Set app title 
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle(app_name) 
        
        # Set geometry and minimum size
        self.setGeometry(100, 100, 1024, 900) 
        self.setMinimumSize(QtCore.QSize(1024, 860))
        
        # Set app main QWidget 
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)   

        # Set app major toolbar
        self.toolbar = self.addToolBar("File")

        # Open 'setup' tab button
        self.setup_button = QtWidgets.QAction(QtGui.QIcon("icons/setup"), "setup", self)
        self.toolbar.addAction(self.setup_button)

        # Open 'dataviz' tab button
        self.dataviz_button = QtWidgets.QAction(QtGui.QIcon("icons/dataviz"), "visualization", self)
        self.toolbar.addAction(self.dataviz_button)

        # Trigger action from buttons in ToolBar
        self.toolbar.actionTriggered[QtWidgets.QAction].connect(self.tbpressed)

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.closeTab)
        self.open_tabs = []
        self.open_tabs_names = []

        # Set the tab layouts
        layout = QtWidgets.QVBoxLayout(self.central_widget)  
        layout.addWidget(self.tabs) 

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
        return

    # @Slot()
    def closeTab(self, index):
        currentQWidget = self.open_tabs[index]
        currentQWidget.deleteLater()
        self.tabs.removeTab(index)
        self.open_tabs.pop(index)
        name = self.open_tabs_names[index]
        self.open_tabs_names.pop(index)
        
        if(name == "setup"):
            self.m_setup_tab = None
        elif(name == "dataviz"):
            self.m_datavis_tab = None
        
        return
   
    
    # @Slot()
    def fileQuit(self):
        if (self.m_setup_tab != None and self.m_setup_tab.m_viewer != None and len(self.m_setup_tab.m_viewer.m_map) > 0):
            self.m_setup_tab.m_viewer.removeTempImages()
        self.close()
        return

    # @Slot()
    def tbpressed(self, a):
        if a.text() == "setup":
            self.createNewTab('setup')
        elif a.text() == "visualization":
            self.createNewTab("dataviz")
        return


    # @Slot()
    def aboutDlg(self):
        sm = """RWNMR\nVersion 1.0.0\n2020\nLicense GPL 3.0\n\nThe authors and the involved Institutions are not responsible for the use or bad use of the program and their results. The authors have no legal dulty or responsability for any person or company for the direct or indirect damage caused resulting from the use of any information or usage of the program available here. The user is responsible for all and any conclusion made with the program. There is no warranty for the program use. """
        msg = QtWidgets.QMessageBox()
        msg.setText(sm)
        msg.setWindowTitle("About")
        msg.exec_()
        return
    
    # @Slot()
    def openImage(self):
        if(self.m_setup_tab != None):
            if(len(self.m_setup_tab.m_viewer.m_map) > 0):
                self.m_setup_tab.m_viewer.clear()        
            self.m_setup_tab.m_viewer.openImage()
        else:
            self.createNewTab("setup")
            self.m_setup_tab.m_viewer.openImage()
        return
    
    # Class methods
    def addConfigTab(self, tabName, index=-1):
        self.m_setup_tab.m_config.createNewTab(tabName, index)
        return

    def createNewTab(self, tabName):
        if(tabName not in self.open_tabs_names):   
            new_tab = QtWidgets.QWidget()
            self.open_tabs.append(new_tab)
            self.open_tabs_names.append(tabName)
            self.tabs.addTab(new_tab, tabName)

            if(tabName == 'setup'):
                self.createSetupTab()
            elif(tabName == 'dataviz'):
                self.createDataVizTab()

            lastTabIndex = len(self.open_tabs) - 1
            self.tabs.setCurrentIndex(lastTabIndex)
        else:
            tab_index = self.open_tabs_names.index(tabName)
            self.tabs.setCurrentIndex(tab_index)
        return  
    
    def createDataVizTab(self):
        print("creating data viz tab")
        return

    def createSetupTab(self):
        if(self.m_setup_tab == None):
            self.m_setup_tab = setup_tab(self, self.open_tabs[self.open_tabs_names.index('setup')])
        return