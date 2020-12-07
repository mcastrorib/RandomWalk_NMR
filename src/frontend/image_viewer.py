import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets 
from scipy import ndimage

class image_viewer():
    def __init__(self, _parent, _widget):
        self.parent = _parent
        self.m_widget = _widget
        self.filepaths = None

        self.m_widget.setMinimumSize(QtCore.QSize(350, 350))

        # a figure instance to plot on
        self.figure = Figure(figsize=(3,4))

        # this is the Canvas widget that displays the 'figure'
        self.canvas = FigureCanvasQTAgg(self.figure)
        matplotlib.rcParams['image.cmap'] = 'gray' # magma, seismic

        # this is the Navigation widget - toolbar for image
        self.toolbar = NavigationToolbar2QT(self.canvas, self.parent)

        # these are the app widgets connected to their slot methods
        self.titleLabel = QtWidgets.QLabel('--- IMAGE VIEWER ---')
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)        
        self.slideBar = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slideBar.setMinimum(0)
        self.slideBar.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slideBar.setTickInterval(1)        
        self.slideBar.setSingleStep(1)
        self.slideBar.setEnabled(False)
        self.slideBar.valueChanged[int].connect(self.changeValue)
        self.buttonPlus = QtWidgets.QPushButton('+')
        self.buttonPlus.setMaximumSize(QtCore.QSize(25, 30))
        self.buttonPlus.setEnabled(False)
        self.buttonPlus.clicked.connect(self.slideMoveUp)
        self.buttonMinus = QtWidgets.QPushButton('-')
        self.buttonMinus.setMaximumSize(QtCore.QSize(25, 30))
        self.buttonMinus.setEnabled(False) 
        self.buttonMinus.clicked.connect(self.slideMoveDown)        
        self.buttonLoad = QtWidgets.QPushButton('Open')
        self.buttonLoad.setMinimumSize(QtCore.QSize(50, 40))
        self.buttonLoad.setEnabled(True)
        self.buttonLoad.clicked.connect(self.openImage)       
        self.labelDimensions = QtWidgets.QLabel('[h=0,w=0]')
        self.labelSliceId = QtWidgets.QLabel('Slice = 0')
        self.labelSliceId.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # set the layouts
        mainLayout = QtWidgets.QVBoxLayout(self.m_widget)
        mainLayout.addWidget(self.titleLabel)
        mainLayout.addWidget(self.toolbar)
        layoutH2 = QtWidgets.QHBoxLayout()
        layoutH3 = QtWidgets.QHBoxLayout()
        layoutH2.addWidget(self.buttonMinus)        
        layoutH2.addWidget(self.slideBar)        
        layoutH2.addWidget(self.buttonPlus)  
        layoutH3.addWidget(self.labelDimensions)
        layoutH3.addItem(QtWidgets.QSpacerItem(15, 15, QtWidgets.QSizePolicy.MinimumExpanding))
        layoutH3.addWidget(self.buttonLoad)
        layoutH3.addItem(QtWidgets.QSpacerItem(15, 15, QtWidgets.QSizePolicy.MinimumExpanding))
        layoutH3.addWidget(self.labelSliceId)
        mainLayout.addWidget(self.canvas, QtWidgets.QSizePolicy.MinimumExpanding)
        mainLayout.addLayout(layoutH2)
        mainLayout.addLayout(layoutH3)   
        mainLayout.setAlignment(QtCore.Qt.AlignTop)   

        # initialize the main image data
        self.m_data = None # numpy array
        self.m_image = None # QImage object
        self.m_map = []  # path of all image files 
    
    def clear(self):
        self.__del__()

    def __del__(self):
        # remove temporary data: 
        self.m_data = None
        self.m_image = None
        if len(self.m_map) > 0:
            self.removeTempImages()

    # @Slot()
    def plotImage(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = ax.imshow(self.m_data,vmin=0,vmax=255)      
        self.figure.colorbar(img)
        ax.figure.canvas.draw()
        # self.buttonPlot.setEnabled(False)  

    # @Slot()
    def changeValue(self, _value):
        filename = self.m_map[_value]
        print(filename)
        self.loadImageData(filename,True)
        self.labelSliceId.setText("Slice = "+str(_value+1))      

    # @Slot()
    def slideMoveUp(self):
        self.slideBar.setValue(self.slideBar.value()+1)

    # @Slot()
    def slideMoveDown(self):
        self.slideBar.setValue(self.slideBar.value()-1)
    
    # @Slot()
    def openImage(self):
        options = QtWidgets.QFileDialog.Options()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self.parent, "Open uCT-image", "","Image Files (*.png);;Image Files (*.png)", options=options)
        if files:
            if len(self.m_map) > 0:
                self.removeTempImages()
            self.m_map.clear() # remove all items
            for filepath in files:
                self.m_map.append( filepath )
            self.loadImageData(files[0],True)
            self.buttonPlus.setEnabled(True) 
            self.buttonMinus.setEnabled(True) 
            self.slideBar.setMaximum(len(self.m_map)-1)
            self.slideBar.setValue(0)
            self.slideBar.setEnabled(True)
            self.labelSliceId.setText("Slice = 1")

            self.filepaths = files
            print(self.filepaths)

    # @Slot()
    def aboutDlg(self):
        sm = """pyTomoViewer\nVersion 1.0.0\n2020\nLicense GPL 3.0\n\nThe authors and the involved Institutions are not responsible for the use or bad use of the program and their results. The authors have no legal dulty or responsability for any person or company for the direct or indirect damage caused resulting from the use of any information or usage of the program available here. The user is responsible for all and any conclusion made with the program. There is no warranty for the program use. """
        msg = QtWidgets.QMessageBox()
        msg.setText(sm)
        msg.setWindowTitle("About")
        msg.exec_()
    
    # method
    def loadImageData(self, _filepath, _updateWindow):
        self.m_image = QtGui.QImage(_filepath)
        # We perform these conversions in order to deal with just 8 bits images:
        # convert Mono format to Indexed8
        if self.m_image.depth() == 1:
            self.m_image = self.m_image.convertToFormat(QtGui.QImage.Format_Indexed8)
        # convert Grayscale16 format to Grayscale8
        if not self.m_image.format() == QtGui.QImage.Format_Grayscale8:
            self.m_image = self.m_image.convertToFormat(QtGui.QImage.Format_Grayscale8)
        self.m_data = convertQImageToNumpy(self.m_image)
        if _updateWindow:
            self.labelDimensions.setText("[h="+str(self.m_data.shape[0])+",w="+str(self.m_data.shape[1])+"]")
            self.plotImage()
    
    # method
    def removeTempImages(self):
        for filepath in self.m_map:
            if "temp_" in filepath :
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except OSError as err:
                        print("Exception handled: {0}".format(err))
                else:
                    print("The file does not exist") 

# This function was adapted from (https://github.com/Entscheider/SeamEater/blob/master/gui/QtTool.py)
# Project: SeamEater; Author: Entscheider; File: QtTool.py; GNU General Public License v3.0 
# Original function name: qimage2numpy(qimg)
# We consider just 8 bits images and convert to single depth:
def convertQImageToNumpy(_qimg):
    h = _qimg.height()
    w = _qimg.width()
    ow = _qimg.bytesPerLine() * 8 // _qimg.depth()
    d = 0
    if _qimg.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
                        QtGui.QImage.Format_ARGB32,
                        QtGui.QImage.Format_RGB32):
        d = 4 # formats: 6, 5, 4.
    elif _qimg.format() in (QtGui.QImage.Format_Indexed8,
                            QtGui.QImage.Format_Grayscale8):
        d = 1 # formats: 3, 24.
    else:
        raise ValueError(".ERROR: Unsupported QImage format!")
    buf = _qimg.bits().asstring(_qimg.byteCount())
    res = np.frombuffer(buf, 'uint8')
    res = res.reshape((h,ow,d)).copy()
    if w != ow:
        res = res[:,:w] 
    if d >= 3:
        res = res[:,:,0].copy()
    else:
        res = res[:,:,0] 
    return res