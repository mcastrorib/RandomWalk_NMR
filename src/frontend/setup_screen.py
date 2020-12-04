from PyQt5 import QtCore, QtGui, QtWidgets 

class setup_screen():
    def __init__(self, _parent, _widget):
        self.parent = _parent
        self.m_widget = _widget

        # these are the app widgets connected to their slot methods
        self.titleLabel = QtWidgets.QLabel('--- RWNMR SETUP ---')
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # set RW config widgets
        self.rwConfigLabel = QtWidgets.QLabel('rw configuration file:')
        self.rwConfigLineEdit = QtWidgets.QLineEdit()
        self.rwConfigLineEdit.setText('default')
        self.rwConfigLineEdit.setEnabled(False)
        self.rwConfigOpenButton = QtWidgets.QPushButton('Open')
        self.rwConfigOpenButton.clicked.connect(self.getRWConfigPath)
        self.rwConfigCreateButton = QtWidgets.QPushButton('Create')
        self.rwConfigCreateButton.clicked.connect(self.createRWConfigFile)

        # set uCT config widgets
        self.uctConfigLabel = QtWidgets.QLabel('uct configuration file:')
        self.uctConfigLineEdit = QtWidgets.QLineEdit()
        self.uctConfigLineEdit.setText('default')
        self.uctConfigLineEdit.setEnabled(False)
        self.uctConfigOpenButton = QtWidgets.QPushButton('Open')
        self.uctConfigOpenButton.clicked.connect(self.getUCTConfigPath)
        self.uctConfigCreateButton = QtWidgets.QPushButton('Create')
        self.uctConfigCreateButton.clicked.connect(self.createUCTConfigFile)       
        

        # set the layouts
        mainLayout = QtWidgets.QVBoxLayout(self.m_widget)        
        mainLayout.addWidget(self.titleLabel)

        # set RW config layout
        layoutH2a = QtWidgets.QHBoxLayout()
        layoutH2a.addWidget(self.rwConfigLabel)
        layoutH2b = QtWidgets.QHBoxLayout()
        layoutH2b.addWidget(self.rwConfigLineEdit)
        layoutH2b.addWidget(self.rwConfigOpenButton)
        layoutH2b.addWidget(self.rwConfigCreateButton)        
        
        # set uCT config layout
        layoutH3a = QtWidgets.QHBoxLayout()
        layoutH3a.addWidget(self.uctConfigLabel)
        layoutH3b = QtWidgets.QHBoxLayout()
        layoutH3b.addWidget(self.uctConfigLineEdit)
        layoutH3b.addWidget(self.uctConfigOpenButton)
        layoutH3b.addWidget(self.uctConfigCreateButton)  

        # adding layouts to main
        mainLayout.addLayout(layoutH2a)
        mainLayout.addLayout(layoutH2b) 
        mainLayout.addLayout(layoutH3a)
        mainLayout.addLayout(layoutH3b)      
        mainLayout.setAlignment(QtCore.Qt.AlignTop)  

        # self.slideBar = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # self.slideBar.setMinimum(0)
        # self.slideBar.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        # self.slideBar.setTickInterval(1)        
        # self.slideBar.setSingleStep(1)
        # self.slideBar.setEnabled(False)
        # self.slideBar.valueChanged[int].connect(self.changeValue)
        # self.buttonPlus = QtWidgets.QPushButton('+')
        # self.buttonPlus.setMaximumSize(QtCore.QSize(25, 30))
        # self.buttonPlus.setEnabled(False)
        # self.buttonPlus.clicked.connect(self.slideMoveUp)
        # self.buttonMinus = QtWidgets.QPushButton('-')
        # self.buttonMinus.setMaximumSize(QtCore.QSize(25, 30))
        # self.buttonMinus.setEnabled(False) 
        # self.buttonMinus.clicked.connect(self.slideMoveDown)        
        # self.buttonLoad = QtWidgets.QPushButton('Open')
        # self.buttonLoad.setMinimumSize(QtCore.QSize(50, 40))
        # self.buttonLoad.setEnabled(True)
        # self.buttonLoad.clicked.connect(self.openImage)       
        # self.labelDimensions = QtWidgets.QLabel('[h=0,w=0]')
        # self.labelSliceId = QtWidgets.QLabel('Slice = 0')
        # self.labelSliceId.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return

    # @Slot()
    def getRWConfigPath(self):
        options = QtWidgets.QFileDialog.Options()
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent, "Open", "","Config Files (*.config);;Config Files (*.config)", options=options)
        if filepath != '':
            print(filepath)
            self.rwConfigLineEdit.setText(filepath)                    

        return

    # @Slot()
    def getUCTConfigPath(self):
        options = QtWidgets.QFileDialog.Options()
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent, "Open", "","Config Files (*.config);;Config Files (*.config)", options=options)
        if filepath != '':
            print(filepath)
            self.uctConfigLineEdit.setText(filepath)                    

        return

    def createRWConfigFile(self):
        self.parent.addConfigTab('rwnmr')
        return
    
    def createUCTConfigFile(self):
        self.parent.addConfigTab('uct')
        return