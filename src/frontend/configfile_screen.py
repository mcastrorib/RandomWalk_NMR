from PyQt5 import QtCore, QtGui, QtWidgets 

class configfile_screen():
    def __init__(self, _parent, _widget):
        self.parent = _parent
        self.m_widget = _widget
        self.m_widget.setMinimumSize(QtCore.QSize(350, 350))

        # rwnmr config fields
        self.nameLineEdit = None
        self.walkersLineEdit = None
        self.placementBox = None
        self.rhoTypeBox = None
        self.rhoLineEdit = None
        self.D0LineEdit = None
        self.stepsLineEdit = None
        self.histLineEdit = None
        self.histSizeLineEdit = None
        self.ompBox = None
        self.ompLineEdit = None
        self.gpuBox = None
        self.saveInfoBox = None
        self.ompLineEdit = None
        self.saveBinImgBox = None
        self.rwFileLineEdit = None

        #uct config fields
        self.uctDirPathLineEdit = None
        self.filenameLineEdit = None
        self.firstIndexLineEdit = None
        self.digitsLineEdit = None
        self.extensionLineEdit = None
        self.slicesLineEdit = None
        self.resolutionLineEdit = None
        self.voxelDivisionsLineEdit = None 
        self.uctFileLineEdit = None

        # these are the app widgets connected to their slot methods
        self.titleLabel = QtWidgets.QLabel('--- CONFIG SETUP ---')
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.closeTab)
        self.open_tabs = []
        self.open_tabs_names = []

        # set the layouts
        mainLayout = QtWidgets.QVBoxLayout(self.m_widget)        
        mainLayout.addWidget(self.titleLabel)
        mainLayout.addWidget(self.tabs)

        # set top alignment
        mainLayout.setAlignment(QtCore.Qt.AlignTop)
        return
    
    def closeTab(self, index):
        currentQWidget = self.open_tabs[index]
        currentQWidget.deleteLater()
        self.tabs.removeTab(index)
        self.open_tabs.pop(index)
        self.open_tabs_names.pop(index)
        return

    def createNewTab(self, tabName):
        if(tabName not in self.open_tabs_names):   
            new_tab = QtWidgets.QWidget()
            scrollbar = QtWidgets.QScrollArea()
            scrollbar.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            scrollbar.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            scrollbar.setWidgetResizable(True)
            scrollbar.setWidget(new_tab)
            self.open_tabs.append(new_tab)
            self.open_tabs_names.append(tabName)
            self.tabs.addTab(scrollbar, tabName)

            if(tabName == 'rwnmr'):
                self.createNewRWConfigTab()
            elif(tabName == 'uct'):
                self.createNewUCTConfigTab()
            else:
                print("new tab")

            lastTabIndex = len(self.open_tabs) - 1
            self.tabs.setCurrentIndex(lastTabIndex)
        else:
            tab_index = self.open_tabs_names.index(tabName)
            self.tabs.setCurrentIndex(tab_index)
        return     
    
    def createNewRWConfigTab(self):
        boolOptions = ['true', 'false']

        # these are the app widgets connected to their slot methods
        titleLabel = QtWidgets.QLabel('--- RW CONFIGURATION ---')
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # RW config fields
        nameLabel = QtWidgets.QLabel('Simulation name:')
        self.nameLineEdit = QtWidgets.QLineEdit()
        self.nameLineEdit.setText('NMR_Simulation')        

        walkersHeaderLabel = QtWidgets.QLabel('-- RW Parameters --')
        walkersHeaderLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)       
    
        walkersLabel = QtWidgets.QLabel('Walkers:')
        self.walkersLineEdit = QtWidgets.QLineEdit()
        self.walkersLineEdit.setText('10000')

        placementLabel = QtWidgets.QLabel('Placement:')
        self.placementBox = QtWidgets.QComboBox()
        placementOptions = ['random', 'point', 'cubic']
        self.placementBox.addItems(placementOptions)

        stepsLabel = QtWidgets.QLabel('Steps/echo:')
        self.stepsLineEdit = QtWidgets.QLineEdit()
        self.stepsLineEdit.setText('1')

        seedLabel = QtWidgets.QLabel('RNG seed:')
        self.seedLineEdit = QtWidgets.QLineEdit()
        self.seedLineEdit.setText('0')
        seedLabel2 = QtWidgets.QLabel('(0 == random)')

        physicalsLabel = QtWidgets.QLabel('-- Physical parameters --')
        physicalsLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)       

        rhoLabel = QtWidgets.QLabel('Superficial relaxivity:')
        self.rhoTypeBox = QtWidgets.QComboBox()
        rhoTypeOptions = ['uniform', 'sigmoid']
        self.rhoTypeBox.addItems(rhoTypeOptions)
        self.rhoLineEdit = QtWidgets.QLineEdit()
        self.rhoLineEdit.setText('{0.0}')
        rhoUnitLabel = QtWidgets.QLabel('um/ms')

        D0Label = QtWidgets.QLabel('Fluid diffusion coeficient:')
        self.D0LineEdit = QtWidgets.QLineEdit()
        self.D0LineEdit.setText('0.0')
        D0UnitLabel = QtWidgets.QLabel('um²/ms')      

        histHeaderLabel = QtWidgets.QLabel('-- Collision histogram --')
        histHeaderLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)       
        histLabel = QtWidgets.QLabel('Histograms:')
        self.histLineEdit = QtWidgets.QLineEdit()
        self.histLineEdit.setText('1')
        histSizeLabel = QtWidgets.QLabel('Size:')
        self.histSizeLineEdit = QtWidgets.QLineEdit()
        self.histSizeLineEdit.setText('1024')
        
        
        performanceLabel = QtWidgets.QLabel('-- Performance boost --')
        performanceLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        ompLabel = QtWidgets.QLabel('Multithreads:')
        self.ompBox = QtWidgets.QComboBox()
        self.ompBox.addItems(boolOptions)        
        gpuLabel = QtWidgets.QLabel('GPU:')
        self.gpuBox = QtWidgets.QComboBox()
        self.gpuBox.addItems(boolOptions)


        savingsLabel = QtWidgets.QLabel('-- Savings --')
        savingsLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        saveInfoLabel = QtWidgets.QLabel("Save info:")
        self.saveInfoBox = QtWidgets.QComboBox()
        self.saveInfoBox.addItems(boolOptions) 

        saveBinImgLabel = QtWidgets.QLabel("Save binary image:")
        self.saveBinImgBox = QtWidgets.QComboBox()
        self.saveBinImgBox.addItems(boolOptions)

        fileLabel = QtWidgets.QLabel("-- Config file --")
        fileLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        self.rwFileLineEdit = QtWidgets.QLineEdit()
        self.rwFileLineEdit.setText("rwnmr")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.clicked.connect(self.saveRWConfig)

        # set the layouts
        mainLayout = QtWidgets.QVBoxLayout(self.open_tabs[self.open_tabs_names.index('rwnmr')])        
        mainLayout.addWidget(titleLabel)

        # set RW config layout
        nameLayout = QtWidgets.QHBoxLayout()
        nameLayout.addWidget(nameLabel)
        nameLayout.addWidget(self.nameLineEdit)
        
        walkersHeaderLayout = QtWidgets.QHBoxLayout()
        walkersHeaderLayout.addWidget(walkersHeaderLabel)

        walkersLayout = QtWidgets.QHBoxLayout()
        walkersLayout.addWidget(walkersLabel)
        walkersLayout.addWidget(self.walkersLineEdit)  
        walkersLayout.addWidget(placementLabel)
        walkersLayout.addWidget(self.placementBox)  

        stepsLayout = QtWidgets.QHBoxLayout()
        stepsLayout.addWidget(stepsLabel)
        stepsLayout.addWidget(self.stepsLineEdit)

        seedLayout = QtWidgets.QHBoxLayout()
        seedLayout.addWidget(seedLabel)
        seedLayout.addWidget(self.seedLineEdit)
        seedLayout.addWidget(seedLabel2)   

        physicalsLayout = QtWidgets.QHBoxLayout()
        physicalsLayout.addWidget(physicalsLabel)

        rhoLayout = QtWidgets.QHBoxLayout()
        rhoLayout.addWidget(rhoLabel) 
        rhoLayout.addWidget(self.rhoTypeBox) 
        rhoLayout.addWidget(self.rhoLineEdit) 
        rhoLayout.addWidget(rhoUnitLabel)

        D0Layout = QtWidgets.QHBoxLayout()
        D0Layout.addWidget(D0Label)
        D0Layout.addWidget(self.D0LineEdit)
        D0Layout.addWidget(D0UnitLabel)

        # --
        histHeaderLayout = QtWidgets.QHBoxLayout()
        histHeaderLayout.addWidget(histHeaderLabel)

        histLayout = QtWidgets.QHBoxLayout()
        histLayout.addWidget(histLabel)
        histLayout.addWidget(self.histLineEdit)
        histLayout.addWidget(histSizeLabel)
        histLayout.addWidget(self.histSizeLineEdit)

        performanceLayout = QtWidgets.QHBoxLayout()
        performanceLayout.addWidget(performanceLabel)
        ompLayout = QtWidgets.QHBoxLayout()
        ompLayout.addWidget(ompLabel)
        ompLayout.addWidget(self.ompBox)
        gpuLayout = QtWidgets.QHBoxLayout()
        gpuLayout.addWidget(gpuLabel)
        gpuLayout.addWidget(self.gpuBox)


        savingsLayout = QtWidgets.QHBoxLayout()
        savingsLayout.addWidget(savingsLabel)
        saveInfoLayout = QtWidgets.QHBoxLayout()
        saveInfoLayout.addWidget(saveInfoLabel)
        saveInfoLayout.addWidget(self.saveInfoBox)
        saveBinImgLayout = QtWidgets.QHBoxLayout()
        saveBinImgLayout.addWidget(saveBinImgLabel)
        saveBinImgLayout.addWidget(self.saveBinImgBox)
           
        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayout = QtWidgets.QHBoxLayout()
        fileLayout.addWidget(fileNameLabel)
        fileLayout.addWidget(self.rwFileLineEdit)
        fileLayout.addWidget(fileExtensionLabel)        
        fileLayout.addWidget(saveButton)
        

        # adding layouts to main
        mainLayout.addLayout(nameLayout)
        
        mainLayout.addLayout(walkersHeaderLayout)    
        mainLayout.addLayout(walkersLayout)
        mainLayout.addLayout(stepsLayout)
        mainLayout.addLayout(seedLayout)
        
        mainLayout.addLayout(physicalsLayout)
        mainLayout.addLayout(rhoLayout)
        mainLayout.addLayout(D0Layout)
        
        mainLayout.addLayout(histHeaderLayout)
        mainLayout.addLayout(histLayout)

        mainLayout.addLayout(performanceLayout)
        mainLayout.addLayout(ompLayout)
        mainLayout.addLayout(gpuLayout)
        
        mainLayout.addLayout(savingsLayout)
        mainLayout.addLayout(saveInfoLayout)
        mainLayout.addLayout(saveBinImgLayout)
        
        mainLayout.addLayout(fileHeaderLayout)    
        mainLayout.addLayout(fileLayout)           
        
        mainLayout.setAlignment(QtCore.Qt.AlignTop)  
        return

    # @Slot()
    def saveRWConfig(self):
        filename = self.rwFileLineEdit.text() + ".config"
        print("saving file", filename)
        return
    
    def createNewUCTConfigTab(self):
        print('create uct tab')
        boolOptions = ['true', 'false']

        # these are the app widgets connected to their slot methods
        titleLabel = QtWidgets.QLabel('--- uCT IMAGE CONFIGURATION ---')
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # RW config fields
        dirPathLabel = QtWidgets.QLabel('Directory:')
        self.uctDirPathLineEdit = QtWidgets.QLineEdit()
        self.uctDirPathLineEdit.setText('')
        path_button = QtWidgets.QPushButton('Browse')
        path_button.clicked.connect(self.getUCTDirPath)

        filenameLabel = QtWidgets.QLabel('File name:')
        self.filenameLineEdit = QtWidgets.QLineEdit('')

        extensionLabel = QtWidgets.QLabel('Extension:')
        self.extensionLineEdit = QtWidgets.QLineEdit('.png')
        self.extensionLineEdit.setFixedWidth(50)
        
        firstIndexLabel = QtWidgets.QLabel('First index:')
        firstIndexLabel.setFixedWidth(80)
        self.firstIndexLineEdit = QtWidgets.QLineEdit('0')
        self.firstIndexLineEdit.setFixedWidth(40)

        digitsLabel = QtWidgets.QLabel('Digits:')
        digitsLabel.setFixedWidth(45)
        self.digitsLineEdit = QtWidgets.QLineEdit('1')
        self.digitsLineEdit.setFixedWidth(40)        

        slicesLabel = QtWidgets.QLabel('Slices:')
        slicesLabel.setFixedWidth(45)
        self.slicesLineEdit = QtWidgets.QLineEdit('1')
        self.slicesLineEdit.setFixedWidth(40)   

        resolutionLabel = QtWidgets.QLabel('Resolution:')
        resolutionLabel.setFixedWidth(80)
        self.resolutionLineEdit = QtWidgets.QLineEdit('1.0')
        self.resolutionLineEdit.setFixedWidth(40)
        resolutionUnitLabel = QtWidgets.QLabel('um/voxel')
        resolutionUnitLabel.setFixedWidth(80)

        voxelDivisionsLabel = QtWidgets.QLabel('Voxel divisions:')
        voxelDivisionsLabel.setFixedWidth(110)
        self.voxelDivisionsLineEdit = QtWidgets.QLineEdit('0')
        self.voxelDivisionsLineEdit.setFixedWidth(40)

        fileLabel = QtWidgets.QLabel("-- Config file --")
        fileLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        self.uctFileLineEdit = QtWidgets.QLineEdit()
        self.uctFileLineEdit.setText("uct")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.clicked.connect(self.saveUCTConfig)


        # set layouts
        mainLayout = QtWidgets.QVBoxLayout(self.open_tabs[self.open_tabs_names.index('uct')])
        mainLayout.addWidget(titleLabel)

        # set other layouts
        dirPathLayout = QtWidgets.QHBoxLayout()
        dirPathLayout.addWidget(dirPathLabel)
        dirPathLayout.addWidget(self.uctDirPathLineEdit)
        dirPathLayout.addWidget(path_button)

        filenameLayout = QtWidgets.QHBoxLayout()
        filenameLayout.addWidget(filenameLabel)
        filenameLayout.addWidget(self.filenameLineEdit)
        filenameLayout.addWidget(extensionLabel)
        filenameLayout.addWidget(self.extensionLineEdit)

        ImgLayout = QtWidgets.QHBoxLayout()
        ImgLayout.addWidget(firstIndexLabel)
        ImgLayout.addWidget(self.firstIndexLineEdit)
        ImgLayout.addWidget(digitsLabel)
        ImgLayout.addWidget(self.digitsLineEdit)        
        ImgLayout.addWidget(slicesLabel)
        ImgLayout.addWidget(self.slicesLineEdit)

        resolutionLayout = QtWidgets.QHBoxLayout()
        resolutionLayout.addWidget(resolutionLabel)
        resolutionLayout.addWidget(self.resolutionLineEdit)
        resolutionLayout.addWidget(resolutionUnitLabel)

        voxelDivisionsLayout = QtWidgets.QHBoxLayout()
        voxelDivisionsLayout.addWidget(voxelDivisionsLabel)
        voxelDivisionsLayout.addWidget(self.voxelDivisionsLineEdit)

        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayout = QtWidgets.QHBoxLayout()
        fileLayout.addWidget(fileNameLabel)
        fileLayout.addWidget(self.uctFileLineEdit)
        fileLayout.addWidget(fileExtensionLabel)        
        fileLayout.addWidget(saveButton)
        

        # add to main layout
        mainLayout.addLayout(dirPathLayout)
        mainLayout.addLayout(filenameLayout)
        mainLayout.addLayout(ImgLayout)
        mainLayout.addLayout(resolutionLayout)
        mainLayout.addLayout(voxelDivisionsLayout)
        mainLayout.addLayout(fileHeaderLayout)
        mainLayout.addLayout(fileLayout)
        

        # top alignment 
        ImgLayout.setAlignment(QtCore.Qt.AlignLeft)
        resolutionLayout.setAlignment(QtCore.Qt.AlignLeft)
        voxelDivisionsLayout.setAlignment(QtCore.Qt.AlignLeft)
        mainLayout.setAlignment(QtCore.Qt.AlignTop)  

        return
    
    # @Slot()
    def getUCTDirPath(self):
        print("browse dirpath")
        fileDialog = QtWidgets.QFileDialog(self.parent, "Choose directory")
        fileDialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        fileDialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly)
        
        if not fileDialog.exec_():
            return
        filepath = fileDialog.selectedFiles()[0]
        self.uctDirPathLineEdit.setText(filepath)            
        return
    
    # @Slot()
    def saveUCTConfig(self):
        filename = self.uctFileLineEdit.text() + ".config"
        print("saving file", filename)
        return