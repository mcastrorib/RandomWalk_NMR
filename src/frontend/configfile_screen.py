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

        # procedure tabs
        self.procedures_names = []
        self.procedures_qwidgets = []
        self.procedure_tabs_paths = []       

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

    def createNewTab(self, tabName, path_index=-1):
        new_tab = QtWidgets.QWidget()
        scrollbar = QtWidgets.QScrollArea()
        scrollbar.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollbar.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scrollbar.setWidgetResizable(True)
        scrollbar.setWidget(new_tab)
        if(tabName == 'rwnmr' or tabName == 'uct'):
            if(tabName not in self.open_tabs_names):   
                
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
        else:
            self.procedures_names.append(tabName)
            self.procedure_tabs_paths.append(self.parent.m_setup_tab.m_setup.procedure_paths[path_index])
            self.open_tabs.append(new_tab)
            self.open_tabs_names.append(tabName)
            self.tabs.addTab(scrollbar, tabName)

            if(tabName == 'pfgse'):
                self.createNewPFGSEConfigTab(path_index)
            elif(tabName == 'cpmg'):
                self.createNewCPMGConfigTab(path_index)
            elif(tabName == 'ga'):
                self.createNewGAConfigTab()

            lastTabIndex = len(self.open_tabs) - 1
            self.tabs.setCurrentIndex(lastTabIndex)

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
        mainLayout = QtWidgets.QVBoxLayout(self.open_tabs[-1])        
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
        self.parent.m_setup_tab.m_setup.rwConfigLineEdit.setText(filename)
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
        mainLayout = QtWidgets.QVBoxLayout(self.open_tabs[-1])
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
        self.parent.m_setup_tab.m_setup.uctConfigLineEdit.setText(filename)
        return

    def createNewPFGSEConfigTab(self, _index):
        print('create pfgse tab')
        boolOptions = ['true', 'false']

        # these are the app widgets connected to their slot methods
        titleLabel = QtWidgets.QLabel('--- PFGSE CONFIGURATION ---')
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        header1 = QtWidgets.QLabel('-- Physical attributes')
        # header1.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        header2 = QtWidgets.QLabel('-- Gradient vector')
        # header2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        header3 = QtWidgets.QLabel('-- Time sequence')
        # header3.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        header4 = QtWidgets.QLabel('-- Threshold for D(t) recover')
        # header4.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        header5 = QtWidgets.QLabel('-- Savings')
        # header5.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        
        # PFGSE config fields
        # header 1
        D0Label = QtWidgets.QLabel('Fluid free diffusion coefficient:')
        D0LineEdit = QtWidgets.QLineEdit('2.5')
        D0UnitLabel = QtWidgets.QLabel('um²/ms')

        gyroLabel = QtWidgets.QLabel('Fluid gyromagnetic ratio:')
        gyroLineEdit = QtWidgets.QLineEdit('42.576')
        gyroUnitLabel = QtWidgets.QLabel('MHz/T')

        pulseWidthLabel = QtWidgets.QLabel('Pulse width:')
        pulseWidthLineEdit = QtWidgets.QLineEdit('0.1')
        pulseWidthUnitLabel = QtWidgets.QLabel('ms')

        # header 2
        maxGradientLabel = QtWidgets.QLabel('Max gradient: {')
        maxGradientXLineEdit = QtWidgets.QLineEdit('0.0')
        exLabel = QtWidgets.QLabel('i, ')
        maxGradientYLineEdit = QtWidgets.QLineEdit('0.0')
        eyLabel = QtWidgets.QLabel('j, ')
        maxGradientZLineEdit = QtWidgets.QLineEdit('0.0')
        ezLabel = QtWidgets.QLabel('k }')
        gradientUnitLabel = QtWidgets.QLabel('Gauss/cm')

        gradientSamplesLabel = QtWidgets.QLabel('Samples: ')
        gradientSamplesLineEdit = QtWidgets.QLineEdit('200')

        # header 3
        sequenceTypes = ['manual', 'linear', 'log']
        sequenceLabel = QtWidgets.QLabel('Type: ')
        sequenceBox = QtWidgets.QComboBox()
        sequenceBox.addItems(sequenceTypes)
        timeSamplesLabel = QtWidgets.QLabel('Samples: ')
        timeSamplesLineEdit = QtWidgets.QLineEdit('4')
        timeValuesLabel = QtWidgets.QLabel('Values: ')
        timeValuesLineEdit = QtWidgets.QLineEdit('{0.2, 0.5, 1.0, 2.0}')
        timeUnitLabel = QtWidgets.QLabel('ms')
        timeMinLabel = QtWidgets.QLabel('min: ')
        timeMinLineEdit = QtWidgets.QLineEdit('-1.0')
        timeMaxLabel = QtWidgets.QLabel('max: ')
        timeMaxLineEdit = QtWidgets.QLineEdit('1.0')
        scaleLabel = QtWidgets.QLabel('Apply scale factor: ')
        scaleBox = QtWidgets.QComboBox()
        scaleBox.addItems(boolOptions)
        inspectionLengthLabel = QtWidgets.QLabel('Inspection length: ')
        inspectionLengthLineEdit = QtWidgets.QLineEdit('5.0')
        inspectionLengthUnitLabel = QtWidgets.QLabel('um')

        # header 4
        thresholdTypes = ['none', 'samples', 'lhs', 'rhs']
        thresholdTypeLabel = QtWidgets.QLabel('type: ')
        thresholdTypeBox = QtWidgets.QComboBox()
        thresholdTypeBox.addItems(thresholdTypes)
        thresholdValueLabel = QtWidgets.QLabel('value: ')
        thresholdValueLineEdit = QtWidgets.QLineEdit('0')

        # header 5
        saveModeLabel = QtWidgets.QLabel("Save mode:")
        saveModeBox = QtWidgets.QComboBox()
        saveModeBox.addItems(boolOptions)
        savePFGSELabel = QtWidgets.QLabel("Save pfgse:")
        savePFGSEBox = QtWidgets.QComboBox()
        savePFGSEBox.addItems(boolOptions)
        saveCollisionsLabel = QtWidgets.QLabel("Save collisions:")
        saveCollisionsBox = QtWidgets.QComboBox()
        saveCollisionsBox.addItems(boolOptions)
        saveDecayLabel = QtWidgets.QLabel("Save decay:")
        saveDecayBox = QtWidgets.QComboBox()
        saveDecayBox.addItems(boolOptions)
        saveHistogramLabel = QtWidgets.QLabel("Save histogram:")
        saveHistogramBox = QtWidgets.QComboBox()
        saveHistogramBox.addItems(boolOptions)
        saveHistListLabel = QtWidgets.QLabel("Save histogram list:")    
        saveHistListBox = QtWidgets.QComboBox()
        saveHistListBox.addItems(boolOptions)
        
        # config file 
        fileLabel = QtWidgets.QLabel("-- Config file --")
        # fileLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        pfgseFileLineEdit = QtWidgets.QLineEdit()
        pfgseFileLineEdit.setText("pfgse")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.clicked.connect(lambda: self.savePFGSEConfig(_index))


        # set layouts
        mainLayout = QtWidgets.QVBoxLayout(self.open_tabs[-1])
        mainLayout.addWidget(titleLabel)

        # set other layouts
        # header 1
        layoutV1 = QtWidgets.QVBoxLayout()
        layoutV1.addWidget(header1)
        
        layoutV1H1 = QtWidgets.QHBoxLayout()
        layoutV1H1.addWidget(D0Label)
        layoutV1H1.addWidget(D0LineEdit)
        layoutV1H1.addWidget(D0UnitLabel)
        layoutV1.addLayout(layoutV1H1)
        
        layoutV1H2 = QtWidgets.QHBoxLayout()
        layoutV1H2.addWidget(gyroLabel)
        layoutV1H2.addWidget(gyroLineEdit)
        layoutV1H2.addWidget(gyroUnitLabel)
        layoutV1.addLayout(layoutV1H2)

        layoutV1H3 = QtWidgets.QHBoxLayout()
        layoutV1H3.addWidget(pulseWidthLabel)
        layoutV1H3.addWidget(pulseWidthLineEdit)
        layoutV1H3.addWidget(pulseWidthUnitLabel)
        layoutV1.addLayout(layoutV1H3)

        layoutV1.addWidget(QtWidgets.QLabel(''))

        # header 2
        layoutV2 = QtWidgets.QVBoxLayout()
        layoutV2.addWidget(header2)
        
        layoutV2H1 = QtWidgets.QHBoxLayout()
        layoutV2H1.addWidget(maxGradientLabel)
        layoutV2H1.addWidget(maxGradientXLineEdit)
        layoutV2H1.addWidget(exLabel)
        layoutV2H1.addWidget(maxGradientYLineEdit)
        layoutV2H1.addWidget(eyLabel)
        layoutV2H1.addWidget(maxGradientZLineEdit)
        layoutV2H1.addWidget(ezLabel)
        layoutV2H1.addWidget(gradientUnitLabel)
        layoutV2.addLayout(layoutV2H1)

        layoutV2H2 = QtWidgets.QHBoxLayout()
        layoutV2H2.addWidget(gradientSamplesLabel)
        layoutV2H2.addWidget(gradientSamplesLineEdit)
        layoutV2.addLayout(layoutV2H2)

        layoutV2.addWidget(QtWidgets.QLabel(''))

        # header 3
        layoutV3 = QtWidgets.QVBoxLayout()
        layoutV3.addWidget(header3)
        
        layoutV3H1 = QtWidgets.QHBoxLayout()
        layoutV3H1.addWidget(sequenceLabel)
        layoutV3H1.addWidget(sequenceBox)
        layoutV3H1.addWidget(timeSamplesLabel)
        layoutV3H1.addWidget(timeSamplesLineEdit)
        layoutV3.addLayout(layoutV3H1)

        layoutV3H3 = QtWidgets.QHBoxLayout()
        layoutV3H3.addWidget(timeValuesLabel)
        layoutV3H3.addWidget(timeValuesLineEdit)
        layoutV3.addLayout(layoutV3H3)

        layoutV3H4 = QtWidgets.QHBoxLayout()
        layoutV3H4.addWidget(timeMinLabel)
        layoutV3H4.addWidget(timeMinLineEdit)
        layoutV3H4.addWidget(timeMaxLabel)
        layoutV3H4.addWidget(timeMaxLineEdit)
        layoutV3.addLayout(layoutV3H4)

        layoutV3H5 = QtWidgets.QHBoxLayout()
        layoutV3H5.addWidget(scaleLabel)
        layoutV3H5.addWidget(scaleBox)
        layoutV3H5.addWidget(inspectionLengthLabel)
        layoutV3H5.addWidget(inspectionLengthLineEdit)
        layoutV3H5.addWidget(inspectionLengthUnitLabel)        
        layoutV3.addLayout(layoutV3H5) 

        layoutV3.addWidget(QtWidgets.QLabel(''))

        # header 4
        layoutV4 = QtWidgets.QVBoxLayout()
        layoutV4.addWidget(header4)

        layoutV4H1 = QtWidgets.QHBoxLayout()
        layoutV4H1.addWidget(thresholdTypeLabel)
        layoutV4H1.addWidget(thresholdTypeBox)
        layoutV4H1.addWidget(thresholdValueLabel)
        layoutV4H1.addWidget(thresholdValueLineEdit)
        layoutV4.addLayout(layoutV4H1)

        layoutV4.addWidget(QtWidgets.QLabel(''))

        # header 5
        layoutV5 = QtWidgets.QVBoxLayout()
        layoutV5.addWidget(header5)

        layoutV5H1 = QtWidgets.QHBoxLayout()
        layoutV5H1.addWidget(saveModeLabel)
        layoutV5H1.addWidget(saveModeBox)
        layoutV5.addLayout(layoutV5H1)

        layoutV5H2 = QtWidgets.QHBoxLayout()
        layoutV5H2.addWidget(savePFGSELabel)
        layoutV5H2.addWidget(savePFGSEBox)
        layoutV5.addLayout(layoutV5H2)

        layoutV5H3 = QtWidgets.QHBoxLayout()
        layoutV5H3.addWidget(saveCollisionsLabel)
        layoutV5H3.addWidget(saveCollisionsBox)
        layoutV5.addLayout(layoutV5H3)
        
        layoutV5H4 = QtWidgets.QHBoxLayout()
        layoutV5H4.addWidget(saveDecayLabel)
        layoutV5H4.addWidget(saveDecayBox)
        layoutV5.addLayout(layoutV5H4)

        layoutV5H5 = QtWidgets.QHBoxLayout()
        layoutV5H5.addWidget(saveHistogramLabel)
        layoutV5H5.addWidget(saveHistogramBox)
        layoutV5.addLayout(layoutV5H5)

        layoutV5H6 = QtWidgets.QHBoxLayout()
        layoutV5H6.addWidget(saveHistListLabel)
        layoutV5H6.addWidget(saveHistListBox)
        layoutV5.addLayout(layoutV5H6)       
        
        layoutV5.addWidget(QtWidgets.QLabel(''))

        # file configs
        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayout = QtWidgets.QHBoxLayout()
        fileLayout.addWidget(fileNameLabel)
        fileLayout.addWidget(pfgseFileLineEdit)
        fileLayout.addWidget(fileExtensionLabel)        
        fileLayout.addWidget(saveButton)
        

        # add to main layout
        mainLayout.addLayout(layoutV1)
        mainLayout.addLayout(layoutV2)
        mainLayout.addLayout(layoutV3)
        mainLayout.addLayout(layoutV4)
        mainLayout.addLayout(layoutV5)
        mainLayout.addLayout(fileHeaderLayout)
        mainLayout.addLayout(fileLayout)
        

        # top alignment 
        mainLayout.setAlignment(QtCore.Qt.AlignTop)  

        # data wraping
        widgets = {}
        widgets["D0"] = D0LineEdit
        widgets["gyro"] = gyroLineEdit
        widgets["pulseWidth"] = pulseWidthLineEdit
        widgets["maxGradientX"] = maxGradientXLineEdit
        widgets["maxGradientY"] = maxGradientYLineEdit
        widgets["maxGradientZ"] = maxGradientZLineEdit
        widgets["gradientSamples"] = gradientSamplesLineEdit
        widgets["timeSequence"] = sequenceBox
        widgets["timeSamples"] = timeSamplesLineEdit
        widgets["timeValues"] = timeValuesLineEdit
        widgets["timeMin"] = timeMinLineEdit
        widgets["timeMax"] = timeMaxLineEdit
        widgets["scaleFactor"] = scaleBox
        widgets["inspectionLength"] = inspectionLengthLineEdit
        widgets["thresholdType"] = thresholdTypeBox 
        widgets["thresholdValue"] = thresholdValueLineEdit
        widgets["saveMode"] = saveModeBox
        widgets["savePFGSE"] = savePFGSEBox
        widgets["saveCollisions"] = saveCollisionsBox
        widgets["saveDecay"] = saveDecayBox
        widgets["saveHistogram"] = saveHistogramBox
        widgets["saveHistList"] = saveHistListBox
        widgets["configFilename"] = pfgseFileLineEdit          
        widgets["saveButton"] = saveButton

        self.procedures_qwidgets.append(widgets)        
        return
        
    # @Slot()
    def savePFGSEConfig(self, _index):
        filename = self.procedures_qwidgets[_index]["configFilename"].text() + ".config"
        self.parent.m_setup_tab.m_setup.procedure_paths[_index].setText(filename)
        return
    
    def createNewCPMGConfigTab(self, _index):
        boolOptions = ['true', 'false']

        # these are the app widgets connected to their slot methods
        titleLabel = QtWidgets.QLabel('--- PFGSE CONFIGURATION ---')
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        header1 = QtWidgets.QLabel('-- Physical attributes')
        # header1.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        header2 = QtWidgets.QLabel('-- Laplace inversion parameters')
        # header2.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        header3 = QtWidgets.QLabel('-- Savings')
        # header3.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        
        # PFGSE config fields
        # header 1
        D0Label = QtWidgets.QLabel('Fluid free diffusion coefficient:')
        D0LineEdit = QtWidgets.QLineEdit('2.5')
        D0UnitLabel = QtWidgets.QLabel('um²/ms')

        expTimeLabel = QtWidgets.QLabel('Experiment time:')
        expTimeLineEdit = QtWidgets.QLineEdit('2000.0')
        expTimeUnitLabel = QtWidgets.QLabel('ms')

        methodLabel = QtWidgets.QLabel('Pulse width:')
        methodBox = QtWidgets.QComboBox()
        methodBox.addItems(['image-based','histogram'])

        # header 2
        minT2Label = QtWidgets.QLabel('Min T2:')
        minT2LineEdit = QtWidgets.QLineEdit('0.1')
        maxT2Label = QtWidgets.QLabel('Max T2:')
        maxT2LineEdit = QtWidgets.QLineEdit('10000')
        T2binsLabel = QtWidgets.QLabel("T2 bins:")
        T2binsLineEdit = QtWidgets.QLineEdit('128')        
        logspaceLabel = QtWidgets.QLabel('Use logspace:')
        logspaceBox = QtWidgets.QComboBox()
        logspaceBox.addItems(boolOptions)
        minLambdaLabel = QtWidgets.QLabel('Min lambda:')
        minLambdaLineEdit = QtWidgets.QLineEdit('-4.0')
        maxLambdaLabel = QtWidgets.QLabel('Max lambda:')
        maxLambdaLineEdit = QtWidgets.QLineEdit('2.0')
        numLambdasLabel = QtWidgets.QLabel('Lambdas:')
        numLambdasLineEdit = QtWidgets.QLineEdit('512')
        pruneNumLabel = QtWidgets.QLabel('Prunes:')
        pruneNumLineEdit = QtWidgets.QLineEdit('512')
        noiseAmpLabel = QtWidgets.QLabel('Noise amplitude:')
        noiseAmpLineEdit = QtWidgets.QLineEdit('0.0')   

        # header 3
        saveModeLabel = QtWidgets.QLabel("Save mode:")
        saveModeBox = QtWidgets.QComboBox()
        saveModeBox.addItems(boolOptions)
        saveT2Label = QtWidgets.QLabel("Save T2:")
        saveT2Box = QtWidgets.QComboBox()
        saveT2Box.addItems(boolOptions)
        saveCollisionsLabel = QtWidgets.QLabel("Save collisions:")
        saveCollisionsBox = QtWidgets.QComboBox()
        saveCollisionsBox.addItems(boolOptions)
        saveDecayLabel = QtWidgets.QLabel("Save decay:")
        saveDecayBox = QtWidgets.QComboBox()
        saveDecayBox.addItems(boolOptions)
        saveHistogramLabel = QtWidgets.QLabel("Save histogram:")
        saveHistogramBox = QtWidgets.QComboBox()
        saveHistogramBox.addItems(boolOptions)
        saveHistListLabel = QtWidgets.QLabel("Save histogram list:")    
        saveHistListBox = QtWidgets.QComboBox()
        saveHistListBox.addItems(boolOptions)
        
        # config file 
        fileLabel = QtWidgets.QLabel("-- Config file --")
        # fileLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        pfgseFileLineEdit = QtWidgets.QLineEdit()
        pfgseFileLineEdit.setText("cpmg")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.clicked.connect(lambda: self.saveCPMGConfig(_index))


        # set layouts
        mainLayout = QtWidgets.QVBoxLayout(self.open_tabs[-1])
        mainLayout.addWidget(titleLabel)

        # set other layouts
        # header 1
        layoutV1 = QtWidgets.QVBoxLayout()
        layoutV1.addWidget(header1)
        
        layoutV1H1 = QtWidgets.QHBoxLayout()
        layoutV1H1.addWidget(D0Label)
        layoutV1H1.addWidget(D0LineEdit)
        layoutV1H1.addWidget(D0UnitLabel)
        layoutV1.addLayout(layoutV1H1)
        
        layoutV1H2 = QtWidgets.QHBoxLayout()
        layoutV1H2.addWidget(expTimeLabel)
        layoutV1H2.addWidget(expTimeLineEdit)
        layoutV1H2.addWidget(expTimeUnitLabel)
        layoutV1.addLayout(layoutV1H2)

        layoutV1H3 = QtWidgets.QHBoxLayout()
        layoutV1H3.addWidget(methodLabel)
        layoutV1H3.addWidget(methodBox)
        layoutV1.addLayout(layoutV1H3)

        layoutV1.addWidget(QtWidgets.QLabel(''))

        # header 2
        layoutV2 = QtWidgets.QVBoxLayout()
        layoutV2.addWidget(header2)
        
        layoutV2H1 = QtWidgets.QHBoxLayout()
        layoutV2H1.addWidget(minT2Label)
        layoutV2H1.addWidget(minT2LineEdit)
        layoutV2H1.addWidget(maxT2Label)
        layoutV2H1.addWidget(maxT2LineEdit)
        layoutV2.addLayout(layoutV2H1)
        
        layoutV2H2 = QtWidgets.QHBoxLayout()
        layoutV2H2.addWidget(T2binsLabel)
        layoutV2H2.addWidget(T2binsLineEdit)
        layoutV2.addLayout(layoutV2H2)

        layoutV2H3 = QtWidgets.QHBoxLayout()
        layoutV2H3.addWidget(logspaceLabel)
        layoutV2H3.addWidget(logspaceBox)
        layoutV2.addLayout(layoutV2H3)

        layoutV2H4 = QtWidgets.QHBoxLayout()
        layoutV2H4.addWidget(minLambdaLabel)
        layoutV2H4.addWidget(minLambdaLineEdit)
        layoutV2H4.addWidget(maxLambdaLabel)
        layoutV2H4.addWidget(maxLambdaLineEdit)
        layoutV2.addLayout(layoutV2H4)

        layoutV2H5 = QtWidgets.QHBoxLayout()
        layoutV2H5.addWidget(numLambdasLabel)
        layoutV2H5.addWidget(numLambdasLineEdit)
        layoutV2.addLayout(layoutV2H5)

        layoutV2H6 = QtWidgets.QHBoxLayout()
        layoutV2H6.addWidget(pruneNumLabel)
        layoutV2H6.addWidget(pruneNumLineEdit)
        layoutV2.addLayout(layoutV2H6)

        layoutV2H7 = QtWidgets.QHBoxLayout()
        layoutV2H7.addWidget(noiseAmpLabel)
        layoutV2H7.addWidget(noiseAmpLineEdit)
        layoutV2.addLayout(layoutV2H7)

        layoutV2.addWidget(QtWidgets.QLabel(''))

        # header 3
        layoutV3 = QtWidgets.QVBoxLayout()
        layoutV3.addWidget(header3)

        layoutV3H1 = QtWidgets.QHBoxLayout()
        layoutV3H1.addWidget(saveModeLabel)
        layoutV3H1.addWidget(saveModeBox)
        layoutV3.addLayout(layoutV3H1)

        layoutV3H2 = QtWidgets.QHBoxLayout()
        layoutV3H2.addWidget(saveT2Label)
        layoutV3H2.addWidget(saveT2Box)
        layoutV3.addLayout(layoutV3H2)

        layoutV3H3 = QtWidgets.QHBoxLayout()
        layoutV3H3.addWidget(saveCollisionsLabel)
        layoutV3H3.addWidget(saveCollisionsBox)
        layoutV3.addLayout(layoutV3H3)
        
        layoutV3H4 = QtWidgets.QHBoxLayout()
        layoutV3H4.addWidget(saveDecayLabel)
        layoutV3H4.addWidget(saveDecayBox)
        layoutV3.addLayout(layoutV3H4)

        layoutV3H5 = QtWidgets.QHBoxLayout()
        layoutV3H5.addWidget(saveHistogramLabel)
        layoutV3H5.addWidget(saveHistogramBox)
        layoutV3.addLayout(layoutV3H5)

        layoutV3H6 = QtWidgets.QHBoxLayout()
        layoutV3H6.addWidget(saveHistListLabel)
        layoutV3H6.addWidget(saveHistListBox)
        layoutV3.addLayout(layoutV3H6)       
        
        layoutV3.addWidget(QtWidgets.QLabel(''))

        # file configs
        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayout = QtWidgets.QHBoxLayout()
        fileLayout.addWidget(fileNameLabel)
        fileLayout.addWidget(pfgseFileLineEdit)
        fileLayout.addWidget(fileExtensionLabel)        
        fileLayout.addWidget(saveButton)
        

        # add to main layout
        mainLayout.addLayout(layoutV1)
        mainLayout.addLayout(layoutV2)
        mainLayout.addLayout(layoutV3)
        mainLayout.addLayout(fileHeaderLayout)
        mainLayout.addLayout(fileLayout)
        

        # top alignment 
        mainLayout.setAlignment(QtCore.Qt.AlignTop)  

        # data wraping
        widgets = {}
        widgets["D0"] = D0LineEdit
        widgets["expTime"] = expTimeLineEdit
        widgets["method"] = methodBox

        widgets["minT2"] = minT2LineEdit
        widgets["maxT2"] = maxT2LineEdit
        widgets["T2bins"] = T2binsLineEdit
        widgets["logspace"] = logspaceBox
        widgets["minLambda"] = minLambdaLineEdit
        widgets["maxLambda"] = maxLambdaLineEdit
        widgets["numLambdas"] = numLambdasLineEdit
        widgets["pruneNum"] = pruneNumLineEdit
        widgets["noiseAmp"] = noiseAmpLineEdit

        widgets["saveMode"] = saveModeBox
        widgets["saveT2"] = saveT2Box
        widgets["saveCollisions"] = saveCollisionsBox
        widgets["saveDecay"] = saveDecayBox
        widgets["saveHistogram"] = saveHistogramBox
        widgets["saveHistList"] = saveHistListBox
        widgets["configFilename"] = pfgseFileLineEdit          
        widgets["saveButton"] = saveButton

        self.procedures_qwidgets.append(widgets)        
        return
        
    # @Slot()
    def saveCPMGConfig(self, _index):
        filename = self.procedures_qwidgets[_index]["configFilename"].text() + ".config"
        self.parent.m_setup_tab.m_setup.procedure_paths[_index].setText(filename)
        return

    def createNewGAConfigTab(self):
        return
    
    # @Slot()
    def getGADirPath(self):
        return
    
    # @Slot()
    def saveGAConfig(self):
        return