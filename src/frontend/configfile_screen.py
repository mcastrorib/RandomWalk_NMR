from PyQt5 import QtCore, QtGui, QtWidgets 

# this should not be used like this...
CONFIG_PATH = '/home/matheus/Documentos/doutorado_ic/tese/NMR/rwnmr_2.0/config/'

class configfile_screen():
    def __init__(self, _parent, _widget):
        self.parent = _parent
        self.m_widget = _widget
        self.m_widget.setMinimumSize(QtCore.QSize(350, 350))

        # rwnmr config fields
        self.nameLineEdit = None
        self.walkersLineEdit = None
        self.placementBox = None
        self.placementDeviationLineEdit = None
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

        titleFont=QtGui.QFont("Arial",15)
        titleFont.setBold(True)
        headerFont = QtGui.QFont("Arial",12)
        headerFont.setBold(True)     

        # these are the app widgets connected to their slot methods
        self.titleLabel = QtWidgets.QLabel('Configurations setup')
        self.titleLabel.setFont(titleFont)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        # Initialize tab screen
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setFixedWidth(480)
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
        headerFont = QtGui.QFont("Arial",12)
        headerFont.setBold(True) 
        labelSize = 150
        fieldSize = 100

        # these are the app widgets connected to their slot methods
        titleLabel = QtWidgets.QLabel('--- RW Configuration ---')
        titleLabel.setFont(headerFont)
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # RW config fields
        nameLabel = QtWidgets.QLabel('Simulation name:')
        nameLabel.setFixedWidth(labelSize)
        self.nameLineEdit = QtWidgets.QLineEdit()
        self.nameLineEdit.setText('NMR_Simulation')        

        walkersHeaderLabel = QtWidgets.QLabel('Random-walk parameters')
        walkersHeaderLabel.setFont(headerFont)    
    
        walkersLabel = QtWidgets.QLabel('Walkers:')
        walkersLabel.setFixedWidth(labelSize)
        self.walkersLineEdit = QtWidgets.QLineEdit()
        self.walkersLineEdit.setFixedWidth(fieldSize)
        self.walkersLineEdit.setText('10000')

        placementLabel = QtWidgets.QLabel('Placement:')
        placementLabel.setFixedWidth(labelSize)
        self.placementBox = QtWidgets.QComboBox()
        placementOptions = ['random', 'point', 'cubic']
        self.placementBox.addItems(placementOptions)
        self.placementBox.setFixedWidth(fieldSize)

        deviationLabel = QtWidgets.QLabel('Deviation:')
        deviationLabel.setFixedWidth(labelSize)
        self.placementDeviationLineEdit = QtWidgets.QLineEdit('0')
        self.placementDeviationLineEdit.setFixedWidth(fieldSize)

        stepsLabel = QtWidgets.QLabel('Steps/echo:')
        stepsLabel.setFixedWidth(labelSize)
        self.stepsLineEdit = QtWidgets.QLineEdit()
        self.stepsLineEdit.setText('1')
        self.stepsLineEdit.setFixedWidth(fieldSize)

        seedLabel = QtWidgets.QLabel('RNG seed:')
        seedLabel.setFixedWidth(labelSize)
        self.seedLineEdit = QtWidgets.QLineEdit()
        self.seedLineEdit.setText('0')
        self.seedLineEdit.setFixedWidth(fieldSize)
        seedLabel2 = QtWidgets.QLabel('(0 == random)')

        physicalsLabel = QtWidgets.QLabel('Physical parameters')  
        physicalsLabel.setFont(headerFont)

        rhoLabel = QtWidgets.QLabel('Superficial relaxivity:')
        rhoLabel.setFixedWidth(labelSize)
        self.rhoTypeBox = QtWidgets.QComboBox()
        self.rhoTypeBox.setFixedWidth(fieldSize)
        rhoTypeOptions = ['uniform', 'sigmoid']
        self.rhoTypeBox.addItems(rhoTypeOptions)
        self.rhoLineEdit = QtWidgets.QLineEdit('{0.0}')
        self.rhoLineEdit.setFixedWidth(fieldSize)
        rhoUnitLabel = QtWidgets.QLabel('um/ms')

        D0Label = QtWidgets.QLabel('Fluid free diffusion coef.:')
        D0Label.setFixedWidth(labelSize)
        D0Label.setFont(QtGui.QFont("Arial", 10))
        self.D0LineEdit = QtWidgets.QLineEdit('0.0')
        self.D0LineEdit.setFixedWidth(fieldSize)
        D0UnitLabel = QtWidgets.QLabel('um²/ms')      

        histHeaderLabel = QtWidgets.QLabel('Collision histogram') 
        histHeaderLabel.setFont(headerFont)    
        histLabel = QtWidgets.QLabel('Histograms:')
        histLabel.setFixedWidth(labelSize)
        self.histLineEdit = QtWidgets.QLineEdit('1')
        self.histLineEdit.setFixedWidth(fieldSize)
        histSizeLabel = QtWidgets.QLabel('Size:')
        histSizeLabel.setFixedWidth(labelSize)
        self.histSizeLineEdit = QtWidgets.QLineEdit('1024')
        self.histSizeLineEdit.setFixedWidth(fieldSize)
        
        
        performanceLabel = QtWidgets.QLabel('Performance boost')
        performanceLabel.setFont(headerFont)
        ompLabel = QtWidgets.QLabel('Multi-thread:')
        ompLabel.setFixedWidth(labelSize)
        self.ompBox = QtWidgets.QComboBox()
        self.ompBox.setFixedWidth(fieldSize)
        self.ompBox.addItems(boolOptions)        
        gpuLabel = QtWidgets.QLabel('GPU:')
        gpuLabel.setFixedWidth(labelSize)
        self.gpuBox = QtWidgets.QComboBox()
        self.gpuBox.setFixedWidth(fieldSize)
        self.gpuBox.addItems(boolOptions)


        savingsLabel = QtWidgets.QLabel('Savings')
        savingsLabel.setFont(headerFont)
        saveInfoLabel = QtWidgets.QLabel("Save info:")
        saveInfoLabel.setFixedWidth(labelSize)
        self.saveInfoBox = QtWidgets.QComboBox()
        self.saveInfoBox.setFixedWidth(fieldSize)
        self.saveInfoBox.addItems(boolOptions) 

        saveBinImgLabel = QtWidgets.QLabel("Save binary image:")
        saveBinImgLabel.setFixedWidth(labelSize)
        self.saveBinImgBox = QtWidgets.QComboBox()
        self.saveBinImgBox.setFixedWidth(fieldSize)
        self.saveBinImgBox.addItems(boolOptions)
        self.saveBinImgBox.setCurrentIndex(1)

        fileLabel = QtWidgets.QLabel("Config file")
        fileLabel.setFont(headerFont)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        fileNameLabel.setFixedWidth(labelSize)
        self.rwFileLineEdit = QtWidgets.QLineEdit()
        self.rwFileLineEdit.setText("rwnmr")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.setFixedWidth(50)
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

        walkersLayout = QtWidgets.QVBoxLayout()
        walkersLayoutH1 = QtWidgets.QHBoxLayout()
        walkersLayoutH1.addWidget(walkersLabel)
        walkersLayoutH1.addWidget(self.walkersLineEdit) 
        walkersLayoutH1.setAlignment(QtCore.Qt.AlignLeft)  
        walkersLayoutH2 = QtWidgets.QHBoxLayout()
        walkersLayoutH2.addWidget(placementLabel)
        walkersLayoutH2.addWidget(self.placementBox)
        walkersLayoutH2.setAlignment(QtCore.Qt.AlignLeft)
        walkersLayoutH3 = QtWidgets.QHBoxLayout()
        walkersLayoutH3.addWidget(deviationLabel)
        walkersLayoutH3.addWidget(self.placementDeviationLineEdit)
        walkersLayoutH3.setAlignment(QtCore.Qt.AlignLeft)
        walkersLayout.addLayout(walkersLayoutH1)
        walkersLayout.addLayout(walkersLayoutH2)
        walkersLayout.addLayout(walkersLayoutH3)  

        stepsLayout = QtWidgets.QHBoxLayout()
        stepsLayout.addWidget(stepsLabel)
        stepsLayout.addWidget(self.stepsLineEdit)
        stepsLayout.setAlignment(QtCore.Qt.AlignLeft)

        seedLayout = QtWidgets.QHBoxLayout()
        seedLayout.addWidget(seedLabel)
        seedLayout.addWidget(self.seedLineEdit)
        seedLayout.addWidget(seedLabel2)   
        seedLayout.setAlignment(QtCore.Qt.AlignLeft)

        physicalsLayout = QtWidgets.QHBoxLayout()
        physicalsLayout.addWidget(physicalsLabel)

        rhoLayout = QtWidgets.QHBoxLayout()
        rhoLayout.addWidget(rhoLabel) 
        rhoLayout.addWidget(self.rhoTypeBox) 
        rhoLayout.addWidget(self.rhoLineEdit) 
        rhoLayout.addWidget(rhoUnitLabel)
        rhoLayout.setAlignment(QtCore.Qt.AlignLeft)

        D0Layout = QtWidgets.QHBoxLayout()
        D0Layout.addWidget(D0Label)
        D0Layout.addWidget(self.D0LineEdit)
        D0Layout.addWidget(D0UnitLabel)
        D0Layout.setAlignment(QtCore.Qt.AlignLeft)

        # --
        histHeaderLayout = QtWidgets.QHBoxLayout()
        histHeaderLayout.addWidget(histHeaderLabel)

        histLayout = QtWidgets.QVBoxLayout()
        histLayoutH1 = QtWidgets.QHBoxLayout()
        histLayoutH1.addWidget(histLabel)
        histLayoutH1.addWidget(self.histLineEdit)
        histLayoutH1.setAlignment(QtCore.Qt.AlignLeft)
        histLayoutH2 = QtWidgets.QHBoxLayout()
        histLayoutH2.addWidget(histSizeLabel)
        histLayoutH2.addWidget(self.histSizeLineEdit)
        histLayoutH2.setAlignment(QtCore.Qt.AlignLeft)
        histLayout.addLayout(histLayoutH1)
        histLayout.addLayout(histLayoutH2)

        performanceLayout = QtWidgets.QHBoxLayout()
        performanceLayout.addWidget(performanceLabel)
        ompLayout = QtWidgets.QHBoxLayout()
        ompLayout.addWidget(ompLabel)
        ompLayout.addWidget(self.ompBox)
        ompLayout.setAlignment(QtCore.Qt.AlignLeft)
        gpuLayout = QtWidgets.QHBoxLayout()
        gpuLayout.addWidget(gpuLabel)
        gpuLayout.addWidget(self.gpuBox)
        gpuLayout.setAlignment(QtCore.Qt.AlignLeft)


        savingsLayout = QtWidgets.QHBoxLayout()
        savingsLayout.addWidget(savingsLabel)
        saveInfoLayout = QtWidgets.QHBoxLayout()
        saveInfoLayout.addWidget(saveInfoLabel)
        saveInfoLayout.addWidget(self.saveInfoBox)
        saveInfoLayout.setAlignment(QtCore.Qt.AlignLeft)
        saveBinImgLayout = QtWidgets.QHBoxLayout()
        saveBinImgLayout.addWidget(saveBinImgLabel)
        saveBinImgLayout.addWidget(self.saveBinImgBox)
        saveBinImgLayout.setAlignment(QtCore.Qt.AlignLeft)
           
        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayout = QtWidgets.QVBoxLayout()
        fileLayoutH1 = QtWidgets.QHBoxLayout()
        fileLayoutH1.addWidget(fileNameLabel)
        fileLayoutH1.addWidget(self.rwFileLineEdit)
        fileLayoutH1.addWidget(fileExtensionLabel)
        fileLayoutH2 = QtWidgets.QHBoxLayout()  
        fileLayoutH2.addWidget(saveButton)
        fileLayout.addLayout(fileLayoutH1)
        fileLayout.addLayout(fileLayoutH2)

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
        filename = CONFIG_PATH + self.rwFileLineEdit.text() + ".config"
        with open(filename, "w") as file:
            file.write("--- RWNMR Configuration\n")
            file.write("NAME: {}\n".format(self.nameLineEdit.text()))
            
            file.write("-- PARAMETERS\n")
            file.write("WALKERS: {}\n".format(self.walkersLineEdit.text()))
            file.write("WALKERS_PLACEMENT: {}\n".format(self.placementBox.currentText()))
            file.write("PLACEMENT_DEVIATION: {}\n".format(self.placementDeviationLineEdit.text()))
            file.write("RHO_TYPE: {}\n".format(self.rhoTypeBox.currentText()))           
            file.write("RHO: {}\n".format(self.rhoLineEdit.text()))
            file.write("D0: {}\n".format(self.D0LineEdit.text()))
            file.write("STEPS_PER_ECHO: {}\n".format(self.stepsLineEdit.text()))
            file.write("SEED: {}\n".format(self.seedLineEdit.text()))

            file.write("-- COLLISION HISTOGRAMS\n")
            file.write("HISTOGRAMS: {}\n".format(self.histLineEdit.text()))
            file.write("HISTOGRAM_SIZE: {}\n".format(self.histSizeLineEdit.text()))

            file.write("-- PERFORMANCE\n")
            file.write("OPENMP_USAGE: {}\n".format(self.ompBox.currentText()))
            file.write("GPU_USAGE: {}\n".format(self.gpuBox.currentText()))
            
            file.write("-- SAVE MODE\n")
            file.write("SAVE_IMG_INFO: {}\n".format(self.saveInfoBox.currentText()))
            file.write("SAVE_BINIMG: {}\n".format(self.saveBinImgBox.currentText()))           
            
        self.parent.m_setup_tab.m_setup.rwConfigLineEdit.setText(filename)
        self.parent.m_setup_tab.m_setup.rwConfigLineEdit.setEnabled(True)        
        return
    
    def createNewUCTConfigTab(self):
        boolOptions = ['true', 'false']
        headerFont = QtGui.QFont("Arial",12)
        headerFont.setBold(True) 
        labelSize = 150
        fieldSize = 100

        titleLabel = QtWidgets.QLabel('--- uCT image configuration ---')
        titleLabel.setFont(headerFont)
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        header1 = QtWidgets.QLabel('Image parameters')
        header1.setFont(headerFont)
            
        slicesLabel = QtWidgets.QLabel('Image slices:')
        slicesLabel.setFixedWidth(labelSize)
        self.slicesLineEdit = QtWidgets.QLineEdit('1')
        self.slicesLineEdit.setFixedWidth(fieldSize)   

        resolutionLabel = QtWidgets.QLabel('Image resolution:')
        resolutionLabel.setFixedWidth(labelSize)
        self.resolutionLineEdit = QtWidgets.QLineEdit('1.0')
        self.resolutionLineEdit.setFixedWidth(fieldSize)
        resolutionUnitLabel = QtWidgets.QLabel('um/voxel')
        resolutionUnitLabel.setFixedWidth(80)

        voxelDivisionsLabel = QtWidgets.QLabel('Voxel subdivisions:')
        voxelDivisionsLabel.setFixedWidth(labelSize)
        self.voxelDivisionsLineEdit = QtWidgets.QLineEdit('0')
        self.voxelDivisionsLineEdit.setFixedWidth(fieldSize)

        fileLabel = QtWidgets.QLabel("Config file")
        fileLabel.setFont(headerFont)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        fileNameLabel.setFixedWidth(labelSize)
        self.uctFileLineEdit = QtWidgets.QLineEdit()
        self.uctFileLineEdit.setText("uct")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.setFixedWidth(50)
        saveButton.clicked.connect(self.saveUCTConfig)


        # set layouts
        mainLayout = QtWidgets.QVBoxLayout(self.open_tabs[-1])
        mainLayout.addWidget(titleLabel)

        headerLayout = QtWidgets.QHBoxLayout()
        headerLayout.addWidget(header1)

        ImgLayout = QtWidgets.QHBoxLayout()     
        ImgLayout.addWidget(slicesLabel)
        ImgLayout.addWidget(self.slicesLineEdit)
        ImgLayout.setAlignment(QtCore.Qt.AlignLeft)

        resolutionLayout = QtWidgets.QHBoxLayout()
        resolutionLayout.addWidget(resolutionLabel)
        resolutionLayout.addWidget(self.resolutionLineEdit)
        resolutionLayout.addWidget(resolutionUnitLabel)
        resolutionLayout.setAlignment(QtCore.Qt.AlignLeft)

        voxelDivisionsLayout = QtWidgets.QHBoxLayout()
        voxelDivisionsLayout.addWidget(voxelDivisionsLabel)
        voxelDivisionsLayout.addWidget(self.voxelDivisionsLineEdit)
        voxelDivisionsLayout.setAlignment(QtCore.Qt.AlignLeft)

        layout1 = QtWidgets.QVBoxLayout()
        layout1.addLayout(headerLayout)
        layout1.addLayout(ImgLayout)
        layout1.addLayout(resolutionLayout)
        layout1.addLayout(voxelDivisionsLayout)
        layout1.setAlignment(QtCore.Qt.AlignLeft)

        fileLayout = QtWidgets.QVBoxLayout()
        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayoutH1 = QtWidgets.QHBoxLayout()
        fileLayoutH1.addWidget(fileNameLabel)
        fileLayoutH1.addWidget(self.uctFileLineEdit)
        fileLayoutH1.addWidget(fileExtensionLabel)        
        fileLayoutH2 = QtWidgets.QHBoxLayout()
        fileLayoutH2.addWidget(saveButton)
        fileLayout.addLayout(fileHeaderLayout)
        fileLayout.addLayout(fileLayoutH1)
        fileLayout.addLayout(fileLayoutH2)      

        # add to main layout
        mainLayout.addLayout(layout1)
        mainLayout.addLayout(fileLayout)
        

        # top alignment 
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
        filename = CONFIG_PATH + self.uctFileLineEdit.text() + ".config"
        with open(filename, "w") as file:
            file.write("--- UCT Configuration\n")
            file.write("SLICES: {}\n".format(self.slicesLineEdit.text()))
            file.write("RESOLUTION: {}\n".format(self.resolutionLineEdit.text()))
            file.write("VOXEL_DIVISION: {}\n".format(self.voxelDivisionsLineEdit.text()))
            file.write("IMG_FILES_LIST: {}\n".format(CONFIG_PATH + 'imgs/ImagesList.txt'))            

        self.parent.m_setup_tab.m_setup.uctConfigLineEdit.setText(filename)
        self.parent.m_setup_tab.m_setup.uctConfigLineEdit.setEnabled(True)
        return

    def createNewPFGSEConfigTab(self, _index):
        boolOptions = ['true', 'false']
        headerFont = QtGui.QFont("Arial",12)
        headerFont.setBold(True) 
        labelSize = 150
        fieldSize = 100

        # these are the app widgets connected to their slot methods
        titleLabel = QtWidgets.QLabel('--- PFGSE Configuration ---')
        titleLabel.setFont(headerFont)
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        header1 = QtWidgets.QLabel('Physical attributes')
        header1.setFont(headerFont)
        header2 = QtWidgets.QLabel('Gradient vector')
        header2.setFont(headerFont)
        header3 = QtWidgets.QLabel('Time sequence')
        header3.setFont(headerFont)
        header4 = QtWidgets.QLabel('Magnetization threshold')
        header4.setFont(headerFont)
        header5 = QtWidgets.QLabel('Savings')
        header5.setFont(headerFont)
        
        # PFGSE config fields
        # header 1
        D0Label = QtWidgets.QLabel('Fluid free diffusion coef.:')
        D0Label.setFixedWidth(labelSize)
        D0Label.setFont(QtGui.QFont("Arial", 10))
        D0LineEdit = QtWidgets.QLineEdit('2.5')
        D0LineEdit.setFixedWidth(fieldSize)
        D0UnitLabel = QtWidgets.QLabel('um²/ms')

        gyroLabel = QtWidgets.QLabel('Fluid gyromagnetic ratio:')
        gyroLabel.setFixedWidth(labelSize)
        gyroLabel.setFont(QtGui.QFont("Arial", 10))
        gyroLineEdit = QtWidgets.QLineEdit('42.576')
        gyroLineEdit.setFixedWidth(fieldSize)
        gyroUnitLabel = QtWidgets.QLabel('MHz/T')

        pulseWidthLabel = QtWidgets.QLabel('Pulse width:')
        pulseWidthLabel.setFixedWidth(labelSize)
        pulseWidthLineEdit = QtWidgets.QLineEdit('0.1')
        pulseWidthLineEdit.setFixedWidth(fieldSize)
        pulseWidthUnitLabel = QtWidgets.QLabel('ms')

        # header 2
        maxGradientXLabel = QtWidgets.QLabel('Max Gx: ')
        maxGradientXLabel.setFixedWidth(labelSize)
        maxGradientXLineEdit = QtWidgets.QLineEdit('0.0')
        maxGradientXLineEdit.setFixedWidth(fieldSize)
        maxGradientYLabel = QtWidgets.QLabel('Max Gy: ')
        maxGradientYLabel.setFixedWidth(labelSize)
        maxGradientYLineEdit = QtWidgets.QLineEdit('0.0')
        maxGradientYLineEdit.setFixedWidth(fieldSize)
        maxGradientZLabel = QtWidgets.QLabel('Max Gz: ')
        maxGradientZLabel.setFixedWidth(labelSize)
        maxGradientZLineEdit = QtWidgets.QLineEdit('0.0')
        maxGradientZLineEdit.setFixedWidth(fieldSize)
        gradientXUnitLabel = QtWidgets.QLabel('Gauss/cm')
        gradientYUnitLabel = QtWidgets.QLabel('Gauss/cm')
        gradientZUnitLabel = QtWidgets.QLabel('Gauss/cm')

        gradientSamplesLabel = QtWidgets.QLabel('Samples: ')
        gradientSamplesLabel.setFixedWidth(labelSize)
        gradientSamplesLineEdit = QtWidgets.QLineEdit('200')
        gradientSamplesLineEdit.setFixedWidth(fieldSize)

        # header 3
        sequenceTypes = ['manual', 'linear', 'log']
        sequenceLabel = QtWidgets.QLabel('Type: ')
        sequenceLabel.setFixedWidth(labelSize)
        sequenceBox = QtWidgets.QComboBox()
        sequenceBox.setFixedWidth(fieldSize)
        sequenceBox.addItems(sequenceTypes)
        timeSamplesLabel = QtWidgets.QLabel('Samples: ')
        timeSamplesLabel.setFixedWidth(labelSize)
        timeSamplesLineEdit = QtWidgets.QLineEdit('4')
        timeSamplesLineEdit.setFixedWidth(fieldSize)
        timeValuesLabel = QtWidgets.QLabel('Values: ')
        timeValuesLabel.setFixedWidth(labelSize)
        timeValuesLineEdit = QtWidgets.QLineEdit('{0.2, 0.5, 1.0, 2.0}')
        timeValuesLineEdit.setFixedWidth(labelSize)
        timeUnitLabel = QtWidgets.QLabel('ms')
        timeMinLabel = QtWidgets.QLabel('min: ')
        timeMinLabel.setFixedWidth(labelSize)
        timeMinLineEdit = QtWidgets.QLineEdit('-1.0')
        timeMinLineEdit.setFixedWidth(fieldSize)
        timeMaxLabel = QtWidgets.QLabel('max: ')
        timeMaxLabel.setFixedWidth(labelSize)
        timeMaxLineEdit = QtWidgets.QLineEdit('1.0')
        timeMaxLineEdit.setFixedWidth(fieldSize)
        scaleLabel = QtWidgets.QLabel('Apply scale factor: ')
        scaleLabel.setFixedWidth(labelSize)  
        scaleBox = QtWidgets.QComboBox()
        scaleBox.setFixedWidth(fieldSize)
        scaleBox.addItems(boolOptions)
        inspectionLengthLabel = QtWidgets.QLabel('Inspection length: ')
        inspectionLengthLabel.setFixedWidth(labelSize)  
        inspectionLengthLineEdit = QtWidgets.QLineEdit('5.0')
        inspectionLengthLineEdit.setFixedWidth(fieldSize)
        inspectionLengthUnitLabel = QtWidgets.QLabel('um')

        # header 4
        thresholdTypes = ['none', 'samples', 'lhs', 'rhs']
        thresholdTypeLabel = QtWidgets.QLabel('type: ')
        thresholdTypeLabel.setFixedWidth(labelSize)
        thresholdTypeBox = QtWidgets.QComboBox()
        thresholdTypeBox.setFixedWidth(fieldSize)
        thresholdTypeBox.addItems(thresholdTypes)
        thresholdValueLabel = QtWidgets.QLabel('value: ')
        thresholdValueLabel.setFixedWidth(labelSize)
        thresholdValueLineEdit = QtWidgets.QLineEdit('0')
        thresholdValueLineEdit.setFixedWidth(fieldSize)

        # header 5
        saveModeLabel = QtWidgets.QLabel("Save mode:")
        saveModeLabel.setFixedWidth(labelSize)
        saveModeBox = QtWidgets.QComboBox()
        saveModeBox.setFixedWidth(fieldSize)
        saveModeBox.addItems(boolOptions)
        savePFGSELabel = QtWidgets.QLabel("Save pfgse:")
        savePFGSELabel.setFixedWidth(labelSize)
        savePFGSEBox = QtWidgets.QComboBox()
        savePFGSEBox.setFixedWidth(fieldSize)
        savePFGSEBox.addItems(boolOptions)
        saveCollisionsLabel = QtWidgets.QLabel("Save collisions:")
        saveCollisionsLabel.setFixedWidth(labelSize)
        saveCollisionsBox = QtWidgets.QComboBox()
        saveCollisionsBox.setFixedWidth(fieldSize)
        saveCollisionsBox.addItems(boolOptions)
        saveDecayLabel = QtWidgets.QLabel("Save decay:")
        saveDecayLabel.setFixedWidth(labelSize)
        saveDecayBox = QtWidgets.QComboBox()
        saveDecayBox.setFixedWidth(fieldSize)
        saveDecayBox.addItems(boolOptions)
        saveHistogramLabel = QtWidgets.QLabel("Save histogram:")
        saveHistogramLabel.setFixedWidth(labelSize)
        saveHistogramBox = QtWidgets.QComboBox()
        saveHistogramBox.setFixedWidth(fieldSize)
        saveHistogramBox.addItems(boolOptions)
        saveHistListLabel = QtWidgets.QLabel("Save histogram list:")  
        saveHistListLabel.setFixedWidth(labelSize)  
        saveHistListBox = QtWidgets.QComboBox()
        saveHistListBox.setFixedWidth(fieldSize)
        saveHistListBox.addItems(boolOptions)
        
        # config file 
        fileLabel = QtWidgets.QLabel("Config file")
        fileLabel.setFont(headerFont)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        fileNameLabel.setFixedWidth(labelSize)  
        pfgseFileLineEdit = QtWidgets.QLineEdit()
        pfgseFileLineEdit.setText("pfgse")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.setFixedWidth(50)
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
        layoutV1H1.setAlignment(QtCore.Qt.AlignLeft)
        layoutV1.addLayout(layoutV1H1)
        
        layoutV1H2 = QtWidgets.QHBoxLayout()
        layoutV1H2.addWidget(gyroLabel)
        layoutV1H2.addWidget(gyroLineEdit)
        layoutV1H2.addWidget(gyroUnitLabel)
        layoutV1H2.setAlignment(QtCore.Qt.AlignLeft)
        layoutV1.addLayout(layoutV1H2)

        layoutV1H3 = QtWidgets.QHBoxLayout()
        layoutV1H3.addWidget(pulseWidthLabel)
        layoutV1H3.addWidget(pulseWidthLineEdit)
        layoutV1H3.addWidget(pulseWidthUnitLabel)
        layoutV1H3.setAlignment(QtCore.Qt.AlignLeft)
        layoutV1.addLayout(layoutV1H3)

        layoutV1.addWidget(QtWidgets.QLabel(''))

        # header 2
        layoutV2 = QtWidgets.QVBoxLayout()
        layoutV2.addWidget(header2)

        layoutV2H2 = QtWidgets.QHBoxLayout()
        layoutV2H2.addWidget(gradientSamplesLabel)
        layoutV2H2.addWidget(gradientSamplesLineEdit)
        layoutV2H2.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H2)
        
        layoutV2H1a = QtWidgets.QHBoxLayout()
        layoutV2H1a.addWidget(maxGradientXLabel)
        layoutV2H1a.addWidget(maxGradientXLineEdit)
        layoutV2H1a.addWidget(gradientXUnitLabel)
        layoutV2H1a.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H1a)
        
        layoutV2H1b = QtWidgets.QHBoxLayout()
        layoutV2H1b.addWidget(maxGradientYLabel)
        layoutV2H1b.addWidget(maxGradientYLineEdit)
        layoutV2H1b.addWidget(gradientYUnitLabel)
        layoutV2H1b.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H1b)
        
        layoutV2H1c = QtWidgets.QHBoxLayout() 
        layoutV2H1c.addWidget(maxGradientZLabel)
        layoutV2H1c.addWidget(maxGradientZLineEdit)
        layoutV2H1c.addWidget(gradientZUnitLabel)
        layoutV2H1c.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H1c)      

        layoutV2.addWidget(QtWidgets.QLabel(''))

        # header 3
        layoutV3 = QtWidgets.QVBoxLayout()
        layoutV3.addWidget(header3)
        
        layoutV3H1a = QtWidgets.QHBoxLayout()
        layoutV3H1a.addWidget(timeSamplesLabel)
        layoutV3H1a.addWidget(timeSamplesLineEdit)
        layoutV3H1a.setAlignment(QtCore.Qt.AlignLeft)    
        layoutV3H1b = QtWidgets.QHBoxLayout()
        layoutV3H1b.addWidget(sequenceLabel)
        layoutV3H1b.addWidget(sequenceBox)    
        layoutV3H1b.setAlignment(QtCore.Qt.AlignLeft)    
        layoutV3.addLayout(layoutV3H1a)
        layoutV3.addLayout(layoutV3H1b)        

        layoutV3H3 = QtWidgets.QHBoxLayout()
        layoutV3H3.addWidget(timeValuesLabel)
        layoutV3H3.addWidget(timeValuesLineEdit)
        layoutV3H3.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H3)

        layoutV3H4a = QtWidgets.QHBoxLayout()
        layoutV3H4a.addWidget(timeMinLabel)
        layoutV3H4a.addWidget(timeMinLineEdit)
        layoutV3H4a.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3H4b = QtWidgets.QHBoxLayout()
        layoutV3H4b.addWidget(timeMaxLabel)
        layoutV3H4b.addWidget(timeMaxLineEdit)
        layoutV3H4b.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H4a)
        layoutV3.addLayout(layoutV3H4b)

        layoutV3H5 = QtWidgets.QHBoxLayout()
        layoutV3H5.addWidget(scaleLabel)
        layoutV3H5.addWidget(scaleBox)   
        layoutV3H5.setAlignment(QtCore.Qt.AlignLeft) 
        layoutV3.addLayout(layoutV3H5) 

        layoutV3H6 = QtWidgets.QHBoxLayout()
        layoutV3H6.addWidget(inspectionLengthLabel)
        layoutV3H6.addWidget(inspectionLengthLineEdit)
        layoutV3H6.addWidget(inspectionLengthUnitLabel) 
        layoutV3H6.setAlignment(QtCore.Qt.AlignLeft)       
        layoutV3.addLayout(layoutV3H6) 

        layoutV3.addWidget(QtWidgets.QLabel(''))

        # header 4
        layoutV4 = QtWidgets.QVBoxLayout()
        layoutV4.addWidget(header4)

        layoutV4H1a = QtWidgets.QHBoxLayout()
        layoutV4H1a.addWidget(thresholdValueLabel)
        layoutV4H1a.addWidget(thresholdValueLineEdit)
        layoutV4H1a.setAlignment(QtCore.Qt.AlignLeft)
        layoutV4H1b = QtWidgets.QHBoxLayout()
        layoutV4H1b.addWidget(thresholdTypeLabel)
        layoutV4H1b.addWidget(thresholdTypeBox)
        layoutV4H1b.setAlignment(QtCore.Qt.AlignLeft)
        layoutV4.addLayout(layoutV4H1a)
        layoutV4.addLayout(layoutV4H1b)

        layoutV4.addWidget(QtWidgets.QLabel(''))

        # header 5
        layoutV5 = QtWidgets.QVBoxLayout()
        layoutV5.addWidget(header5)

        layoutV5H1 = QtWidgets.QHBoxLayout()
        layoutV5H1.addWidget(saveModeLabel)
        layoutV5H1.addWidget(saveModeBox)
        layoutV5H1.setAlignment(QtCore.Qt.AlignLeft)
        layoutV5.addLayout(layoutV5H1)

        layoutV5H2 = QtWidgets.QHBoxLayout()
        layoutV5H2.addWidget(savePFGSELabel)
        layoutV5H2.addWidget(savePFGSEBox)
        layoutV5H2.setAlignment(QtCore.Qt.AlignLeft)
        layoutV5.addLayout(layoutV5H2)

        layoutV5H3 = QtWidgets.QHBoxLayout()
        layoutV5H3.addWidget(saveCollisionsLabel)
        layoutV5H3.addWidget(saveCollisionsBox)
        layoutV5H3.setAlignment(QtCore.Qt.AlignLeft)
        layoutV5.addLayout(layoutV5H3)
        
        layoutV5H4 = QtWidgets.QHBoxLayout()
        layoutV5H4.addWidget(saveDecayLabel)
        layoutV5H4.addWidget(saveDecayBox)
        layoutV5H4.setAlignment(QtCore.Qt.AlignLeft)
        layoutV5.addLayout(layoutV5H4)

        layoutV5H5 = QtWidgets.QHBoxLayout()
        layoutV5H5.addWidget(saveHistogramLabel)
        layoutV5H5.addWidget(saveHistogramBox)
        layoutV5H5.setAlignment(QtCore.Qt.AlignLeft)
        layoutV5.addLayout(layoutV5H5)

        layoutV5H6 = QtWidgets.QHBoxLayout()
        layoutV5H6.addWidget(saveHistListLabel)
        layoutV5H6.addWidget(saveHistListBox)
        layoutV5H6.setAlignment(QtCore.Qt.AlignLeft)
        layoutV5.addLayout(layoutV5H6)       
        
        layoutV5.addWidget(QtWidgets.QLabel(''))

        # file configs
        fileLayout = QtWidgets.QVBoxLayout()
        
        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayoutH1 = QtWidgets.QHBoxLayout()
        fileLayoutH1.addWidget(fileNameLabel)
        fileLayoutH1.addWidget(pfgseFileLineEdit)
        fileLayoutH1.addWidget(fileExtensionLabel)        
        fileLayoutH2 = QtWidgets.QHBoxLayout()
        fileLayoutH2.addWidget(saveButton)
        fileLayout.addLayout(fileHeaderLayout)
        fileLayout.addLayout(fileLayoutH1)
        fileLayout.addLayout(fileLayoutH2)
        

        # add to main layout
        mainLayout.addLayout(layoutV1)
        mainLayout.addLayout(layoutV2)
        mainLayout.addLayout(layoutV3)
        mainLayout.addLayout(layoutV4)
        mainLayout.addLayout(layoutV5)
        # mainLayout.addLayout(fileHeaderLayout)
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
        filename = CONFIG_PATH + self.procedures_qwidgets[_index]["configFilename"].text() + ".config"
        with open(filename, "w") as file:
            file.write("--- PFGSE Configuration\n")
            file.write("-- PHYSICAL ATTRIBUTES\n")
            file.write("D0: {}\n".format(self.procedures_qwidgets[_index]["D0"].text()))
            file.write("GIROMAGNETIC_RATIO: {}\n".format(self.procedures_qwidgets[_index]["gyro"].text()))
            file.write("PULSE_WIDTH: {}\n".format(self.procedures_qwidgets[_index]["pulseWidth"].text()))
            
            file.write("-- GRADIENT VECTOR\n")
            file.write("GRADIENT SAMPLES: {}\n".format(self.procedures_qwidgets[_index]["gradientSamples"].text()))
            gradX_str = self.procedures_qwidgets[_index]["maxGradientX"].text()
            gradY_str = self.procedures_qwidgets[_index]["maxGradientY"].text()
            gradZ_str = self.procedures_qwidgets[_index]["maxGradientZ"].text()            
            gradientVectorString = "{" + gradX_str + ", " + gradY_str + ", " + gradZ_str + "}"
            file.write("MAX_GRADIENT: {}\n".format(gradientVectorString))

            file.write("-- TIME SEQUENCE\n")
            file.write("TIME_SEQ: {}\n".format(self.procedures_qwidgets[_index]["timeSequence"].currentText()))
            file.write("TIME_SAMPLES: {}\n".format(self.procedures_qwidgets[_index]["timeSamples"].text()))
            file.write("TIME_VALUES: {}\n".format(self.procedures_qwidgets[_index]["timeValues"].text()))
            file.write("TIME_MIN: {}\n".format(self.procedures_qwidgets[_index]["timeMin"].text()))
            file.write("TIME_MAX: {}\n".format(self.procedures_qwidgets[_index]["timeMax"].text()))
            file.write("APPLY_SCALE_FACTOR: {}\n".format(self.procedures_qwidgets[_index]["scaleFactor"].currentText()))
            file.write("INSPECTION_LENGTH: {}\n".format(self.procedures_qwidgets[_index]["inspectionLength"].text()))

            file.write("-- THRESHOLD\n")            
            file.write("THRESHOLD_TYPE: {}\n".format(self.procedures_qwidgets[_index]["thresholdType"].currentText()))
            file.write("THRESHOLD_VALUE: {}\n".format(self.procedures_qwidgets[_index]["thresholdValue"].text()))

            file.write("-- SAVE MODE\n")
            file.write("SAVE_MODE: {}\n".format(self.procedures_qwidgets[_index]["saveMode"].currentText()))
            file.write("SAVE_PFGSE: {}\n".format(self.procedures_qwidgets[_index]["savePFGSE"].currentText()))
            file.write("SAVE_COLLISIONS: {}\n".format(self.procedures_qwidgets[_index]["saveCollisions"].currentText()))
            file.write("SAVE_DECAY: {}\n".format(self.procedures_qwidgets[_index]["saveDecay"].currentText()))
            file.write("SAVE_HISTOGRAM: {}\n".format(self.procedures_qwidgets[_index]["saveHistogram"].currentText()))
            file.write("SAVE_HISTOGRAM_LIST: {}\n".format(self.procedures_qwidgets[_index]["saveHistList"].currentText()))

        self.parent.m_setup_tab.m_setup.procedure_paths[_index].setText(filename)
        self.parent.m_setup_tab.m_setup.procedure_paths[_index].setEnabled(True)
        return
    
    def createNewCPMGConfigTab(self, _index):
        boolOptions = ['true', 'false']
        headerFont = QtGui.QFont("Arial",12)
        headerFont.setBold(True) 
        labelSize = 150
        fieldSize = 100

        # these are the app widgets connected to their slot methods
        titleLabel = QtWidgets.QLabel('--- CPMG Configuration ---')
        titleLabel.setFont(headerFont)
        titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        header1 = QtWidgets.QLabel('Physical attributes')
        header1.setFont(headerFont)
        header2 = QtWidgets.QLabel('Laplace inversion')
        header2.setFont(headerFont)
        header3 = QtWidgets.QLabel('Savings')
        header3.setFont(headerFont)
        
        
        # PFGSE config fields
        # header 1
        D0Label = QtWidgets.QLabel('Fluid free diffusion coef:')
        D0Label.setFixedWidth(labelSize)
        D0Label.setFont(QtGui.QFont("Arial",10))
        D0LineEdit = QtWidgets.QLineEdit('2.5')
        D0LineEdit.setFixedWidth(fieldSize)
        D0UnitLabel = QtWidgets.QLabel('um²/ms')

        expTimeLabel = QtWidgets.QLabel('Experiment time:')
        expTimeLabel.setFixedWidth(labelSize)
        expTimeLineEdit = QtWidgets.QLineEdit('2000.0')
        expTimeLineEdit.setFixedWidth(fieldSize)
        expTimeUnitLabel = QtWidgets.QLabel('ms')

        methodLabel = QtWidgets.QLabel('RW Method:')
        methodLabel.setFixedWidth(labelSize)
        methodBox = QtWidgets.QComboBox()
        methodBox.setFixedWidth(labelSize)
        methodBox.addItems(['image-based','histogram'])

        # header 2
        minT2Label = QtWidgets.QLabel('Min T2:')
        minT2Label.setFixedWidth(labelSize)
        minT2LineEdit = QtWidgets.QLineEdit('0.1')
        minT2LineEdit.setFixedWidth(fieldSize)
        maxT2Label = QtWidgets.QLabel('Max T2:')
        maxT2Label.setFixedWidth(labelSize)
        maxT2LineEdit = QtWidgets.QLineEdit('10000')
        maxT2LineEdit.setFixedWidth(fieldSize)
        T2binsLabel = QtWidgets.QLabel("T2 bins:")
        T2binsLabel.setFixedWidth(labelSize)
        T2binsLineEdit = QtWidgets.QLineEdit('128')
        T2binsLineEdit.setFixedWidth(fieldSize)        
        logspaceLabel = QtWidgets.QLabel('Use logspace:')
        logspaceLabel.setFixedWidth(labelSize)
        logspaceBox = QtWidgets.QComboBox()
        logspaceBox.setFixedWidth(fieldSize)
        logspaceBox.addItems(boolOptions)
        minLambdaLabel = QtWidgets.QLabel('Min lambda:')
        minLambdaLabel.setFixedWidth(labelSize)
        minLambdaLineEdit = QtWidgets.QLineEdit('-4.0')
        minLambdaLineEdit.setFixedWidth(fieldSize)
        maxLambdaLabel = QtWidgets.QLabel('Max lambda:')
        maxLambdaLabel.setFixedWidth(labelSize)
        maxLambdaLineEdit = QtWidgets.QLineEdit('2.0')
        maxLambdaLineEdit.setFixedWidth(fieldSize)
        numLambdasLabel = QtWidgets.QLabel('Lambdas:')
        numLambdasLabel.setFixedWidth(labelSize)
        numLambdasLineEdit = QtWidgets.QLineEdit('512')
        numLambdasLineEdit.setFixedWidth(fieldSize)
        pruneNumLabel = QtWidgets.QLabel('Prunes:')
        pruneNumLabel.setFixedWidth(labelSize)
        pruneNumLineEdit = QtWidgets.QLineEdit('512')
        pruneNumLineEdit.setFixedWidth(fieldSize)
        noiseAmpLabel = QtWidgets.QLabel('Noise amplitude:')
        noiseAmpLabel.setFixedWidth(labelSize)
        noiseAmpLineEdit = QtWidgets.QLineEdit('0.0')  
        noiseAmpLineEdit.setFixedWidth(fieldSize) 

        # header 3
        saveModeLabel = QtWidgets.QLabel("Save mode:")
        saveModeBox = QtWidgets.QComboBox()
        saveModeLabel.setFixedWidth(labelSize)
        saveModeBox.setFixedWidth(fieldSize)
        saveModeBox.addItems(boolOptions)
        saveT2Label = QtWidgets.QLabel("Save T2:")
        saveT2Label.setFixedWidth(labelSize)
        saveT2Box = QtWidgets.QComboBox()
        saveT2Box.setFixedWidth(fieldSize)
        saveT2Box.addItems(boolOptions)
        saveCollisionsLabel = QtWidgets.QLabel("Save collisions:")
        saveCollisionsLabel.setFixedWidth(labelSize)
        saveCollisionsBox = QtWidgets.QComboBox()
        saveCollisionsBox.setFixedWidth(fieldSize)
        saveCollisionsBox.addItems(boolOptions)
        saveDecayLabel = QtWidgets.QLabel("Save decay:")
        saveDecayLabel.setFixedWidth(labelSize)
        saveDecayBox = QtWidgets.QComboBox()
        saveDecayBox.setFixedWidth(fieldSize)
        saveDecayBox.addItems(boolOptions)
        saveHistogramLabel = QtWidgets.QLabel("Save histogram:")
        saveHistogramLabel.setFixedWidth(labelSize)
        saveHistogramBox = QtWidgets.QComboBox()
        saveHistogramBox.setFixedWidth(fieldSize)
        saveHistogramBox.addItems(boolOptions)
        saveHistListLabel = QtWidgets.QLabel("Save histogram list:")    
        saveHistListLabel.setFixedWidth(labelSize)
        saveHistListBox = QtWidgets.QComboBox()
        saveHistListBox.setFixedWidth(fieldSize)
        saveHistListBox.addItems(boolOptions)
        
        # config file 
        fileLabel = QtWidgets.QLabel("Config file")
        fileLabel.setFont(headerFont)
        fileNameLabel = QtWidgets.QLabel("Name: ")
        fileNameLabel.setFixedWidth(labelSize)
        pfgseFileLineEdit = QtWidgets.QLineEdit()
        pfgseFileLineEdit.setText("cpmg")
        fileExtensionLabel = QtWidgets.QLabel(".config")

        saveButton = QtWidgets.QPushButton("Save")
        saveButton.setFixedWidth(50)
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
        layoutV1H1.setAlignment(QtCore.Qt.AlignLeft)
        layoutV1.addLayout(layoutV1H1)
        
        layoutV1H2 = QtWidgets.QHBoxLayout()
        layoutV1H2.addWidget(expTimeLabel)
        layoutV1H2.addWidget(expTimeLineEdit)
        layoutV1H2.addWidget(expTimeUnitLabel)
        layoutV1H2.setAlignment(QtCore.Qt.AlignLeft)
        layoutV1.addLayout(layoutV1H2)

        layoutV1H3 = QtWidgets.QHBoxLayout()
        layoutV1H3.addWidget(methodLabel)
        layoutV1H3.addWidget(methodBox)
        layoutV1H3.setAlignment(QtCore.Qt.AlignLeft)
        layoutV1.addLayout(layoutV1H3)

        layoutV1.addWidget(QtWidgets.QLabel(''))

        # header 2
        layoutV2 = QtWidgets.QVBoxLayout()
        layoutV2.addWidget(header2)
        
        layoutV2H1a = QtWidgets.QHBoxLayout()
        layoutV2H1a.addWidget(minT2Label)
        layoutV2H1a.addWidget(minT2LineEdit)
        layoutV2H1a.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H1a)

        layoutV2H1b = QtWidgets.QHBoxLayout()
        layoutV2H1b.addWidget(maxT2Label)
        layoutV2H1b.addWidget(maxT2LineEdit)
        layoutV2H1b.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H1b)
        
        layoutV2H2 = QtWidgets.QHBoxLayout()
        layoutV2H2.addWidget(T2binsLabel)
        layoutV2H2.addWidget(T2binsLineEdit)
        layoutV2H2.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H2)

        layoutV2H3 = QtWidgets.QHBoxLayout()
        layoutV2H3.addWidget(logspaceLabel)
        layoutV2H3.addWidget(logspaceBox)
        layoutV2H3.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H3)

        layoutV2H4a = QtWidgets.QHBoxLayout()
        layoutV2H4a.addWidget(minLambdaLabel)
        layoutV2H4a.addWidget(minLambdaLineEdit)
        layoutV2H4a.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H4a)

        layoutV2H4b = QtWidgets.QHBoxLayout()
        layoutV2H4b.addWidget(maxLambdaLabel)
        layoutV2H4b.addWidget(maxLambdaLineEdit)
        layoutV2H4b.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H4b)

        layoutV2H5 = QtWidgets.QHBoxLayout()
        layoutV2H5.addWidget(numLambdasLabel)
        layoutV2H5.addWidget(numLambdasLineEdit)
        layoutV2H5.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H5)

        layoutV2H6 = QtWidgets.QHBoxLayout()
        layoutV2H6.addWidget(pruneNumLabel)
        layoutV2H6.addWidget(pruneNumLineEdit)
        layoutV2H6.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H6)

        layoutV2H7 = QtWidgets.QHBoxLayout()
        layoutV2H7.addWidget(noiseAmpLabel)
        layoutV2H7.addWidget(noiseAmpLineEdit)
        layoutV2H7.setAlignment(QtCore.Qt.AlignLeft)
        layoutV2.addLayout(layoutV2H7)

        layoutV2.addWidget(QtWidgets.QLabel(''))

        # header 3
        layoutV3 = QtWidgets.QVBoxLayout()
        layoutV3.addWidget(header3)

        layoutV3H1 = QtWidgets.QHBoxLayout()
        layoutV3H1.addWidget(saveModeLabel)
        layoutV3H1.addWidget(saveModeBox)
        layoutV3H1.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H1)

        layoutV3H2 = QtWidgets.QHBoxLayout()
        layoutV3H2.addWidget(saveT2Label)
        layoutV3H2.addWidget(saveT2Box)
        layoutV3H2.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H2)

        layoutV3H3 = QtWidgets.QHBoxLayout()
        layoutV3H3.addWidget(saveCollisionsLabel)
        layoutV3H3.addWidget(saveCollisionsBox)
        layoutV3H3.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H3)
        
        layoutV3H4 = QtWidgets.QHBoxLayout()
        layoutV3H4.addWidget(saveDecayLabel)
        layoutV3H4.addWidget(saveDecayBox)
        layoutV3H4.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H4)

        layoutV3H5 = QtWidgets.QHBoxLayout()
        layoutV3H5.addWidget(saveHistogramLabel)
        layoutV3H5.addWidget(saveHistogramBox)
        layoutV3H5.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H5)

        layoutV3H6 = QtWidgets.QHBoxLayout()
        layoutV3H6.addWidget(saveHistListLabel)
        layoutV3H6.addWidget(saveHistListBox)
        layoutV3H6.setAlignment(QtCore.Qt.AlignLeft)
        layoutV3.addLayout(layoutV3H6)       
        
        layoutV3.addWidget(QtWidgets.QLabel(''))

        # file configs
        fileLayout = QtWidgets.QVBoxLayout()
        fileHeaderLayout = QtWidgets.QHBoxLayout()
        fileHeaderLayout.addWidget(fileLabel)
        fileLayoutH1 = QtWidgets.QHBoxLayout()
        fileLayoutH1.addWidget(fileNameLabel)
        fileLayoutH1.addWidget(pfgseFileLineEdit)
        fileLayoutH1.addWidget(fileExtensionLabel)        
        fileLayoutH2 = QtWidgets.QHBoxLayout()
        fileLayoutH2.addWidget(saveButton)
        fileLayout.addLayout(fileHeaderLayout)
        fileLayout.addLayout(fileLayoutH1)
        fileLayout.addLayout(fileLayoutH2)
        

        # add to main layout
        mainLayout.addLayout(layoutV1)
        mainLayout.addLayout(layoutV2)
        mainLayout.addLayout(layoutV3)
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
        filename = CONFIG_PATH + self.procedures_qwidgets[_index]["configFilename"].text() + ".config"
        with open(filename, "w") as file:
            file.write("--- CPMG Configuration\n")
            file.write("METHOD: {}\n".format(self.procedures_qwidgets[_index]["method"].currentText()))
            file.write("D0: {}\n".format(self.procedures_qwidgets[_index]["D0"].text()))
            file.write("OBS_TIME: {}\n".format(self.procedures_qwidgets[_index]["expTime"].text()))
            
            file.write("-- LAPLACE INVERSION PARAMETERS\n")
            file.write("MIN_T2: {}\n".format(self.procedures_qwidgets[_index]["minT2"].text()))
            file.write("MAX_T2: {}\n".format(self.procedures_qwidgets[_index]["maxT2"].text()))
            file.write("USE_T2_LOGSPACE: {}\n".format(self.procedures_qwidgets[_index]["T2bins"].text()))
            file.write("NUM_T2_BINS: {}\n".format(self.procedures_qwidgets[_index]["logspace"].currentText()))
            file.write("MIN_LAMBDA: {}\n".format(self.procedures_qwidgets[_index]["minLambda"].text()))
            file.write("MAX_LAMBDA: {}\n".format(self.procedures_qwidgets[_index]["maxLambda"].text()))
            file.write("NUM_LAMBDAS: {}\n".format(self.procedures_qwidgets[_index]["numLambdas"].text()))
            file.write("PRUNE_NUM: {}\n".format(self.procedures_qwidgets[_index]["pruneNum"].text()))
            file.write("NOISE_AMP: {}\n".format(self.procedures_qwidgets[_index]["noiseAmp"].text()))

            file.write("-- SAVE MODE\n")
            file.write("SAVE_MODE: {}\n".format(self.procedures_qwidgets[_index]["saveMode"].currentText()))
            file.write("SAVE_T2: {}\n".format(self.procedures_qwidgets[_index]["saveT2"].currentText()))
            file.write("SAVE_COLLISIONS: {}\n".format(self.procedures_qwidgets[_index]["saveCollisions"].currentText()))
            file.write("SAVE_DECAY: {}\n".format(self.procedures_qwidgets[_index]["saveDecay"].currentText()))
            file.write("SAVE_HISTOGRAM: {}\n".format(self.procedures_qwidgets[_index]["saveHistogram"].currentText()))
            file.write("SAVE_HISTOGRAM_LIST: {}\n".format(self.procedures_qwidgets[_index]["saveHistList"].currentText()))

        self.parent.m_setup_tab.m_setup.procedure_paths[_index].setText(filename)
        self.parent.m_setup_tab.m_setup.procedure_paths[_index].setEnabled(True)  
        
        # close tab after save
        # realIdx = 0
        # for idx in range(_index):
        #     realIdx += 1
        #     if(self.open_tabs_names[idx] == "rwnmr" or self.open_tabs_names[idx]):
        #         realIdx += 1
        # self.closeTab(realIdx)          
        return

    def createNewGAConfigTab(self):
        return
    
    # @Slot()
    def getGADirPath(self):
        return
    
    # @Slot()
    def saveGAConfig(self):
        return