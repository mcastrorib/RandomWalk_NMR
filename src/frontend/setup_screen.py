from PyQt5 import QtCore, QtGui, QtWidgets 

class setup_screen():
    def __init__(self, _parent, _widget):
        self.parent = _parent
        self.m_widget = _widget

        self.procedures = []
        self.procedure_paths = []
        self.procedure_tabs = []
        self.procedure_buttons = []
        self.procedure_layouts = []

        # these are the app widgets connected to their slot methods
        self.titleLabel = QtWidgets.QLabel('--- RWNMR SETUP ---')
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # set RW config widgets
        self.rwConfigLabel = QtWidgets.QLabel('rw configuration file:')
        self.rwConfigLineEdit = QtWidgets.QLineEdit('default')
        self.rwConfigLineEdit.setEnabled(False)
        self.rwConfigOpenButton = QtWidgets.QPushButton('Open')
        self.rwConfigOpenButton.clicked.connect(self.getRWConfigPath)
        self.rwConfigCreateButton = QtWidgets.QPushButton('Create')
        self.rwConfigCreateButton.clicked.connect(self.createRWConfigFile)

        # set uCT config widgets
        self.uctConfigLabel = QtWidgets.QLabel('uct configuration file:')
        self.uctConfigLineEdit = QtWidgets.QLineEdit('default')
        self.uctConfigLineEdit.setEnabled(False)
        self.uctConfigOpenButton = QtWidgets.QPushButton('Open')
        self.uctConfigOpenButton.clicked.connect(self.getUCTConfigPath)
        self.uctConfigCreateButton = QtWidgets.QPushButton('Create')
        self.uctConfigCreateButton.clicked.connect(self.createUCTConfigFile)      

        # Set procedures zone
        self.procedureLabel = QtWidgets.QLabel("-- Procedures --")
        self.procedureLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # Set add procedure
        procedureOptions = ['cpmg', 'pfgse', 'ga']
        self.newProcedureBox = QtWidgets.QComboBox()
        self.newProcedureBox.addItems(procedureOptions)
        self.addProcedureButton = QtWidgets.QPushButton("Add")
        self.addProcedureButton.clicked.connect(lambda: self.addProcedure(self.newProcedureBox.currentText())) 
        

        # set the layouts
        self.mainLayout = QtWidgets.QVBoxLayout(self.m_widget)        
        self.mainLayout.addWidget(self.titleLabel)

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

        # set Procedures title layout
        layout3 = QtWidgets.QHBoxLayout()
        layout3.addWidget(self.procedureLabel)

        self.addLayout = QtWidgets.QVBoxLayout()
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addHBLayout = QtWidgets.QHBoxLayout()
        self.addHBLayout.addWidget(QtWidgets.QLabel('New procedure:'))
        self.addHBLayout.addWidget(self.newProcedureBox)        
        self.addHBLayout.addWidget(self.addProcedureButton)
        self.addLayout.addLayout(self.addHBLayout)

        # adding layouts to main
        self.mainLayout.addLayout(layoutH2a)
        self.mainLayout.addLayout(layoutH2b) 
        self.mainLayout.addLayout(layoutH3a)
        self.mainLayout.addLayout(layoutH3b) 
        self.mainLayout.addLayout(layout3)
        self.mainLayout.addLayout(self.addLayout)     
        self.mainLayout.setAlignment(QtCore.Qt.AlignTop)  

        

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
    def addProcedure(self, procedure):
        print("Adding", procedure, "procedure")

        index = len(self.procedures)            

        # set RW config widgets
        newLabel1 = QtWidgets.QLabel(procedure + ' configuration file:')
        newLineEdit = QtWidgets.QLineEdit('default')
        newLineEdit.setEnabled(False)
        newOpenButton = QtWidgets.QPushButton('Open')
        newOpenButton.clicked.connect(lambda: self.getProcedureConfigPath(index))
        newCreateButton = QtWidgets.QPushButton('Create')
        newCreateButton.clicked.connect(lambda: self.createProcedureConfigFile(index))
        newRemoveButton = QtWidgets.QPushButton('Remove')
        newRemoveButton.clicked.connect(lambda: self.removeProcedure(index))
        

        # set new config layout
        layoutH2a = QtWidgets.QHBoxLayout()
        layoutH2a.addWidget(newLabel1)

        layoutH2b = QtWidgets.QHBoxLayout()
        layoutH2b.addWidget(newLineEdit)
        layoutH2b.addWidget(newOpenButton)
        layoutH2b.addWidget(newCreateButton) 
        layoutH2b.addWidget(newRemoveButton) 
        
        blankLayout = QtWidgets.QVBoxLayout()
        blankLayout.addWidget(QtWidgets.QLabel(''))   

        procedureLayout = QtWidgets.QVBoxLayout()
        procedureLayout.addLayout(layoutH2a)
        procedureLayout.addLayout(layoutH2b) 
        procedureLayout.addLayout(blankLayout)     

        # adding layouts to main
        self.mainLayout.addLayout(procedureLayout)
        self.deleteBox(self.addLayout)
        self.createAddProcedureLayout()
        self.mainLayout.setAlignment(QtCore.Qt.AlignTop)  

        self.procedures.append(procedure)
        self.procedure_paths.append(newLineEdit)
        self.procedure_tabs.append(None)
        self.procedure_buttons.append(newCreateButton)
        self.procedure_layouts.append(procedureLayout)
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

    # @Slot()
    def createRWConfigFile(self):
        self.parent.addConfigTab('rwnmr')
        return
    
    # @Slot()
    def createUCTConfigFile(self):
        self.parent.addConfigTab('uct')
        return

    # @Slot()
    def getProcedureConfigPath(self, index):
        print("opening", self.procedures[index], "config file", index)
        options = QtWidgets.QFileDialog.Options()
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self.parent, "Open", "","Config Files (*.config);;Config Files (*.config)", options=options)
        if filepath != '':
            print(filepath)
            self.procedure_paths[index].setText(filepath)  
        return

    # @Slot()
    def createProcedureConfigFile(self, index):
        print("creating ", str(self.procedures[index]), "config file", index)
        self.parent.addConfigTab(str(self.procedures[index]), index)
        self.procedure_buttons[index].setEnabled(False)
        return
    
    # Method
    def deleteItemsOfLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self.deleteItemsOfLayout(item.layout())
    
    # Method
    def deleteBox(self, box):
        for i in range(self.mainLayout.count()):
            layout_item = self.mainLayout.itemAt(i)
            if layout_item.layout() == box:
                self.deleteItemsOfLayout(layout_item.layout())
                self.mainLayout.removeItem(layout_item)
                break
    
    # Method
    def createAddProcedureLayout(self):
        # Set add procedure
        procedureOptions = ['cpmg', 'pfgse', 'ga']
        self.newProcedureBox = QtWidgets.QComboBox()
        self.newProcedureBox.addItems(procedureOptions)
        self.addProcedureButton = QtWidgets.QPushButton("Add")
        self.addProcedureButton.clicked.connect(lambda: self.addProcedure(self.newProcedureBox.currentText()))

        self.addLayout = QtWidgets.QVBoxLayout()
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addHBLayout = QtWidgets.QHBoxLayout()
        self.addHBLayout.addWidget(QtWidgets.QLabel('New procedure:'))
        self.addHBLayout.addWidget(self.newProcedureBox)        
        self.addHBLayout.addWidget(self.addProcedureButton)
        self.addLayout.addLayout(self.addHBLayout) 

        # adding layouts to main
        self.mainLayout.addLayout(self.addLayout) 
        return 

    # @Slot()
    def removeProcedure(self, index):
        self.deleteBox(self.procedure_layouts[index])
        return