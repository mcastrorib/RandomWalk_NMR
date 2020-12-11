from PyQt5 import QtCore, QtGui, QtWidgets 

class setup_screen():
    def __init__(self, _parent, _widget):
        self.parent = _parent
        self.m_widget = _widget

        self.procedureCount = 0
        self.procedures = []
        self.procedure_actives = []
        self.procedure_paths = []
        self.procedure_tabs = []
        self.procedure_buttons = []
        self.procedure_layouts = []
        self.wrappedArgs = []

        titleFont=QtGui.QFont("Arial",15)
        titleFont.setBold(True)
        headerFont = QtGui.QFont("Arial",12)
        headerFont.setBold(True)

        # these are the app widgets connected to their slot methods
        self.titleLabel = QtWidgets.QLabel('RWNMR Setup')
        self.titleLabel.setFont(titleFont)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # set RW config widgets
        self.rwConfigLabel = QtWidgets.QLabel('rw configuration file:')
        self.rwConfigLabel.setFont(headerFont)
        self.rwConfigLineEdit = QtWidgets.QLineEdit('default')
        self.rwConfigLineEdit.setEnabled(False)
        self.rwConfigOpenButton = QtWidgets.QPushButton('Open')
        self.rwConfigOpenButton.clicked.connect(self.getRWConfigPath)
        self.rwConfigCreateButton = QtWidgets.QPushButton('Create')
        self.rwConfigCreateButton.clicked.connect(self.createRWConfigFile)

        # set uCT config widgets
        self.uctConfigLabel = QtWidgets.QLabel('uct configuration file:')
        self.uctConfigLabel.setFont(headerFont)
        self.uctConfigLineEdit = QtWidgets.QLineEdit('default')
        self.uctConfigLineEdit.setEnabled(False)
        self.uctConfigOpenButton = QtWidgets.QPushButton('Open')
        self.uctConfigOpenButton.clicked.connect(self.getUCTConfigPath)
        self.uctConfigCreateButton = QtWidgets.QPushButton('Create')
        self.uctConfigCreateButton.clicked.connect(self.createUCTConfigFile)      

        # Set procedures zone
        self.procedureLabel = QtWidgets.QLabel("Procedures")
        self.procedureLabel.setFont(titleFont)
        self.procedureLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        
        # Set add procedure
        procedureOptions = ['cpmg', 'pfgse', 'ga']
        self.newProcedureLabel = QtWidgets.QLabel('New procedure:')
        # self.newProcedureLabel.setFont(headerFont)
        self.newProcedureBox = QtWidgets.QComboBox()
        self.newProcedureBox.addItems(procedureOptions)
        self.addProcedureButton = QtWidgets.QPushButton("Add")
        self.addProcedureButton.clicked.connect(lambda: self.addProcedure(self.newProcedureBox.currentText())) 
        

        # set the layouts
        self.mainLayout = QtWidgets.QVBoxLayout(self.m_widget)        
        self.mainLayout.addWidget(self.titleLabel)

        # essentials layout
        essentialsLayout = QtWidgets.QVBoxLayout()

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

        essentialsLayout.addLayout(layoutH2a)
        essentialsLayout.addLayout(layoutH2b)
        essentialsLayout.addLayout(layoutH3a)
        essentialsLayout.addLayout(layoutH3b)
        essentialsLayout.addWidget(QtWidgets.QLabel(''))
        

        # set Procedures title layout
        layout3 = QtWidgets.QHBoxLayout()
        layout3.addWidget(self.procedureLabel)

        self.addLayout = QtWidgets.QVBoxLayout()
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addLayout.addWidget(QtWidgets.QLabel(''))
        self.addHBLayout = QtWidgets.QHBoxLayout()
        self.addHBLayout.addWidget(self.newProcedureLabel)
        self.addHBLayout.addWidget(self.newProcedureBox)        
        self.addHBLayout.addWidget(self.addProcedureButton)
        self.addLayout.addLayout(self.addHBLayout)

        # adding layouts to main
        self.mainLayout.addLayout(essentialsLayout)
        self.mainLayout.addLayout(layout3)
        self.mainLayout.addLayout(self.addLayout)     
        self.mainLayout.setAlignment(QtCore.Qt.AlignTop)  
        return

    # @Slot()
    def addProcedure(self, procedure):
        print("Adding", procedure, "procedure")
        headerFont = QtGui.QFont("Arial", 12)
        headerFont.setBold(True)
        index = self.procedureCount            
        self.procedureCount += 1

        # set RW config widgets
        newLabel1 = QtWidgets.QLabel(procedure + ' configuration file:')
        newLabel1.setFont(headerFont)
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
        self.procedure_actives.append(True)
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
        self.addHBLayout.addWidget(self.newProcedureLabel)
        self.addHBLayout.addWidget(self.newProcedureBox)        
        self.addHBLayout.addWidget(self.addProcedureButton)
        self.addLayout.addLayout(self.addHBLayout) 

        # adding layouts to main
        self.mainLayout.addLayout(self.addLayout) 
        return 

    # @Slot()
    def removeProcedure(self, index):
        self.procedure_actives[index] = False
        self.deleteBox(self.procedure_layouts[index])
        return

    def build(self):
        self.wrapArgs()
        return self.wrappedArgs
    
    def wrapArgs(self):
        args = []
        cmd = ''
        path = ''

        cmd  = "-rwconfig"
        path = str(self.rwConfigLineEdit.text())
        if(path != 'default'):
            args.append([cmd, path])

        cmd  = "-uctconfig"
        path = str(self.uctConfigLineEdit.text())
        if(path != 'default'):
            args.append([cmd, path])
        
        numberOfProcedures = len(self.procedures)
        for idx in range(numberOfProcedures):
            if(self.procedure_actives[idx] == True):
                cmd = self.procedures[idx]
                flag = "-config"
                path = self.procedure_paths[idx].text()
                if(path != "default"):
                    args.append([cmd, flag, path])
                else:
                    args.append([cmd])
        
        self.wrappedArgs = args
        return
        



