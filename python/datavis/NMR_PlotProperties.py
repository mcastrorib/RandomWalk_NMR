class NMR_PlotProperties():
	def __init__(self):
		self.title = ''
		self.xlabel = ''
		self.ylabel = ''
		self.xlim = []
		self.ylim = []
		self.fig_size = [10,10]
		self.dpi = 100
		self.show = True
		return

	def setTitle(self, title):
		self.title = title
		return

	def setXLabel(self, xlabel):
		self.xlabel = xlabel
		return

	def setYLabel(self, ylabel):
		self.ylabel = ylabel
		return

	def setXLim(self, xlim):
		self.xlim = xlim
		return

	def setYLim(self, ylim):
		self.ylim = ylim
		return

	def setFigureSize(self, size):
		if(len(size) == 2):
			self.fig_size = [size[0], size[1]]
		else:
			print("could not change figure size.")
		return

	def setDPI(self, dpi):
		self.dpi = dpi
		return

	def setShow(self, bvalue):
		if(bvalue):
			self.show = True
		else:
			self.show = False
		return 

