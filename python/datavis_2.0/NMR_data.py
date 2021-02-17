import numpy as np
import math

class NMR_data():
	def __init__(self):
		self.x_data = []
		self.y_data = []
		self.label = ''
		self.marker = '-'
		self.color = 'blue'
		self.show = True
		return

	def setXData(self, data):
		self.x_data = data
		return

	def setYData(self, data):
		self.y_data = data
		return

	def setLabel(self, label):
		self.label = label
		return

	def setMarker(self, marker):
		self.marker = marker
		return

	def setColor(self, color):
		self.color = color
		return

	def setShow(self, bvalue):
		if(bvalue):
			self.show = True
		else:
			self.show = False
		return 

	def print(self):
		print("x: ", self.x_data)
		print("y: ", self.y_data)
		print("label: ", self.label)
		print("marker: ", self.marker)
		print("color: ", self.color)
		print("show: ", self.show)
		return
