import numpy as np
import math

class LeastSquaresRegression:
	def __init__(self):
		# numeric data
		self.points = 0
		self.X = []
		self.Y = []
		self.meanX = 0.0
		self.meanY = 0.0
		self.A = 0.0
		self.B = 0.0

		# control flags
		self.X_isSet = False
		self.Y_isSet = False
		self.meanX_isSet = False
		self.meanY_isSet = False
		self.A_isSet = False
		self.B_isSet = False
		self.solved = False
		return

	# ------------
	# -- points --
	# ------------
	def set_points(self, _points):
		self.points = _points
		return

	def get_points(self):
		return self.points

	# --------------
	# -- vector X --
	# --------------
	def allocateX(self):
		self.X = np.zeros(self.points)
		return

	def setX(self, data):
		if(len(data) >= self.get_points()):
			self.allocateX()
			for idx in range(self.get_points()):
				self.X[idx] = data[idx]
			self.set_X_as_set(True)
			self.set_meanX_as_set(False)
			self.set_A_as_set(False)
			self.set_B_as_set(False)
			self.set_as_solved(False)
		else:
			print('error: input data length is lower than expected.')
		return

	def set_X_as_set(self, bvalue: bool):
		self.X_isSet = bvalue
		return

	def X_is_set(self):
		return self.X_isSet

	def getX(self):
		return self.X

	def getX(self, index):
		if(self.X_is_set() and index < self.get_points()):
			return self.X[index]
		else:
			print('error: index out of range')
			return

	# --------------
	# -- vector Y --
	# --------------	
	def allocateY(self):
		self.Y = np.zeros(self.points)
		return

	def setY(self, data):
		if(len(data) >= self.get_points()):
			self.allocateY()
			for idx in range(self.get_points()):
				self.Y[idx] = data[idx]
			self.set_Y_as_set(True)
			self.set_meanY_as_set(False)
			self.set_A_as_set(False)
			self.set_B_as_set(False)
			self.set_as_solved(False)
		else:
			print('error: input data length is lower than expected.')
		return

	def set_Y_as_set(self, bvalue: bool):
		self.Y_isSet = bvalue
		return

	def Y_is_set(self):
		return self.Y_isSet

	def getY(self):
		return self.Y

	def getY(self, index):
		if(self.Y_is_set() and index < self.get_points()):
			return self.Y[index]
		else:
			print('error: index out of range')
			return

	# -----------
	# -- meanX --
	# -----------
	def set_meanX(self):
		if(self.X_is_set()):
			sumX = 0.0
			for idx in range(self.get_points()):
				sumX += self.getX(idx)
			self.meanX = sumX / float(self.get_points())
			self.set_meanX_as_set(True)
		else:
			print('error: vector X is not set.')
		return

	def set_meanX_as_set(self, bvalue: bool):
		self.meanX_isSet = bvalue
		return

	def meanX_is_set(self):
		return self.meanX_isSet

	def get_meanX(self):
		if(self.meanX_is_set()):
			return self.meanX
		else:
			print('error: meanX is not set.')
			return 

	# -----------
	# -- meanY --
	# -----------
	def set_meanY(self):
		if(self.Y_is_set()):
			sumY = 0.0
			for idx in range(self.get_points()):
				sumY += self.getY(idx)
			self.meanY = sumY / float(self.get_points())
			self.set_meanY_as_set(True)
		else:
			print('error: vector Y is not set.')
		return

	def set_meanY_as_set(self, bvalue: bool):
		self.meanY_isSet = bvalue
		return

	def meanY_is_set(self):
		return self.meanY_isSet

	def get_meanY(self):
		if(self.meanY_is_set()):
			return self.meanY
		else:
			print('error: meanY is not set.')
			return 

	# -----------
	# ---- A ----
	# -----------
	def set_A(self):
		if(self.meanY_is_set() and self.meanX_is_set() and self.B_is_set()):
			self.A = self.get_meanY() - (self.get_B() * self.get_meanX())
			self.set_A_as_set(True)
		else:
			print('error: at least one of meanX, meanY or B is not set.')
		return

	def set_A_as_set(self, bvalue: bool):
		self.A_isSet = bvalue
		return

	def A_is_set(self):
		return self.A_isSet

	def get_A(self):
		if(self.A_is_set()):
			return self.A
		else:
			print('error: A is not set.')
			return 

	# -----------
	# ---- B ----
	# -----------
	def set_B(self):
		if(self.X_is_set() and self.Y_is_set() and self.meanX_is_set() and self.meanY_is_set()):
			
			dividend = 0.0
			for idx in range(self.get_points()):
				dividend += self.getX(idx) * (self.getY(idx) - self.get_meanY())

			divisor = 0.0
			for idx in range(self.get_points()):
				divisor += self.getX(idx) * (self.getX(idx) - self.get_meanX())

			if(divisor != 0.0):
				self.B = dividend/divisor
				self.set_B_as_set(True)
			else:
				print('error: divisor is zero.')
		else:
			print('error: at least one X, Y, meanX or meanY is not set.')
		return

	def set_B_as_set(self, bvalue: bool):
		self.B_isSet = bvalue
		return

	def B_is_set(self):
		return self.B_isSet

	def get_B(self):
		if(self.B_is_set()):
			return self.B
		else:
			print('error: B is not set.')
			return 

	def set_as_solved(self, bvalue: bool):
		self.solved = bvalue
		return

	def is_solved(self):
		return self.solved


	# ---------------
	# -- interface --
	# ---------------
	def config(self, x_data, y_data, points=0):
		if(points == 0):
			points = min(len(x_data), len(y_data))
			self.set_points(points)
			self.setX(x_data)
			self.setY(y_data)
		else:
			self.set_points(points)
			self.setX(x_data)
			self.setY(y_data)

	def solve(self):
		if(self.X_is_set() and self.Y_is_set()):
			self.set_meanX()
			self.set_meanY()
			self.set_B()
			self.set_A()
			self.set_as_solved(True)
		else:
			print('error: at least one of X or Y is not set.')

	def results(self):
		if(self.is_solved()):
			results = {}

			results["meanX"] = self.get_meanX()
			results["meanY"] = self.get_meanY()
			results["A"] = self.get_A()
			results["B"] = self.get_B()

			return results
		else:
			print('error: not solved yet.')

		



	


