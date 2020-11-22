import numpy as np
import math

class NMR_walker():
	def __init__(self):
		# initial positions
		self.pos_x0 = 0
		self.pos_y0 = 0
		self.pos_z0 = 0

		# final/current positions
		self.pos_x = 0
		self.pos_y = 0
		self.pos_z = 0

		# collisions count
		self.collisions = 0
		self.xirate = 0.0

		# RNG initial and current/final seed 
		self.rng_initial_seed = None
		self.rng_seed = None
		return

	def print_info(self):
		print('initial position: ({},{},{})'.format(self.get_x0(), self.get_y0(), self.get_z0()))
		print('current position: ({},{},{})'.format(self.get_x(), self.get_y(), self.get_z()))
		print('collisions: {}'.format(self.get_collisions()))
		print('xi rate: {}'.format(self.get_xirate()))
		print('initial rng seed: {}'.format(self.get_rng_initial_seed()))
		print('current rng seed: {}'.format(self.get_rng_seed()))
		return

	def set_x0(self, _x0):
		self.pos_x0 = _x0
		return

	def get_x0(self):
		return self.pos_x0

	def set_y0(self, _y0):
		self.pos_y0 = _y0
		return

	def get_y0(self):
		return self.pos_y0

	def set_z0(self, _z0):
		self.pos_z0 = _z0
		return

	def get_z0(self):
		return self.pos_z0

	def set_x(self, _x):
		self.pos_x = _x
		return

	def get_x(self):
		return self.pos_x

	def set_y(self, _y):
		self.pos_y = _y
		return

	def get_y(self):
		return self.pos_y

	def set_z(self, _z):
		self.pos_z = _z
		return

	def get_z(self):
		return self.pos_z

	def set_collisions(self, _collisions):
		self.collisions = _collisions
		return

	def get_collisions(self):
		return self.collisions

	def set_xirate(self, _xirate):
		self.xirate = _xirate
		return

	def get_xirate(self):
		return self.xirate

	def set_rng_initial_seed(self, _seed):
		self.rng_initial_seed = _seed
		return

	def get_rng_initial_seed(self):
		return self.rng_initial_seed 

	def set_rng_seed(self, _seed):
		self.rng_seed = _seed

	def get_rng_seed(self):
		return self.rng_seed 

	def reset_rng_seed(self):
		self.rng_seed = self.get_rng_initial_seed()
		return




