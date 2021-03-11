import numpy as np

def get_porosity(a,r):
	x = r/a
	result = 0.0
	if(x<=0.5):
		result = 1.0 - ((4.0/3.0) * np.pi * (x**3))
	else:
		result = 1 + (1.0/4.0)*np.pi - 3.0*np.pi*x*x + (8.0/3.0)*np.pi*x**3
	
	return result
	
def get_SVp(a,r):
	x = r/a
	result = 0.0
	if(x<0.5):
		result = (4.0 * np.pi * x**2) / (1.0 - (4.0/3.0)*np.pi*x**3)
	else:
		result = (2.0 * np.pi * x * (3.0 - 4.0*x)) / (1 + (1.0/4.0)*np.pi - 3.0*np.pi*x*x + (8.0/3.0)*np.pi*x**3)
	
	return result
