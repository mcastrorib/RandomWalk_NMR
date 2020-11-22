import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as spl 
import math


rho = 500	# in um/s (micrometer per second)
a = 10		# in um (micrometer)
D = 2500	# in um²/s (micrometer squared per second)
rhoa_D = rho*a/D
print(rhoa_D)

size=10000
x = np.linspace(-25,25,size)

ftan_x = np.zeros(size)
for idx in range(size):
	ftan_x[idx] = np.tan(x[idx])

fcot_x = np.zeros(size)
for idx in range(size):
	fcot_x[idx] = 1.0/ftan_x[idx]

fx = np.zeros(size)
for idx in range(size):
	fx[idx] = x[idx]

fXI = np.zeros(size)
for idx in range(size):
	fXI[idx] = fx[idx]*ftan_x[idx] - rhoa_D

fXI2 = np.zeros(size)
for idx in range(size):
	fXI2[idx] = fx[idx]*fcot_x[idx] + rhoa_D

order = 1
Jn = np.zeros(size)
dJn = np.zeros(size)
for idx in range(size):
	Jn[idx] = spl.jn(order, x[idx])
	dJn[idx] = spl.jvp(order, x[idx], 1)

sph_Jn = np.zeros(size)
sph_dJn = np.zeros(size)
for idx in range(size):
	sph_Jn[idx] = spl.spherical_jn(order, x[idx])
	sph_dJn[idx] = spl.spherical_jn(order, x[idx], 1) 

fXI3 = np.zeros(size)
fXI3b = np.zeros(size)
for idx in range(size):
	fXI3[idx] = fx[idx]*dJn[idx]/Jn[idx] + rhoa_D
	fXI3b[idx] = fx[idx]*dJn[idx]/Jn[idx] + 2

fXI4 = np.zeros(size)
for idx in range(size):
	fXI4[idx] = fx[idx]*sph_dJn[idx]/sph_Jn[idx] + rhoa_D



# set plot size
# plt.plot(x, fx, label="g(x)=x")

# plt.plot(x, ftan_x, label="f(x)=tan(x)")
# plt.plot(x, fXI, '-', label="h1(x)=f.g - Ma/D")

# plt.plot(x, fcot_x, label="f(x)=cot(x)")
# plt.plot(x, fXI2,'-', label="h2(x)=f.g + Ma/D")

# plt.plot(x, Jn,'--', label="Bessel Function - order 0")
plt.plot(x, dJn,'--', label="Derivative of Bessel Function - order 0")
# plt.plot(x, fXI3,'-', label="h2(x)=xJn'(x)/Jn(x) + Ma/D")
# plt.plot(x, fXI3b,'-', label="h2(x)=xJn'(x)/Jn(x) + Ma/D")

# plt.plot(x, sph_Jn,'-', label="Spherical Bessel Function - order 0")
# plt.plot(x, sph_dJn,'--', label="Derivative of Spherical Bessel Function - order 0")
# plt.plot(x, fXI4,'-', label="h2(x)=xJn'(x)/Jn(x) + Ma/D")



# Plot curve legend
# plt.legend(loc="best")

# set y axis limit
plt.xlim(0, 20)
plt.ylim(-5, 5)

# Show the major grid lines with dark grey lines
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.5)

# Show the minor grid lines with very faint and almost transparent grey lines
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)

# Show image
plt.show()
