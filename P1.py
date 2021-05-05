import numpy as np
import math
import matplotlib.pyplot as plt
import time

def dn_sin(n):
	'''
	Compute the n^th derivative of sin(x) at x=0

	input:
		n - int: the order of the derivative to compute
	output:
		float nth derivative of sin(0)
	purpose:
		problem 1) Write a python function dn_sin0(n) to evaluate the  nth  derivate of  sin(0). 
		This should provide a chance to use the if-elif-else control structure.
	'''

	x = 0
	if n%4 == 1: return np.cos(x)
	elif n%4 == 2: return -np.sin(x)
	elif n%4 == 3: return -np.cos(x)
	else: return np.sin(x)
	# pass

def taylor_sin(x, n):
	'''
	Evaluate the Taylor series of sin(x) about x=0 neglecting terms of order x^n
	
	input:
		x - float: argument of sin
		n - int: number of terms of the taylor series to use in approximation
	output:
		float value computed using the taylor series truncated at the nth term
	'''
	y = 0
	for k in range(n):
		y += dn_sin(k) * (x**k) / math.factorial(k) 
	return y
	# pass


def measure_diff(ary1, ary2):
	'''
	Compute a scalar measure of difference between 2 arrays

	input:
		ary1 - numpy array of float values
		ary2 - numpy array of float values
	output:
		a float scalar quantifying difference between the arrays
	'''
	diff = ary1 - ary2
	return np.sum(np.absolute(diff))
	# pass


def escape(cx, cy, dist,itrs, x0=0, y0=0):
	'''
	Compute the number of iterations of the logistic map, 
	f(x+j*y)=(x+j*y)**2 + cx +j*cy with initial values x0 and y0 
	with default values of 0, to escape from a cirle centered at the origin.

	inputs:
		cx - float: the real component of the parameter value
		cy - float: the imag component of the parameter value
		dist: radius of the circle
		itrs: int max number of iterations to compute
		x0: initial value of x; default value 0
		y0: initial value of y; default value 0
	returns:
		an int scalar interation count
	'''
	x = x0
	y = y0
	r = 0
	for i in range(itrs):
		r = math.sqrt(x**2 + y**2)
		if dist > r:
			x_n = x**2 - y**2 + cx 
			y_n = 2*x*y + cy
			x = x_n
			y = y_n
		else: return i
	return 0
	# pass
	# 
    # xtemp := x×x - y×y + x0
    # y := 2×x×y + y0
    # x := xtemp
    # iteration := iteration + 1

def mandelbrot(cx,cy,dist,itrs):
	'''
	Compute escape iteration counts for an array of parameter values

	input:
		cx - array: 1d array of real part of parameter
		cy - array: 1d array of imaginary part of parameter
		dist - float: radius of circle for escape
		itrs - int: maximum number of iterations to compute
	output:
		a 2d array of iteration count for each parameter value (indexed pair of values cx, cy)
	'''
	#create a 2D numpy array (init to zero) to store n_ss values at each of the m values of r.
	# x = numpy.zeros([m,n_ss])
	f = np.zeros([len(cx), len(cy)])
	for i in range(len(cx)):
		for j in range(len(cy)):
			f[i][j] = escape(cx[i], cy[j], dist, itrs)
	return f
	# pass


if __name__ == '__main__':
	# problem 1
	print('Problem 1')
	# print(dn_sin.__doc__)
	print(dn_sin(1)) # first: 1


	#Problem 2/3
	print('Problem 2/3')	
	x = np.linspace(-2, 2, 100)
	for n in range(2,16,2): #for n in len(range(2,16,2)):
		#compute taylor series
		y = []
		for i in range(len(x)):
			x_i = x[i]
			y.append(taylor_sin(x_i, n)) #collect list of y in n's row
		#plot taylor series	
		plt.plot(x, y, label = n) # colect plot for n_th terms
	plt.plot(x, np.sin(x), label = 'sin(x)')
	plt.figure(1)
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Taylor series (n = num of term)')	
	plt.show()
	

	#Problem 4
	print('Problem 4')	
	# diff measure to be less than 1e-2
	# use diff fn.
	# evaluate your functions at 50 points equally spaced across the interval [0,pi/4]
	x = np.linspace(0, np.pi/4, 50)
	y0 = np.sin(x)
	diff = 100
	n = 0
	while diff > 10**-2:
		n += 1	
		y = []
		for i in range(len(x)):
			x_i = x[i]
			y.append(taylor_sin(x_i, n))
		diff = measure_diff(y, y0)
	print('current difference = ', diff)
	print('truncation order = ', n)


	# problem 5
	print('Problem 5')	
	nx = 512
	ny = 512
	unit_away = 2.5
	i = 256

	x = np.linspace(-2, 1, nx)
	y = np.linspace(-1.5, 1.5, ny)
	# print time required to comput 512 x 512
	t0 = time.time()
	f = mandelbrot(x, y, unit_away, i)
	t1 = time.time() - t0
	print('Time took for calcualting mandelbrot (second) = ', t1)
	print("1(yellow) part is where it can escape.")
	print('0(purbpe) part where it cannot escape')

	for i in range(len(x)):
		for j in range(len(y)):
			if f[i][j] > 0: f[i][j] = 1

	# plot
	plt.figure(2)
	plt.imshow(f.T, extent = (-2.5, 2.5, -2.5, 2.5))
	plt.xlabel('')
	plt.ylabel('')
	plt.title('Mandelbrot graph')	
	plt.show()
	# plt.colorbar()

	# A map is a function: it takes an input value (and possibly paramter values) and returns an output value
	# rewrite the map
