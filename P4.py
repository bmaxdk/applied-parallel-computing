# @cuda.reduce
def max_kernel(a, b):
    return max(a,b)

# @cuda.reduce
def sum_kernel(a, b):
    return a + b

# @cuda.jit
def heat_step(u, out, stencil, t):
	'''
	u: input device array
	out: output device array
	stencil: derivative stencil coefficients
	t: current time step
	'''
	pass

# @cuda.jit
def integrate_kernel(y, out, quad):
	'''
	y: input device array
	out: output device array
	quad: quadrature stencil coefficients
	'''
	pass

def integrate(y, quad):
	'''
	y: input array
	quad: quadrature stencil coefficients
	'''
	pass

# @cuda.jit
def monte_carlo_kernel_sphere_intertia(rng_states, iters, out):
	'''
	rng_states: rng state array generated from xoroshiro random number generator
	iters: number of monte carlo sample points each thread will test
	out: output array
	'''
	pass

# @cuda.jit
def monte_carlo_kernel_sphere_vol(rng_states, iters, out):
	pass

# @cuda.jit
def monte_carlo_kernel_shell_intertia(rng_states, iters, out):
	pass

# @cuda.jit
def monte_carlo_kernel_shell_vol(rng_states, iters, out):
	pass

def monte_carlo(threads, blocks, iters, kernel, seed = 1):
	'''
	threads: number of threads to use for the kernel
	blocks: number of blocks to use for the kernel
	iters: number of monte carlo sample points each thread will test 
	kernel: monte_carlo kernel to use
	seed: seed used when generating the random numbers (if the seed is left at one the number generated will be the same each time)
	'''
	pass

# @cuda.jit(device = True)
def chi(f, levelset):
	'''
	f: function value
	levelset: surface levelset
	'''
	return f <= levelset

# @cuda.jit
def grid_integrate_sphere_intertia(y, out, stencil):
	'''
	y: input device array
	out: output device array
	stencil: derivative stencil coefficients
	'''
	pass

# @cuda.jit
def grid_integrate_sphere_vol(y, out, stencil):
	pass

# @cuda.jit
def grid_integrate_shell_intertia(y, out, stencil):
	pass

# @cuda.jit
def grid_integrate_shell_vol(y, out, stencil):
	pass

def grid_integrate(kernel):
	'''
	kernel: grid integration kernel to use
	'''
	pass

if __name__ == '__main__':
	pass


