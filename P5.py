import numpy as np
##Import other packages here##
import cupy as cp
import matplotlib.pyplot as plt
from cupy.fft import fftfreq, fft, ifft
import math
from cupyx import scipy

# cupy source
# https://docs-cupy.chainer.org/_/downloads/en/v7.0.0/pdf/



# 1a) install cupy
# fft.fftfreq to get frequency array corresponding to number of sample points
# cupy.fft.fftfreq #Return the FFT sample frequencies.

# compute the fft of the full signal
# cupy.fft.fft #Compute the one-dimensional FFT.

# invert the filtered fft with numpy.fft.ifft
# cupy.fft.ifft #Compute the one-dimensional inverse FFT.

# move the array to the host.
# plot -> plt.plot(cp.asnumpy(x), cp.asnumpy(y))

def cupy_fft():
    
    """
    Modify the code below so it calls the cupy fft function
    """

    pts = 1000
    L = 100
    w0 = 2.0 * cp.pi/L
    n1, n2, n3 = 10.0, 20.0, 30.0
    a1, a2, a3 = 1., 2., 3.

    #create signal data with 3 frequency components
    x = cp.linspace(0,L,pts)
    y1 = a1*cp.cos(n1*w0*x)
    y2 = a2*cp.sin(n2*w0*x)
    y3 = a3*cp.sin(n3*w0*x)
    y = y1 + y2 + y3

    #create signal including only 2 components
    y12 = y1 + y2

    #analytic derivative of signal
    dy = w0*(-n1*a1*cp.sin(n1*w0*x)
            +n2*a2*cp.cos(n2*w0*x)
            +n3*a3*cp.cos(n3*w0*x) )

    #use fft.fftfreq to get frequency array corresponding to number of sample points
    freqs = fftfreq(pts)
    #compute number of cycles and radians in sample window for each frequency
    nwaves = freqs*pts
    nwaves_2pi = w0*nwaves

    # compute the fft of the full signal
    fft_vals = fft(y)

    #mask the negative frequencies
    mask = freqs>0
    #double count at positive frequencies
    fft_theo = 2.0 * cp.abs(fft_vals/pts)
    #plot fft of signal
    plt.xlim((0,50))
    plt.xlabel('cycles in window')
    plt.ylabel('original amplitude')
    plt.plot(cp.asnumpy(nwaves[mask]), cp.asnumpy(fft_theo[mask]))
    plt.show()

    #create a copy of the original fft to be used for filtering
    fft_new = cp.copy(fft_vals)
    #filter out y3 by setting corr. frequency component(s) to zero
    fft_new[cp.abs(nwaves)==n3] = 0.
    #plot fft of filtered signal
    plt.xlim((0,50))
    plt.xlabel('cycles in window')
    plt.ylabel('filtered amplitude')
    plt.plot(cp.asnumpy(nwaves[mask]), cp.asnumpy(2.0*cp.abs(fft_new[mask]/pts)))
    plt.show()

    #invert the filtered fft with numpy.fft.ifft
    filt_data = cp.real(ifft(fft_new))
    #plot filtered data and compare with y12
    plt.plot(cp.asnumpy(x), cp.asnumpy(y12), label='original signal')
    plt.plot(cp.asnumpy(x), cp.asnumpy(filt_data), label='filtered signal')
    plt.xlim((0,50))
    plt.legend()
    plt.show()

    #multiply fft by 2*pi*sqrt(-1)*frequency to get fft of derivative
    dy_fft = 1.0j*nwaves_2pi*fft_vals
    #invert to reconstruct sampled values of derivative
    dy_recon = cp.real(ifft(dy_fft))
    #plot reconstructed derivative and compare with analuytical version
    plt.plot(cp.asnumpy(x), cp.asnumpy(dy), label='exact derivative')
    plt.plot(cp.asnumpy(x), cp.asnumpy(dy_recon), label='fft derivative')
    plt.xlim((0,50))
    plt.legend()
    plt.show()

def cupy_filter():
    
    """
    Implement code below to:
    Create noise consisting of an array of pts random values chosen from a uniform distribution over the interval [−3,3]
    Create a noisy signal by adding noise to the original signal: y_n = y + noise
    Compute and plot the frequency content of the noisy signal.
    Create and apply an appropriate filter to suppress noise in the frequency domain.
    Invert the filtered fft to obtain a "denoised signal".
    Plot and compare the original, noisy, and denoised signals.
    """

    # pass
    pts = 1000
    L = 100
    w0 = 2.0 * cp.pi/L
    n1, n2, n3 = 10.0, 20.0, 30.0
    a1, a2, a3 = 1., 2., 3.

    #create signal data with 3 frequency components
    x = cp.linspace(0,L,pts)
    y1 = a1*cp.cos(n1*w0*x)
    y2 = a2*cp.sin(n2*w0*x)
    y3 = a3*cp.sin(n3*w0*x)
    y = y1 + y2 + y3

    #noisy signal: uniform distribution range[-3 3]
    ns = cp.random.uniform(-3, 3, pts)

    # y_n = y + noise
    y_n = y + ns

    # #create signal including only 2 components
    # y12 = y1 + y2

    # #analytic derivative of signal
    # dy = w0*(-n1*a1*cp.sin(n1*w0*x)
    #         +n2*a2*cp.cos(n2*w0*x)
    #         +n3*a3*cp.cos(n3*w0*x) )

    #use fft.fftfreq to get frequency array corresponding to number of sample points
    freqs = fftfreq(pts)
    #compute number of cycles and radians in sample window for each frequency
    nwaves = freqs*pts
    # nwaves_2pi = w0*nwaves

    # compute the fft of the full signal
    fft_vals = fft(y)

    # with noisy signal:
    fft_vals_ns = fft(y_n)


    #mask the negative frequencies
    mask = freqs>0
    
    #double count at positive frequencies
    fft_theo = 2.0 * cp.abs(fft_vals/pts)
    
    # with noisy signal:
    fft_theo_ns = 2.0 * cp.abs(fft_vals_ns/pts)


# plot
    #plot fft of signal
    plt.xlim((0,50))
    plt.xlabel('cycles in window')
    plt.ylabel('original amplitude')
    plt.plot(cp.asnumpy(nwaves[mask]), cp.asnumpy(fft_theo[mask]))
    plt.show()




    #create a copy of the original fft to be used for filtering
    fft_new = cp.copy(fft_vals)
    #filter out y3 by setting corr. amplitude component(s) to zero  ##helping source # checked about single side spectrum
    fft_new[cp.abs(fft_new)<=a1/2] = 0. 
    #plot fft of filtered signal

# plot fft noisy signal
    plt.xlim((0,50))
    plt.xlabel('cycles in window')
    plt.ylabel('filtered amplitude')
    # plt.plot(cp.asnumpy(nwaves[mask]), cp.asnumpy(2.0*cp.abs(fft_new[mask]/pts)))
    plt.plot(cp.asnumpy(nwaves[mask]),cp.asnumpy(fft_theo_ns[mask]))
    plt.show()



    #invert the filtered fft with numpy.fft.ifft
    filt_data = cp.real(ifft(fft_new))


    #plot filtered data and compare origin, noisy signal, denoised signal
    plt.plot(cp.asnumpy(x), cp.asnumpy(y), label='original signal')
    plt.plot(cp.asnumpy(x), cp.asnumpy(y_n), label='filtered signal')
    plt.plot(cp.asnumpy(x), cp.asnumpy(filt_data), label='denoised signal')
    plt.xlim((0,50))
    plt.legend()
    plt.show()

    # #multiply fft by 2*pi*sqrt(-1)*frequency to get fft of derivative
    # dy_fft = 1.0j*nwaves_2pi*fft_vals
    # #invert to reconstruct sampled values of derivative
    # dy_recon = cp.real(ifft(dy_fft))
    # #plot reconstructed derivative and compare with analuytical version
    # plt.plot(cp.asnumpy(x), cp.asnumpy(dy), label='exact derivative')
    # plt.plot(cp.asnumpy(x), cp.asnumpy(dy_recon), label='fft derivative')
    # plt.xlim((0,50))
    # plt.legend()
    # plt.show()

def cupy_eig(mat):
    """
    Returns the eigen values of square matrix mat
    """
    # https://docs-cupy.chainer.org/en/stable/reference/generated/cupy.linalg.eigh.html
    w, v = cp.linalg.eigh(mat)
    # w eigenvalues
    # v eigenvectors
    return w, v

def cupy_J(n):
    """
    Constructs the nxn matrix J (as described in q2)
    Returns J, leading eigenvalue of J, and the associated eigenvector
    """
    # main diagonal where each of the entries is 1/2
    # Use cupyx.scipy.spare.diags()
    # http://wwwens.aero.jussieu.fr/lefrere/master/SPE/docs-python/scipy-doc/generated/scipy.sparse.diags.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
    # 𝑛×𝑛 matrix 𝐽(𝑛) full of zeros except for the diagonals adjacent to the main diagonal where each of the entries is  1/2
    J = scipy.sparse.diags([1/2, 1/2], [1,-1], shape=(n,n)).toarray()
    w, v = cupy_eig(J)

    # max leading eigval (one with large magnitude)
    idx = cp.argmax(abs(w))

    return J, abs(w[idx]), v[idx]

def rand_mat_gauss(n):
    """
    Returns nxn size array of random numbers sampled from a normal distribution with a mean of 0 and standard deviation of 1
    (Uses cupy to generate the random matrix)
    """
    # 3a)
    #B 𝑁(0,1) , the normal distribution with mean zero and standard deviation 1.
    # cupy.random.normal #Returns an array of normally distributed samples.
    B = cp.random.normal(0, 1, (n,n))
    A = (1/cp.sqrt(2)) * (B + cp.transpose(B))
    return A

def rand_mat_plusminus(n):
    """
    Returns nxn size array of random numbers uniformly distributed on the 2-value set {-1,1}
    OPTIONAL
    """
    # p4a)
    # repeat problem 3 but with matrix entries chosen uniformly from  {−1,1}  
    # instead of from a normal distribution
    # cupy.random.rand #Returns an array of uniform random values over the in- terval[0, 1).
    # cupy.random.randint #Returns a scalar or an array of integer values over [low, high).
    # cupy.random.random_integers #Returnascalaroranarrayofintegervaluesover[low, high]

    B = cp.random.random_integers(-1, 1, (n,n))
    A = (1/2) * (B + cp.transpose(B))
    return A

def rand_mat_uniform(n):
    """
    Returns nxn size array of random numbers uniformly distrubted on the range [-1,1]
    OPTIONAL
    """
    # p4b)
    # Repeat 4a but with matrix entries chosen randomly from a uniform distribution on the interval  [−1,1]
    # produces an  𝑛×𝑛  array that is symmetric and any individual element is chosen randomly from  [−1,1] .
    B = cp.random.uniform(-1, 1, (n,n))
    A = (1/2) * (B + cp.transpose(B))
    return A   

# https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


if __name__ == "__main__":

    #Fill in code here to call functions
    # pass

# Problem1)
    # 1b)
    cupy_fft()

    # 1c)
    cupy_filter()



# 
# 
# 

# Problem2)
    # 2) For n = 50
    n = 50
    J, w, v = cupy_J(n)
    print('For n = {0}, The eigenvalue of leading eigenvalue: {1}'.format(n, w))
    print('For n = {0}, The eigenvector that associate with the leading eigenvalue: \n{1}'.format(n, v))
    # For n = 500
    n = 500
    J, w, v = cupy_J(n)
    print('For n = {0}, The eigenvalue of leading eigenvalue: {1}'.format(n, w))
    print('For n = {0}, The eigenvector that associate with the leading eigenvalue: \n{1}'.format(n, v))

    # What is the value of the leading eigenvalue and how does it behave as  𝑛  becomes large?
    print('As n increases, the leading eigenvalue also increases')
    print('The magnitude of eigenvector matrix are shows symmetric')




# 
# 
# 

# problem 3)
    # 3b)
    # 10x10 array
    # Verify symmetric matrix
    # plot histogram N
    n = 10
    
    Anxn = rand_mat_gauss(n)
    # verify symmetric matrix
    check = check_symmetric(Anxn)
    print('This is {} that the matrix is symmetric'.format(check))    



    # plot histogram N
    mu, sig = cp.mean(Anxn), cp.std(Anxn)
    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))


# # https://www.tutorialspoint.com/python_data_science/python_normal_distribution.htm

    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)
    s = cp.squeeze(Anxn)
    s = cp.asnumpy(s)

    plt.figure()
    count, bins, ignored = plt.hist(s, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.show()




    # 3c)
    # Create a matrix m = rand_mat_gauss(n) with  𝑛=1000 
    n = 1000
    m = rand_mat_gauss(n)

    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))


    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('3c) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('The normal distribution of the eigenvalues smoothly follows in histogram\n\n')
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))



    # 3d) 
    # repeat 3c with  𝑛=2000  and  𝑛=4000
    # For n = 2000
    n = 2000 #200
    m = rand_mat_gauss(n)

    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))


    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('3d) n = ' + str(n))    
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('The normal distribution of the eigenvalues smoothly follows in histogram')
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))


    # For n = 4000
    n = 4000
    m = rand_mat_gauss(n)

    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))


    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('3d) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('The normal distribution of the eigenvalues smoothly follows in histogram')
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))    

# What features are independent of  𝑛 ?
    print('As shown in graph when n is increases, they are proportionally increases. \nThe distribution of the eigenvalues are independent of n. \nLeading value of the eigvalues are proprotional to n')



# 
# 
# 

# problem4)
    # p4a)
    # epeat problem 3 but with matrix entries chosen uniformly from  {−1,1}  
    # instead of from a normal distribution
    # a)
    # For n = 1000, 2000, 4000
# For n = 1000
    n = 1000
    m = rand_mat_plusminus(n)
    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))
    # For this problem, exclude the largest eigenvalue before computing the histogram.
    # max leading eigval (one with large magnitude)
    idx = cp.argmax(abs(w))
    w = np.delete(cp.asnumpy(w), cp.asnumpy(idx))

    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('4a) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))    


# For n = 2000
    n = 2000
    m = rand_mat_plusminus(n)
    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))
    # For this problem, exclude the largest eigenvalue before computing the histogram.
    # max leading eigval (one with large magnitude)
    idx = cp.argmax(abs(w))
    w = np.delete(cp.asnumpy(w), cp.asnumpy(idx))
    

    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('4a) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))    


# For n = 4000
    n = 4000
    m = rand_mat_plusminus(n)
    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))
    # For this problem, exclude the largest eigenvalue before computing the histogram.
    # max leading eigval (one with large magnitude)
    idx = cp.argmax(abs(w))
    w = np.delete(cp.asnumpy(w), cp.asnumpy(idx))
    

    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('4a) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))    

# When the distribution from which elements are chosen is changed, what is preserved and what changes?
    print('As shown in histgram, when n increases, histogram also increases.\n As n is increases the the distribution doesnt changing much.')


# 
# 
# 
    # p4b)
    # b)
# Again explore the distribution of the eigenvalues and the dependence on  𝑛 . When the distribution is changed this time, what is preserved and what changes?
    # For n = 1000, 2000, 4000
# For n = 1000
    n = 1000
    m = rand_mat_uniform(n)
    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))
    # For this problem, exclude the largest eigenvalue before computing the histogram.
    # max leading eigval (one with large magnitude)
    idx = cp.argmax(abs(w))
    w = np.delete(cp.asnumpy(w), cp.asnumpy(idx))

    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('4b) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))    


# For n = 2000
    n = 2000
    m = rand_mat_uniform(n)
    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))
    # For this problem, exclude the largest eigenvalue before computing the histogram.
    # max leading eigval (one with large magnitude)
    idx = cp.argmax(abs(w))
    w = np.delete(cp.asnumpy(w), cp.asnumpy(idx))
    

    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('4b) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))    


# For n = 4000
    n = 4000
    m = rand_mat_uniform(n)
    # cupy to compute the eigenvalues of the matrix
    w, v = cupy_eig(m)
    # print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))
    # For this problem, exclude the largest eigenvalue before computing the histogram.
    # max leading eigval (one with large magnitude)
    idx = cp.argmax(abs(w))
    w = np.delete(cp.asnumpy(w), cp.asnumpy(idx))
    

    # plot the histrogram of the eigenvalues
    mu, sig = cp.mean(m), cp.std(m)
    mu = cp.asnumpy(mu)
    sigma = cp.asnumpy(sig)

    plt.figure()
    count, bins, ignored = plt.hist(cp.asnumpy(w), bins = 30)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')
    plt.title('4b) n = ' + str(n))
    plt.show()

    print('For n = {0}, mu = {1}, sig = {2}'.format(n, mu, sig))
    print('For n = {0}, in rand_mat_gauss: \neigval = \n{1}'.format(n, w))    

# Again explore the distribution of the eigenvalues and the dependence on  𝑛 . When the distribution is changed this time, what is preserved and what changes?
    print('As shown in the graph we can see now that it does not change as n increases. It shows that distribution are most likely set')
