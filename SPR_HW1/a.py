import numpy as np
import matplotlib.pyplot as plt


# Generate samples from three normal distributions 1-D   -----part-a-----
mu, sigma = 5, 3 # mean and standard deviation
s = np.random.normal(mu, sigma, 500)
# display outputs
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.show()