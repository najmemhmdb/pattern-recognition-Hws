import numpy as np
import matplotlib.pyplot as plt


# Display the generated samples                    -----part-c3-----
mean = [2, -1]
cov = [[3, 0], [0, 0.2]]
x, y = np.random.multivariate_normal(mean, cov, 500).T
plt.plot(x, y, '.')
plt.axis('equal')
plt.show()
