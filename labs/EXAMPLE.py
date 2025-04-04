import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 *  rng.randn(50) - 1

model = LinearRegression(fit_intercept=True)
X = x[:, np.newaxis]
model.fit(X, y)

xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)

plt.show()