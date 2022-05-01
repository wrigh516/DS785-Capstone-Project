# RUN TO CREATE NEW DENSITY MODELS FROM CSV FILES IN DATA FOLDER
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import pandas as pd
from joblib import dump, load



# Departing MSP
data = pd.read_csv('data/departmsp.csv')
X = np.array([[x] for x in data['hour']])
kde = KernelDensity(kernel='tophat', bandwidth=0.35).fit(X)
dump(kde, 'models/depart_time_kde.pkl')

# Plot Simulation
Y = kde.sample(2002)
plt.hist(X, 120, label = 'measured', alpha = 1)
plt.hist(Y, 120, label = 'simulated', alpha = .75)
plt.title('Histogram of departing times from MSP')
plt.legend(loc="upper left")
plt.xlabel('Time of day')
plt.show()



# Arriving MSP
data = pd.read_csv('data/arrivemsp.csv')
X = np.array([[x] for x in data['hour']])
kde = KernelDensity(kernel='tophat', bandwidth=0.35).fit(X)
dump(kde, 'models/arrive_time_kde.pkl')

# Plot Simulation
Y = kde.sample(2002)
plt.hist(X, 120, label = 'measured', alpha = 1)
plt.hist(Y, 120, label = 'simulated', alpha = .75)
plt.title('Histogram of arriving times into MSP')
plt.legend(loc="upper left")
plt.xlabel('Time of day')
plt.show()



# Coordinates
data = pd.read_csv('data/coordinates.csv')
X = np.array(list(zip(data['latitude'], data['longitude'])))
Xlat = data['latitude']
Xlong = data['longitude']
kde = KernelDensity(kernel='gaussian', bandwidth=0.10).fit(X)
dump(kde, 'models/coordinates_kde.pkl')

# Plot Simulation
datafile = 'static/mnregion.jpg'
img = plt.imread(datafile)
plt.imshow(img, zorder=0, extent=[-95, -90.5, 43, 47.5])
Y = kde.sample(1000)
Ylat = [item[0] for item in Y]
Ylong = [item[1] for item in Y]
plt.plot(Xlong, Xlat, 'bo', alpha = 1, label = 'measured')
plt.plot(Ylong, Ylat, 'ro', alpha = .75, label = 'simulated')
plt.title('Coordinates of addresses')
plt.legend(loc="upper left")
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Density Map
yspace = np.linspace(43, 47.5, 200)
xspace = np.linspace(-95, -90.5, 200)
T = [[y, x] for y in yspace for x in xspace]
X, Y = np.meshgrid(xspace, yspace)
Z = np.exp(kde.score_samples(T))
Z = np.reshape(Z, X.shape)
plt.contour(X, Y, Z, levels = 50, colors = 'red', linewidths = 0.5)
plt.show()