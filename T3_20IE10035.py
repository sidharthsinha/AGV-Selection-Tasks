import numpy as np
import cv2
import matplotlib.pyplot as plt

class KalmanFilter(object):
    def __init__(self, x, dt, varianceInMeasurementOfV, varianceInMeasurementOfX):
        #Time Duration of measurements: dt
        self.dt = dt
        #State Transition Matrix : A
        #Control Matrix is 0 since change is completely defined by average velocity contained in State Itself
        self.A = np.matrix([[1, self.dt],
                            [0, 1]])
        #Transformation Matrix : H(transforms info from sensers to the state)
        self.H = np.matrix([[1,0]])
        #Process Noise Covariance : The process of prediction is accompanied with a Gaussian noise whose mean value is 0 and covariance is in the form of a matrix called Process Noise Covariance
        self.processNoiseCovariance = np.matrix([[(self.dt**2), 0],
                                                [0, (self.dt**2)]]) * varianceInMeasurementOfV
        #Measurement Noise Covariance : The process of Measurement is accomapnied with a Guassian Noise whose mean value is 0 and covariance is in the form of a matrix called Measurement Noise Matrix
        self.measurementNoiseCovariance = varianceInMeasurementOfX
        #Error Covariance : Error Covariance Matrix represents the error that persists in the final Position estimate we make
        self.errorCovariance = np.eye(self.A.shape[1])
        #This defines the initial state
        self.x = np.matrix([[x], [0]])

    def predict(self, v):
        """Priori Estimate/Prediction Step"""
        #Change velocity artificially each time since no control information is provided. Also it isnt required either since no estimates of Velocity has to be made
        self.x[1, 0] = v
        #Predict the new position via the definition of Average Velocity
        self.x = np.dot(self.A, self.x)
        #Calculate error covariance
        #errorCovariance= A*errorCovariance*A' + processNoiseCovariance (via the law of Error Propagation : equivalent to simply adding noise)
        self.errorCovariance = np.dot(np.dot(self.A, self.errorCovariance), self.A.T) + self.processNoiseCovariance

    def update(self, z):
        """Posteriori Estimate/Update Step"""
        # S = H*errorCovariance*H'+measurementNoiseCovariance
        S = np.dot(self.H, np.dot(self.errorCovariance, self.H.T)) + self.measurementNoiseCovariance
        #Calculate the Kalman Gain
        #K = errorCovariance * H'* inv(H*errorCovariance*H'+measurementNoiseCovariance)
        K = np.dot(np.dot(self.errorCovariance, self.H.T), np.linalg.inv(S))
        #Update the state
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        #Update the Error Covariance
        I = np.eye(self.H.shape[1])
        self.errorCovariance = (I - (K * self.H)) * self.errorCovariance

z = np.loadtxt("kalmann.txt", dtype='f', delimiter=',')
X_o = z[:, 0]
Y_o = z[:, 1]

X = [372.99815102559614]
Y = [3.686804471625727e-06]

Px = []
Py = []
ind = []

sx = KalmanFilter(372.99815102559614, 1, 0.00001, 37.654038063995124)
sy = KalmanFilter(3.686804471625727e-06, 1, 0.00001, 28.541204671008742)

for i in range (1, 360):
    sx.predict(z[i,2])
    sy.predict(z[i, 3])
    sx.update(z[i, 0])
    sy.update(z[i, 1])
    A = sx.x
    B = sy.x
    X.append(float(A[0]))
    Y.append(float(B[0]))
    Px.append(float(np.linalg.det(sx.errorCovariance)))
    Py.append(float(np.linalg.det(sy.errorCovariance)))
    ind.append(i)

    print("Oiriginal Position:")
    print("X: %f Y: %f" % ( z[i-1,0], z[i-1, 1] ) )
    print("%d Iteration: " %(i))
    print("Position Obtained after applying Kalman Filter:")
    print("X: %d Y: %d" % (X[i],Y[i]))
    print("Covariance Matrix for x is: ")
    print(sx.errorCovariance)
    print("Covariance Matrix for y is:")
    print(sy.errorCovariance)


plot1 = plt.figure(1)
plt.plot(X_o, Y_o)

plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')

plt.axis('scaled')

plt.title('Measured Trajectory')

plt.plot(X, Y)


plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')

plt.axis('scaled')

plt.title('Filtered Trajectory')

plot3=plt.figure(3)
plt.plot(Px, ind)


plt.xlabel('Covariance in X')
plt.ylabel('No. of iterations')



plt.title('Covariance in X')

plot3=plt.figure(4)
plt.plot(Px, ind)


plt.xlabel('Covariance in Y')
plt.ylabel('No. of iterations')

plt.title('Covariance in Y')

plt.show()