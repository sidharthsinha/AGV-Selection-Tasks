import cv2
import cv2.aruco as aruco
import numpy as np
import sys, time, math

# Checks if a matrix is a valid rotation matrix. Rotation Matrices are symmetric therefore Rt = R-1.
# The below function is a predefined standard in the learn opencv documentation regarding euler angles conversion
# Norm indicates the magnitude of matrix
# The function returns the truth value of whether the matrix is symmetric or not
# If it is symmetric (I - Rt*R) should be 0 and hence norm should be 0 and hence less than 10^-6
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# ID to be found to verify whether we have found the right Aruco
id = 23
#Marker Size in cms
markerSize = 10

# Load the Camera Matrix and Camera Distortion Parameters using an alternate txt file which has preloaded calibration matrices
cameraMatrix = np.loadtxt('cameraMatrix.txt', delimiter=',')
cameraDistortion = np.loadtxt('cameraDistortion.txt', delimiter=',')

# 3x3 rotation matrix to rotate a 3d vector around the x axis
flipAroundXMatrix = np.zeros((3, 3), dtype=np.float32)
flipAroundXMatrix[0, 0] = 1.0
flipAroundXMatrix[1, 1] = -1.0
flipAroundXMatrix[2, 2] = -1.0

# Load the required dictionary(6x6 one) into a variable
arucoDictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
#Load the required parameters for detection into a variable
parameters = aruco.DetectorParameters_create()

# Capture the Video Camera, 0 means front camera, 1 means rear camera
cap = cv2.VideoCapture(0)
# The Capture being set to the size the Camera was calibrated with
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load the required font to font variable
font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    #Read the camera frame, frame is the captured image array and ret is return value (either true or false)(whether captured or not)
    ret, frame = cap.read()

    #Convert in gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    #Find all the aruco markers in the image
    """Copied from OpenCV Documentation/other resources for understanding:
    corners - vector of detected marker corners. For each marker, its four corners are provided. For N detected markers, the dimensions of this array is Nx4. The order of the corners is clockwise.
    ids	- vector of identifiers of the detected markers. The identifier is of type int  For N detected markers, the size of ids is also N. The identifiers have the same order than the markers in the imgPoints array.
    rejected - contains the imgPoints of those squares whose inner code has not a correct codification. Useful for debugging purposes.
    
    image - input image
    dictionary - indicates the type of markers that will be searched
    corners - vector of detected marker corners. For each marker, its four corners are provided. For N detected markers, the dimensions of this array is Nx4. The order of the corners is clockwise.
    parameters - marker detection parameters
    cameraMatrix - optional input 3x3 floating-point camera matrix A=⎡⎣⎢fx000fy0cxcy1⎤⎦⎥
    distCoeff - optional vector of distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6],[s1,s2,s3,s4]]) of 4, 5, 8 or 12 elements"""
    corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=arucoDictionary, parameters=parameters, cameraMatrix=cameraMatrix, distCoeff=cameraDistortion)

    if ids is not None and ids[0] == id:
        # This function recieves the detected markers and returns their pose estimation to the camera individually. So for
        # each marker one rotation and translation vector is returned.
        # ret = [rvec, tvec, ?]
        # array of rotation and position of each marker in camera frame
        # rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker with respect to camera frame
        # tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame(x, y, z)
        # rvec is vector representing the rotation axis whose direction is perpendicular to the plane of that rotation
        # and its length encodes the angle in radians to rotate
        ret = aruco.estimatePoseSingleMarkers(corners, markerSize, cameraMatrix, cameraDistortion)
        # Get the first value since we are concerned only about the one marker that we detect
        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

        # Draw a boundary around detected markers
        aruco.drawDetectedMarkers(frame, corners)
        # Draw an Axis representing the coordinate system with marker centre as origin, z axis coming towards us, x axis going right, and
        #y axis going up
        aruco.drawAxis(frame, cameraMatrix, cameraDistortion, rvec, tvec, 10)

        # Print the position of tag in camera frame
        str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f" % (tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, str_position, (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Rodrigues turns the rotation vector in the rotation matrix
        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
        #R_tc is rotation matrix to go from tag to camera
        #R_ct is rotation matrix to go from camera to tag
        R_tc = R_ct.T

        # Using the predefined function we use this rotation matrix and convert it to Euler Angles
        roll, pitch, yaw = rotationMatrixToEulerAngles(flipAroundXMatrix * R_tc)

        # Print the marker's attitude respect to camera frame
        str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
        math.degrees(roll), math.degrees(pitch),
        math.degrees(yaw))
        cv2.putText(frame, str_attitude, (0, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Get Position and attitude of the camera respect to the marker
        pos_camera = -R_tc * np.matrix(tvec).T

        str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (pos_camera[0], pos_camera[1], pos_camera[2])
        cv2.putText(frame, str_position, (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Get the attitude of the camera respect to the frame
        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(flipAroundXMatrix * R_tc)
        str_attitude = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (math.degrees(roll_camera), math.degrees(pitch_camera),math.degrees(yaw_camera))
        cv2.putText(frame, str_attitude, (0, 250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Display the frame
    cv2.imshow('frame', frame)

    #use 'q' to quit
    #cv2.waitKey(1) returns the key that you press
    #1 is the no of milliseconds for which the frame has to be displayed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break