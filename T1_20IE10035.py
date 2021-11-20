import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

startPointCol = (113, 204, 45)
endPointCol = (60, 76, 231)
obstaclesCol = (255, 255, 255)
pathCol = (0, 0, 139)

img = cv2.imread('tinp.png')
startPoint = (0, 0)
endPoint = (0, 0)

for i in range (100):
    for j in range (100):
        if( img[i][j][0] == 113 and img[i][j][1] == 204 and img[i][j][2] == 45 ):
            startPoint = (i, j)
        elif( img[i][j][0] == 60 and img[i][j][1]==76 and img[i][j][2] == 231  ):
            endPoint = (i, j)

def moveCost(x, y):
    mC = np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
    return mC

#Manhattan Distance as Heuristic
def mH(current, goal):
    mH = abs(current[0] - goal[0]) + abs(current[1] - goal[1])
    return mH

#Diagonal Distance as Heuristic
def dH( current, goal ):
    dH = max(abs(current[0] - goal[0]), abs(current[1]-goal[1]))
    return dH

#Euclidean Distance as Heuristic
def eH(current, goal):
    eH = np.sqrt( (current[0]-goal[0])**2 + (current[1]-goal[1])**2 )
    return eH

#Non Admissible Heuristic
def nonH(current, goal):
    nonH = (current[0]-goal[0])**2 + (current[1]-goal[1])**2
    return nonH

#Admissible Heuristic
def aH(current, goal):
    aH = eH(current, goal) - 0.01
    return aH

def isValid(x, y, img) -> bool:
    if(x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1] and (img[x, y] != obstaclesCol).any()):
        return True
    return False

def getNeighboursType1(point, img):
    n = []
    for a, b in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        x = point[0] + a
        y = point[1] + b
        if (isValid(x, y, img)):
            n.append((x, y))
    return n

def getNeighboursType2(point, img):
    n = []
    for a, b in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        x = point[0] + a
        y = point[1] + b
        if (isValid(x, y, img)):
            n.append((x, y))
    return n



def Astar(img, startPoint, endPoint):
    G = {} #Create a Dictionary(in the form of matrix) that stores the G values of all points
    F = {} #Create a Dictionary(in the form of matrix) that stores the F values of all points

    G[startPoint] = 0
    """Change function to dH/eH/mH/nonH/aH to use different Heuristics"""
    F[startPoint] = mH(startPoint, endPoint)

    closedSet = set() #Create a Closed Set to store all points that we have considered
    openSet = set([startPoint])  #Create an Open Set to store all points that we might want to consider
    cameFrom = {} #Create a Dictionary(in the form of matrix) that would store the parent from which each point came off so things can be backtracked

    while (len(openSet) > 0):
        #Get the point in the open set that has the lowest f score
        current = None
        currentFScore = None

        for pos in openSet:
            if current is None or F[pos] < currentFScore:
                currentFScore = F[pos]
                current = pos

        if current == endPoint:
            #Retrace our route Backward
            path = [current] #Create an Array which starts with the Current Point
            while current in cameFrom: #Run a loop while the point in consideration has a parent
                current = cameFrom[current] #Back Track where the point Came From
                path.append(current) #Append that point at the end of the array
            path.reverse() #Now reverse the Path to get the path from startPoint to endPoint
            return path, F[endPoint]

        # Because the point has been considered
        openSet.remove(current)
        closedSet.add(current)

        """You can use getNeighboursType1/2 to use Class1 or Class2"""
        for neighbour in getNeighboursType1(current, img):
           #That point has been considered with a lower f score before, so no need to do it again
           #If we had to take that path only then why not reach that point using the lower f score method
            if neighbour in closedSet:
                continue
            candidateG = G[current] + moveCost(current, neighbour)

            if neighbour not in openSet:
                openSet.add(neighbour)
            elif candidateG >= G[neighbour]:
                continue

            cameFrom[neighbour] = current
            G[neighbour] = candidateG
            """Change aH to dH/eH/mH/nonH/aH to use different Heuristics"""
            H = mH(neighbour, endPoint)
            F[neighbour] = G[neighbour] + H

def dijikstra(img, startPoint, endPoint):
    G = {} #Create a Dictionary(in the form of matrix) that stores the G values of all points
    F = {} #Create a Dictionary(in the form of matrix) that stores the F values of all points

    G[startPoint] = 0
    F[startPoint] = 0

    closedSet = set() #Create a Closed Set to store all points that we have considered
    openSet = set([startPoint])  #Create an Open Set to store all points that we might want to consider
    cameFrom = {} #Create a Dictionary(in the form of matrix) that would store the parent from which each point came off so things can be backtracked

    while (len(openSet) > 0):
        #Get the point in the open set that has the lowest f score
        current = None
        currentFScore = None

        for pos in openSet:
            if current is None or F[pos] < currentFScore:
                currentFScore = F[pos]
                current = pos

        if current == endPoint:
            #Retrace our route Backward
            path = [current] #Create an Array which starts with the Current Point
            while current in cameFrom: #Run a loop while the point in consideration has a parent
                current = cameFrom[current] #Back Track where the point Came From
                path.append(current) #Append that point at the end of the array
            path.reverse() #Now reverse the Path to get the path from startPoint to endPoint
            return path, F[endPoint]

        # Because the point has been considered
        openSet.remove(current)
        closedSet.add(current)
        """You can use getNeighboursType1/2 to use Class1 or Class2"""
        for neighbour in getNeighboursType1(current, img):
            if neighbour in closedSet:
                continue
            candidateG = G[current] + moveCost(current, neighbour)

            if neighbour not in openSet:
                openSet.add(neighbour)
            elif candidateG >= G[neighbour]:
                continue

            cameFrom[neighbour] = current
            G[neighbour] = candidateG

            F[neighbour] = G[neighbour]

def colorPath(img, path):
    for points in path:
        if( (points == startPoint) or (points == endPoint) ):
            continue
        img[points] = pathCol

start = time.time()

"""Change function to Astar or Dijikstra"""
path, costOfPath = Astar(img, startPoint, endPoint)

end = time.time()

colorPath(img, path)

img = cv2.resize(img, (1000, 1000))

cv2.imshow('Final', img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(1)
plt.imshow(img)

plt.show()

print("The time taken is {}.".format(end-start))
print("The cost function is {}.".format(costOfPath))

cv2.destroyAllWindows()