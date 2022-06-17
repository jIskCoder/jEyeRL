import numpy as np
import math

def Ang_3points(origin,p1,p2):
    o1=p1-origin
    o2=p2-origin
    cosine_angle = np.dot(o1, o2) / (np.linalg.norm(o1) * np.linalg.norm(o2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def cart2polar(x,y,z):
    r=math.sqrt(x**2+y**2+z**2)
    theta = math.atan(y/x)
    phi=math.acos(z/r)
    return [r,theta,phi]

