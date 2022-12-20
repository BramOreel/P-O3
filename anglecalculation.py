import numpy as np
import math
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
import operator

# Hoekberekening aan de hand van het best passende vlak door 2n punten. n staat voor aantal richtingen van de kubbstok / het aantal metingen die we nodig achten voor een zo goed mogelijke benadering van het echte vlak.
def angleCalculationplane(uiteinden,n):
    hoeken=[]
    beginpunt = 0
    bovengrens = len(uiteinden)/2-n
    while beginpunt <= bovengrens:
        k=0
        points = []
        while k < n:
            index = 2 * (beginpunt + k)
            lengte = (uiteinden[index+1][0]-uiteinden[index][0])**2+(uiteinden[index+1][1]-uiteinden[index][1])**2+(uiteinden[index+1][2]-uiteinden[index][2])**2
            if lengte < 15**2+5: #werkelijke lengte kubbstok is 15 cm; er wordt een marge van 5 genomen
                points.append(uiteinden[index])
                points.append(uiteinden[index+1])
            k+=1
        plane = Plane.best_fit(points)
        normaalvector = plane.normal
        a=normaalvector[0]
        b=normaalvector[1]
        c=normaalvector[2]
        hoek = np.arccos(abs(c)/(np.sqrt(a**2+b**2+c**2))) * 180 / math.pi
        hoeken.append(hoek)
        beginpunt+=1
    for hoek in hoeken:
        if hoek < 45:
            print('Geen geldige worp!')
        else:
            print('Geldige worp!')




# Hoekbereking aan de hand van het dot/scalair product.
def dotangleCalculation(uiteinden,n):
    hoeken = []
    beginpunt = 0
    bovengrens = len(uiteinden) / 2 - n
    while beginpunt <= bovengrens:
        ki = 0
        richtingen = []
        while ki < n:
            index = 2 * (beginpunt + ki)
            richtingen.append(uiteinden[index+1]-uiteinden[index])
            ki += 1
        beginpunt += 1
        li = 0
        u = (0, 0, 0)
        v = (0, 0, 0)

        while li < len(richtingen):
            if li == len(richtingen) - 2:
                u = tuple(map(operator.add, u, richtingen[len(richtingen)-2]))
                v = tuple(map(operator.add, v, richtingen[len(richtingen)-1]))
                u = (2*u[0]/len(u),2*u[1]/len(u),2*u[2]/len(u))
                v = (2 * v[0] / len(v), 2 * v[1] / len(v), 2 * v[2] / len(v))
                li+=2
            elif li == len(richtingen)-1:
                u = tuple(map(operator.add, u, richtingen[len(richtingen) - 1]))
                u = (2 * u[0] / (len(u)+1), 2 * u[1] / (len(u)+1), 2 * u[2] / (len(u)+1))
                v = (2 * v[0] / (len(v)-1), 2 * v[1] / (len(v)-1), 2 * v[2] / (len(v)-1))
                li += 2
            else:
                u = tuple(map(operator.add, u, richtingen[li]))
                v = tuple(map(operator.add, v, richtingen[li+1]))
                li += 2

        # richting 1
        k = u[0]
        l = u[1]
        m = u[2]

        # richting 2
        n = v[0]
        o = v[1]
        p = v[2]

        if k==n and l==o and m==p:
            print('zelfde rico')
            quit()


        # bepalen van de normaalvector op het vlak (deze staat loodrecht op de 2 rechten door de oorsprong)

        if k == 0:
            if l == 0:
                if o == 0:
                    a=0
                    b=1
                    c=0
                else:
                    a = -1
                    b = n/o
                    c = 0
            else:
                if n == 0:
                    a = 1
                    b = 0
                    c = 0
                else:
                    d = k
                    f = l
                    j = m
                    k = n
                    l = o
                    m = p
                    n = d
                    o = f
                    p = j

                    c = 1
                    g = (n / k * m) - p
                    q = o - (n / k * l)
                    b = g / q
                    a = (-m - (l * b)) / k

        else:
            c = 1
            g = (n/k * m) - p
            q = o - (n/k*l)
            b = g/q
            a = (-m - (l*b)) / k

        hoek = np.arccos(c/(np.sqrt(a**2+b**2+c**2))) * 180 / math.pi
        hoeken.append(hoek)



# Hoekberekening aan de hand van het uitwendig product

def vectAngleCalculation(uiteinden,n):
    hoeken = []
    beginpunt = 0
    bovengrens = len(uiteinden) / 2 - n
    while beginpunt <= bovengrens:
        ki = 0
        richtingen = []
        while ki < n:
            index = 2 * (beginpunt + ki)
            richtingen.append(uiteinden[index+1]-uiteinden[index])
            ki += 1
        beginpunt += 1
        li = 0
        u = (0, 0, 0)
        v = (0, 0, 0)

        while li < len(richtingen):
            if li == len(richtingen) - 2:
                u = tuple(map(operator.add, u, richtingen[len(richtingen)-2]))
                v = tuple(map(operator.add, v, richtingen[len(richtingen)-1]))
                u = (2*u[0]/len(u),2*u[1]/len(u),2*u[2]/len(u))
                v = (2 * v[0] / len(v), 2 * v[1] / len(v), 2 * v[2] / len(v))
                li+=2
            elif li == len(richtingen)-1:
                u = tuple(map(operator.add, u, richtingen[len(richtingen) - 1]))
                u = (2 * u[0] / (len(u)+1), 2 * u[1] / (len(u)+1), 2 * u[2] / (len(u)+1))
                v = (2 * v[0] / (len(v)-1), 2 * v[1] / (len(v)-1), 2 * v[2] / (len(v)-1))
                li += 2
            else:
                u = tuple(map(operator.add, u, richtingen[li]))
                v = tuple(map(operator.add, v, richtingen[li+1]))
                li += 2
        kruisproduct = np.cross(u, v)
        hoek = np.arccos(abs(kruisproduct[3]) / (np.sqrt(kruisproduct[1] ** 2 + kruisproduct[2] ** 2 + kruisproduct[3] ** 2))) * 180 / math.pi
        hoeken.append(hoek)
    return hoeken


def hoekberekening(uiteinden,n):
  richtingen = []
  bewegingen = []
  for i in range(len(uiteinden)):
    if i % 2 == 0:
      x = uiteinden[i+1][0] - uiteinden[i][0]
      y = uiteinden[i+1][1] - uiteinden[i][1]
      z = uiteinden[i+1][2] - uiteinden[i][2]
      richting = (x,y,z)
      richtingen.append(richting)
  for j in range(len(richtingen)-1):
    xb = richtingen[j+1][0] - richtingen[j][0]
    yb = richtingen[j+1][1] - richtingen[j][1]
    zb = richtingen[j+1][2] - richtingen[j][2]
    beweging = (xb, yb, zb)
    bewegingen.append(beweging)
    if j == len(richtingen) / 2:
        bewegingen.append(beweging)

  gemiddelderichtingen = []
  gemiddeldebewegingen = []
  index1 = 0
  while index1 <= len(richtingen)-n:
      index2 = 0
      gemrichting = (0, 0, 0)
      gembeweging = (0, 0, 0)
      while index2 < n:
          gemrichting = tuple(map(operator.add, gemrichting,richtingen[index1 + index2]))
          gembeweging = tuple(map(operator.add, gembeweging, bewegingen[index1 + index2]))
          if index2 == n - 1:
              gemrichting = (gemrichting[0]/n,gemrichting[1]/n,gemrichting[2]/n)
              gemiddelderichtingen.append(gemrichting)
              gembeweging = (gembeweging[0] / n, gembeweging[1] / n, gembeweging[2] / n)
              gemiddeldebewegingen.append(gembeweging)
          index2 += 1
      index1+=1


  hoeken = []
  for k in range(len(gemiddeldebewegingen)):
    kruisproduct = np.cross(gemiddeldebewegingen[k], gemiddelderichtingen[k])
    hoek = np.arccos(abs(kruisproduct[3])/(np.sqrt(kruisproduct[1]**2+kruisproduct[2]**2+kruisproduct[3]**2))) * 180 / math.pi
    hoeken.append(hoek)

  return hoeken