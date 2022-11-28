import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
import operator

def xyz(uiteinden):
    x = []
    y = []
    z = []
    for i in uiteinden:
        x.append(uiteinden[0])
        y.append(uiteinden[1])
        z.append(uiteinden[2])
    return x,y,z

x = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.4,9,9.5,10]
y = [0,7,2.7,3,4.3,5.8,8,7,8,9.9,10,12,13.2,14,14,15.9,16,17,18,19,20]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def formule(x,a,b,c):
  return -a*x**2+b*x+c
def bergParabool(a,b,c):
  return [formule(0,a,b,c), formule(1,a,b,c),formule(2,a,b,c),formule(3,a,b,c),formule(4,a,b,c),formule(5,a,b,c),formule(6,a,b,c),formule(7,a,b,c),formule(8,a,b,c),formule(9,a,b,c),formule(10,a,b,c),formule(11,a,b,c),formule(12,a,b,c),formule(13,a,b,c),formule(14,a,b,c),formule(15,a,b,c),formule(16,a,b,c),formule(17,a,b,c),formule(18,a,b,c),formule(19,a,b,c),formule(20,a,b,c)]

z = bergParabool(1,20,0)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot(x,y,z)

y=[a*slope for a in x]

yBegin=x[0]*slope+intercept
yEind=x[-1]*slope+intercept

lengte = abs(yEind-yBegin)
breedte= abs(x[-1]-x[0])

lengtegebied = np.sqrt(lengte**2+breedte**2)
lengtedeelgebieden = lengtegebied/(len(x)-1)

def puntenOpRechte(lengtedeelgebieden,n):
  nieuwePunten = [0]
  k=1
  for i in range(n-1):
    nieuwePunten.append(round(k*lengtedeelgebieden,2))
    k+=1
  return nieuwePunten
line = puntenOpRechte(lengtedeelgebieden,len(x))
model = np.poly1d(np.polyfit(line, z, 2))

kwadraat = [a * b for a, b in zip(line, line)]
z = [a * model.coeffs[0] + model.coeffs[1]*b + model.coeffs[2] for a,b in zip(kwadraat, line)]
#print(line,z)
'''
u=[0,1,2,3,4,5,6,7,8,9]
x = np.linspace(-4*np.pi,4*np.pi,10)

y = np.linspace(-4*np.pi,4*np.pi,10)

#z = x**2 + y**2

u = [0,1,2,3,4,5,6,7,8,9]
products = [a * b for a, b in zip(u, u)]
z= [a * (-0.1) + b + 200 for a,b in zip(products, u)]
'''

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.plot(x,y,z)

plt.show()


def predictparaboolbaan(middelpunten,n):
    x=[]
    y=[]
    z=[]
    for i in range(n):
        x.append(middelpunten[i][0])
        y.append(middelpunten[i][1])
        z.append(middelpunten[i][2])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z)
    
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    '''
    x = []
    k=0
    for j in range(l):
        x.append(k)
        k += p
    y = [a * slope for a in x]
    '''
    yEind = x[-1] * slope

    lengte = abs(yEind)
    breedte = abs(x[-1] - x[0])

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x) - 1)

    nieuwePunten = [0]
    k = 1
    for i in range(len(x)-1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1

    model = np.poly1d(np.polyfit(nieuwePunten, z, 2))

    kwadraat = [a * b for a, b in zip(nieuwePunten, nieuwePunten)]
    z = [a * model.coeffs[0] + model.coeffs[1] * b + model.coeffs[2] for a, b in zip(kwadraat, nieuwePunten)]


    ax.plot(x, y, z)

    plt.show()

def vlakplot(uiteinden,n):
    '''
        punten = [uiteinden[0], uiteinden[1],uiteinden[2]]
        points = Points(punten)
        plane = Plane.best_fit(points)
        plot_3d(points.plotter(c='k', s=5, depthshade=False), plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)), )
        plt.show()
    '''

    gebieden = []
    aantal = len(uiteinden) // (2*n)
    l = 0
    for k in range(aantal):
        punten = [(0,0,0)]
        for j in range(n):
            punten.append( tuple(map(operator.sub,uiteinden[l],uiteinden[l+1])))
            l += 2
        gebieden.append(punten)

    for i in range(len(gebieden)):
        points = Points(gebieden[i])
        plane = Plane.best_fit(points)
        plot_3d(points.plotter(c='k', s=5, depthshade=False),plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),)
        plt.show()

        ''' 
        x = np.array([0])
        y = np.array([100])
        z = np.array([-50, 60])

        # plotting
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z)
        plt.show()
        '''

def main():
    x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.4, 9, 9.5, 10]
    y = [0, 7, 2.7, 3, 4.3, 5.8, 8, 7, 8, 9.9, 10, 12, 13.2, 14, 14, 15.9, 16, 17, 18, 19, 20]
    z = [0,1,4,9,16,25,36,49,64,81,100,81,64,49,36,25,16,9,4,1,0]
    middelpunten = []
    for i in range(21):
        middelpunten.append((x[i],y[i],z[i]))
    predictparaboolbaan(middelpunten,21)
    vlakplot(middelpunten,3)

main()