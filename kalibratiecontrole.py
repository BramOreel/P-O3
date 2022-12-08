import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
import operator

def uiteinden(x,y,z):
    uiteinden=[]
    for i in range(len(x)):
        uiteinden.append((x[i],y[i],z[i]))
    return uiteinden

def middelpuntcoordinaten(uiteinden):
    middelpunten = []

    for i in range(len(uiteinden)):
        if i % 2 == 0:
            x = (uiteinden[i][0] + uiteinden[i + 1][0]) / 2
            y = (uiteinden[i][1] + uiteinden[i + 1][1]) / 2
            z = (uiteinden[i][2] + uiteinden[i + 1][2]) / 2
            middelpunt = (x, y, z)
            middelpunten.append(middelpunt)
    return middelpunten


def xyz(middelpunten):
    x = []
    y = []
    z = []
    for i in middelpunten:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    return x, y, z


'''
De gegevens voor het bepalen van de parabool en de bijhorende vlakken. Hierbij stellen de x,y,z lijsten 
de coördinaten voor van de uitenden van de kubb stok op verschillende tijdstippen.
'''
x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.4, 9, 9.5, 10]
y = [0, 7, 2.7, 3, 4.3, 5.8, 8, 7, 8, 9.9, 10, 12, 13.2, 14, 14, 15.9, 16, 17, 18, 19, 20]


def formule(x, a, b, c):
    return -a * x ** 2 + b * x + c


def bergParabool(a, b, c):
    return [formule(0, a, b, c), formule(1, a, b, c), formule(2, a, b, c), formule(3, a, b, c), formule(4, a, b, c),
            formule(5, a, b, c), formule(6, a, b, c), formule(7, a, b, c), formule(8, a, b, c), formule(9, a, b, c),
            formule(10, a, b, c), formule(11, a, b, c), formule(12, a, b, c), formule(13, a, b, c),
            formule(14, a, b, c), formule(15, a, b, c), formule(16, a, b, c), formule(17, a, b, c),
            formule(18, a, b, c), formule(19, a, b, c), formule(20, a, b, c)]


z = bergParabool(1, 20, 0)

'''
Nu moeten er berekeningen en benaderingen worden gemaakt om zo goed mogelijke resultaten te verkrijgen.
Vervolgens stellen we de parabool en de vlakken op.
Ten slotte verschijnt de figuur.
'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)


slope, intercept, r, p, std_err = stats.linregress(x, y)
print('De nauwkeurigheid van de benaderde rechte bepaald aan de hand van x en y is' + r)

y = [a * slope + intercept for a in x]

yBegin = x[0] * slope + intercept
yEind = x[-1] * slope + intercept

lengte = yEind - yBegin
breedte = x[-1] - x[0]

lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
lengtedeelgebieden = lengtegebied / (len(x) - 1)


def puntenOpRechte(lengtedeelgebieden, n):
    nieuwePunten = [0]
    k = 1
    for i in range(n - 1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1
    return nieuwePunten


line = puntenOpRechte(lengtedeelgebieden, len(x))
model = np.poly1d(np.polyfit(line, z, 2))

print('Een maat voor de nauwkeurigheid van de parabool bedraagt' + str(model.r))

kwadraat = [a * b for a, b in zip(line, line)]
z = [a * model.coeffs[0] + model.coeffs[1] * b + model.coeffs[2] for a, b in zip(kwadraat, line)]


def vlakplot(uiteinden, n):
    middelpunten = []
    aantal = len(uiteinden) // (2 * n)
    for m in range(aantal):
        if n % 2 == 0:
            punt = tuple(map(operator.add, uiteinden[m * 2 * n + n], uiteinden[m * 2 * n + n + 1]))
        else:
            punt = tuple(map(operator.add, uiteinden[m * 2 * n + n - 1], uiteinden[m * 2 * n + n]))
        middelpunt = tuple(t / 2 for t in punt)
        x = middelpunt[0]
        y = slope * x + intercept
        z = middelpunt[2]
        middelpunten.append((x, y, z))

    gebieden = []
    l = 0
    for k in range(aantal):
        punten = [middelpunten[k]]
        for j in range(n):
            oorsprong = tuple(map(operator.sub, uiteinden[l], uiteinden[l + 1]))
            punten.append(tuple(map(operator.add, oorsprong, middelpunten[k])))
            l += 2
        gebieden.append(punten)

    for i in range(len(gebieden)):
        points = Points(gebieden[i])
        plane = Plane.best_fit(points)
        plane.plot_3d(ax, alpha=0.2, lims_x=(-0.1, 0.1), lims_y=(-0.1, 0.1))
        plane.point.plot_3d(ax, s=30)


ax.plot(x, y, z)

vlakplot(uiteinden, 20)

plt.title("Interpolerende parabool met bijhorende vlakken.")

plt.show()


def predictparaboolbaanUiteindePunten(middelpunten, n):
    x = []
    y = []
    z = []

    x1 = []
    y1 = []
    z1 = []

    x2 = []
    y2 = []
    z2 = []

    for i in range(n // 2):
        x.append(middelpunten[i][0])
        y.append(middelpunten[i][1])
        z.append(middelpunten[i][2])

        x1.append(middelpunten[i][0])
        y1.append(middelpunten[i][1])
        z1.append(middelpunten[i][2])

    for i in range(n // 2):
        x.append(middelpunten[-i][0])
        y.append(middelpunten[-i][1])
        z.append(middelpunten[-i][2])

        x2.append(middelpunten[-i][0])
        y2.append(middelpunten[-i][1])
        z2.append(middelpunten[-i][2])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    nieuwePunten = [0]

    yBegin = x1[0] * slope + intercept
    yEind = x1[-1] * slope + intercept
    lengte = yEind - yBegin
    breedte = x1[-1] - x1[0]

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x1) - 1)

    k = 1
    for i in range(len(x1) - 1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1

    yBegin = x2[0] * slope + intercept
    yEind = x2[-1] * slope + intercept
    lengte = yEind - yBegin
    breedte = x2[-1] - x2[0]

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x2) - 1)

    k = 1
    for i in range(len(x2) - 1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1


    model = np.poly1d(np.polyfit(nieuwePunten, z, 2))

    kwadraat = [a * b for a, b in zip(nieuwePunten, nieuwePunten)]
    z = [a * model.coeffs[0] + model.coeffs[1] * b + model.coeffs[2] for a, b in zip(kwadraat, nieuwePunten)]

    ax.plot(x, y, z)

    plt.title("Interpolerende parabool op basis van de uiteindepunten.")

    plt.show()


def predictparaboolbaanMiddelpunten(middelpunten, n):
    x = []
    y = []
    z = []
    for i in range(n // 4, 3 * n // 4):
        x.append(middelpunten[i][0])
        y.append(middelpunten[i][1])
        z.append(middelpunten[i][2])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)

    slope, intercept, r, p, std_err = stats.linregress(x, y)

    yEind = x[-1] * slope + intercept
    yBegin = x[0] * slope + intercept
    lengte = yEind - yBegin
    breedte = x[-1] - x[0]

    lengtegebied = np.sqrt(lengte ** 2 + breedte ** 2)
    lengtedeelgebieden = lengtegebied / (len(x) - 1)

    nieuwePunten = [0]
    k = 1
    for i in range(len(x) - 1):
        nieuwePunten.append(round(k * lengtedeelgebieden, 2))
        k += 1

    model = np.poly1d(np.polyfit(nieuwePunten, z, 2))

    kwadraat = [a * b for a, b in zip(nieuwePunten, nieuwePunten)]
    z = [a * model.coeffs[0] + model.coeffs[1] * b + model.coeffs[2] for a, b in zip(kwadraat, nieuwePunten)]

    ax.plot(x, y, z)

    plt.title("Interpolerende parabool op basis van de middelpunten.")

    plt.show()

predictparaboolbaanMiddelpunten(middelpunten,n)
predictparaboolbaanUiteindePunten(middelpunten,n)


def main():
    '''
    uiteinden = []
    middelpunten = middelpuntcoordinaten(uiteinden)
    x,y,z = xyz(middelpunten)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    k=0
    p = ['red', 'blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(int(len(uiteinden)/2)):
        x = [uiteinden[2*k][0],uiteinden[2*k+1][0]]
        y = [uiteinden[2*k][1], uiteinden[2*k + 1][1]]
        z = [uiteinden[2*k][2], uiteinden[2*k + 1][2]]
        ax.scatter(x,y,z)
        while i >= len(p):
            i -= len(p)
        ax.scatter(x, y, z, color=p[i])
        k += 1
    plt.show()
