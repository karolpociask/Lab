import numpy as np
import matplotlib.pyplot as plt
'''
# --------------------------------------- array ---------------------------
arr = np.array([1, 2, 3, 4, 5])
print(arr)

A = np.array([[1, 2, 3], [7, 8, 9]])
print(A)
A = np.array([[1, 2, 3],
              [7, 8, 9]])
A = np.array([[1, 2, \
               3],
              [7, 8, 9]])
print(A)
# --------------------------------------- arrange ---------------------------
v = np.arange(1, 7)
print(v, "\n")
v = np.arange(-2, 7)
print(v, "\n")
v = np.arange(1, 10, 3)
print(v, "\n")
v = np.arange(1, 10.1, 3)
print(v, "\n")
v = np.arange(1, 11, 3)
print(v, "\n")
v = np.arange(1, 2, 0.1)
print(v, "\n")
# --------------------------------------- linspace ---------------------------
v = np.linspace(1, 3, 4)
print(v)
v = np.linspace(1, 10, 4)
print(v)

# --------------------------------------- funkcje pomocnicze ---------------------------
X = np.ones((2, 3))
Y = np.zeros((2, 3, 4))
Z = np.eye(2)
Q = np.random.rand(2, 5)
print(X, "\n\n", Y, "\n\n", Z, "\n\n", Q)
# --------------------------------------- mieszane ---------------------------
V = np.block([[
np.block([
np.block([[np.linspace(1, 3, 3)],
[np.zeros((2, 3))]]),
np.ones((3, 1))])
],
[np.array([100, 3, 1/2, 0.333])]])
print(V)
# --------------------------------------- elementy tablicy ---------------------------
print(V[0, 2])
print(V[3, 0])
print(V[3, 3])
print(V[-1, -1])
print(V[-4, -3])
print(V[3, :])
print(V[:, 2])
print(V[3, 0:3])
print(V[np.ix_([0, 2, 3], [0, -1])])
print(V[3])
# --------------------------------------- usuwanie fragmentow ---------------------------
Q = np.delete(V, 2, 0)
print(Q)
Q = np.delete(V, 2, 1)
print(Q)
v = np.arange(1, 7)
print(np.delete(v, 3, 0))
# --------------------------------------- sprawdzanie rozmiarow ---------------------------
np.size(v)
np.shape(v)
np.size(V)
np.shape(V)
# --------------------------------------- operacje na macierzach ---------------------------
A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]])
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]])
print(A+B)
print(A-B)
print(A+2)
print(2*A)
# --------------------------------------- mnozenie macierzowe ---------------------------
MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)
# --------------------------------------- mnozenie tablicowe ---------------------------
MT1 = A*B
print(MT1)
MT2 = B*A
print(MT2)
# --------------------------------------- dzielenie tablicowe ---------------------------
DT1 = A/B
print(DT1)
# --------------------------------------- dzielenie macierzowe URL ---------------------------
C = np.linalg.solve(A, MM1)
print(C)
x = np.ones((3, 1))
b = A@x
y = np.linalg.solve(A, b)
print(y)
# --------------------------------------- potegowanie ---------------------------
PM = np.linalg.matrix_power(A, 2)
PT = A**2
# --------------------------------------- transpozycja ---------------------------
A.T
A.transpose()
A.conj().T
A.conj().transpose()

# --------------------------------------- logika ---------------------------
A == B
A != B
2 < A
A > B
A < B
A >= B
A <= B
np.logical_not(A)
np.logical_and(A, B)
np.logical_or(A, B)
np.logical_xor(A, B)
print(np.all(A))
print(np.any(A))
print(v > 4)
print(np.logical_or(v > 4, v < 2))
print(np.nonzero(v > 4))
print(v[np.nonzero(v > 4)])
print(np.max(A))
print(np.min(A))
print(np.max(A, 0))
print(np.max(A, 1))
print(A.flatten())
print(A.flatten('F'))
# --------------------------------------- wykres ---------------------------
x = [1,2,3]
y = [4,6,5]
plt.plot(x,y)
plt.show()
# --------------------------------------- sinus ---------------------------
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y)
plt.show()
# --------------------------------------- ulepszone ---------------------------
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x, y,'r:',linewidth=6)
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Nasz pierwszy wykres')
plt.grid(True)
plt.show()
# --------------------------------------- kilka wykresow ---------------------------
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
plt.plot(x, y1, 'r:', x, y2, 'g')
plt.legend(('dane y1', 'dane y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
plt.show()
# --------------------------------------- druga wersja ---------------------------
x = np.arange(0.0, 2.0, 0.01)
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x, y, 'b')
l2,l3 = plt.plot(x, y1, 'r:', x, y2, 'g')
plt.legend((l2, l3, l1), ('dane y1', 'dane y2', 'y1*y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)
plt.show()
'''
# zadanie 3
A_1 = np.array([np.linspace(1, 5, 5), np.linspace(5, 1, 5)])
A_2 = np.zeros((3, 2))
A_3 = np.ones((2, 3))*2
A_4 = np.linspace(-90, -70, 3)
A_5 = np.ones((5, 1))*10

A = np.block([[A_3], [A_4]])
A = np.block([A_2, A])
A = np.block([[A_1], [A]])
A = np.block([A, A_5])


# zadanie 4

B = A[1] + A[3]

# zadanie 5

C = np.array([max(A[:, 0]), max(A[:, 1]), max(A[:, 2]), max(A[:, 3]), max(A[:, 4]), max(A[:, 5])])


# zadanie 6

D = np.delete(B, 0)
D = np.delete(D, len(D)-1)

# zadanie 7

for x in range(4):
    if(D[x]==4):
        D[x]=0

# zadanie 8
max = C[0]
max_ID = 0
min = C[0]
min_ID = 0
for x in range(len(C)):
    if(C[x] > max):
        max = C[x]
        max_ID = x
E = np.delete(C, max_ID)
for x in range(len(E)):
    if(E[x] < min):
        min = C[x]
        min_ID = x
E = np.delete(E, min_ID)
print(C)
print(E)