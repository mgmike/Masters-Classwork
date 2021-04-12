import numpy as np

D1 = np.array([[1,8,6,4,4],[1,10,5,2,6],[6,2,1,8,2]])
D2 = np.array([[0,0,-6,-5,3],[0,5,0,6,-2],[2,0,5,1,0]])
D = np.array([[1,8,6,4,4,0,0,-6,-5,3],[1,10,5,2,6,0,5,0,6,-2],[6,2,1,8,2,2,0,5,1,0]])
d,n = D.shape
d1,n1 = D1.shape
print(D.shape)

# PCA
m = np.sum(D, axis=1) 
m = m / D.shape[1]
xm = D - m.reshape(m.shape[0], 1)
print('******PCA******')
print('xk-m: ')
print(xm)
S = xm.dot(xm.T)
print('S: ')
print(S)
w, v = np.linalg.eig(S)
print('Eigenvectors: ')
print(v)
print('Eigenvalues: ')
print(w)

e = v[:,np.argmax(w)].reshape((d,1))
print(e)
ak = e.T.dot(xm).reshape((1,n))
print(ak)

xk_hat = m.reshape(d,1) + np.dot(e,ak)
print(xk_hat)

# LDA
m1 = np.sum(D1, axis=1) / D1.shape[1]
m2 = np.sum(D2, axis=1) / D2.shape[1]
print('*****LDA******')
print(m1)
print(m2)
xm1 = D1 - m1.reshape(m1.shape[0], 1)
xm2 = D2 - m2.reshape(m2.shape[0], 1)

S1 = xm1.dot(xm1.T)
S2 = xm2.dot(xm2.T)
print('S1: ')
print(S1)
print('S2: ')
print(S2)
SLDA = S1 + S2
print('Sw: ')
print(SLDA)

w = np.linalg.inv(SLDA).dot(m1 - m2).reshape((d1,1))
print('w: ')
print(w)
yk = w.T.dot(D).reshape((1,n))
print('yk: ')
print(yk)
xk_prime = w.dot(yk)
print('xk prime: ')
print(xk_prime)

theta = np.abs(xk_prime - 2)
theta = np.sum(theta, axis=1)
theta = np.reshape(theta / n, (theta.shape[0],1))
print('Theta: ')
print(theta)
mew = np.mean(xk_prime, axis=1).reshape((d,1))
print('Mean and standard deviation: ')
print(mew)

sd = np.std(xk_prime,axis=1).reshape((d,1))
sigma = np.sqrt(sd)
print(sd)

lamb = np.array([[1,2],[3,0]])

pw1 = 0.5
pw2 = 0.5

testx = np.array([1,2,3]).reshape((theta.shape[0],1))
pxw1 = np.abs(testx - 2) / theta
theta = np.divide(0.5 , testx)
pxw1 = theta * np.exp(-1 * pxw1)

pxw2_1 = np.divide(1, np.sqrt(2 * np.pi) * sigma)
pxw2_2 = np.exp(-1 * 0.5 * (np.square(testx - mew) / sigma))
pxw2 = pxw2_1 * pxw2_2
print(pxw1)
print(pxw2)

pw1x = (pxw1 * pw1) / (pxw1 * pw1 + pxw2 * pw2)
pw2x = (pxw2 * pw2) / (pxw1 * pw1 + pxw2 * pw2)
print('pw1x: ')
print(pw1x)
print('pw2x: ')
print(pw2x)

print(2 * pw1x)
print(3 * pw2x)
