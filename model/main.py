from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt


def SIGMOID(x):
    return 1/(1+np.exp(-x))

def SLP_SGD(tr_X, tr_y,alpha, rep):
    n = tr_X.shape[1]*tr_y.shape[1]
    random.seed = 123
    w = random.sample(range(1,100),n)
    w = (np.array(w)-50)/100
    w = w.reshape(tr_X.shape[1],-1)
    
    for i in range(rep):
        for k in range(tr_X.shape[0]): #모든 iris 데이터 행 반복하고 그때마다 w update
            x = tr_X[k,:]
            v = np.matmul(x, w)
            y = SIGMOID(v)  # 활성 함수
            e = tr_y[k,:] - y  # 에러
            w = w + alpha * y * (1 - y) * e * x.reshape(4,1)
        error[i]=np.mean(e)
        print("error",i,np.mean(e))

    return w #최종 가중치 반환


##prepare datasets#################
iris = datasets.load_iris()
X = iris.data
target = iris.target
error = np.zeros(1000) #error 저장할 행렬


#one hot encoding
num = np.unique(target,axis=0)
num = num.shape[0]
y = np.eye(num)[target]

##Training (get W) ################
W = SLP_SGD(X,y,alpha = 0.5, rep=1000)
#훈련 시켜서 w 최종 결정

##Test ################
pred = np.zeros(X.shape[0]) #예측값 저장
for i in range(X.shape[0]):
    #print(i)
    v = np.matmul(X[i,:],W) #input,weight
    y = SIGMOID(v)

    pred[i] = np.argmax(y) #3종류 iris 중 예측값 가장 큰 것 선택
    print("target, predict", target[i], pred[i])

print("accuracy :", np.mean(pred==target)) #전체 정확도


##show error change#############
plt.plot(error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('rep')
plt.show()


