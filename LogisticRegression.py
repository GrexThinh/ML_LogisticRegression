from sympy import evaluate
import numpy as np
from map_feature import map_feature

class LogisticRegression:
    def __init__(self, alpha=0.5, iters=10000, lamb=1):
        self.alpha = alpha
        self.iters = iters
        self.lamb = lamb
        self.theta = None

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def compute_cost(self, X, y, theta, lamb):
        m=len(X)
        h_theta=self.sigmoid(X.dot(theta))
        J=1/m*(-y.T.dot(np.log(h_theta))-(1-y.T).dot(np.log(1-h_theta)))+lamb/(2*m)*np.sum(theta[1:]**2)
        return J

    def compute_gradient(self, X, y, theta, lamb):
        m=len(X)
        h_theta=self.sigmoid(X.dot(theta))
        dJ=1/m*((h_theta-y).dot(X))
        dJ[1:]+=(lamb/m)*theta[1:]
        return dJ

    def gradient_descent(self, X, y):
        # Mapping feature
        X=np.array(X)
        x1=X[:,0]
        x2=X[:,1]
        X=map_feature(x1,x2)
        theta = np.zeros(X.shape[1])
        print(f'The total of training sample: {len(y)}')
        for i in range(self.iters):
            J, dJ = self.compute_cost(X, y, theta, self.lamb), self.compute_gradient(X, y, theta, self.lamb)
            theta = theta - self.alpha * dJ
        return theta

    def fit(self, X, y):
        self.theta = self.gradient_descent(X, y)

    def predict(self, X):
        # Mapping feature
        X=np.array(X)
        x1=X[:,0]
        x2=X[:,1]
        X=map_feature(x1,x2)
        results = []
        for Xi in X:
            h = self.sigmoid(Xi.dot(self.theta))
            pred=1 if h>=0.5 else 0
            results.append(pred)
        return results

    def evaluate(self, y, y_pred):
        m=len(y)
        TP_0, TN_0, FN_0, FP_0=0, 0, 0, 0
        TP_1, TN_1, FN_1, FP_1=0, 0, 0, 0
        for i in range(m):
            if y[i]==1:
                if y_pred[i]==1:
                    TN_0+=1
                    TP_1+=1
                else:
                    FP_0+=1
                    FN_1+=1
            else:
                if y_pred[i]==1:
                    FN_0+=1
                    FP_1+=1
                else:
                    TP_0+=1
                    TN_1+=1
        precision_0=round(TP_0/(TP_0+FP_0),2)
        recall_0=round(TP_0/(TP_0+FN_0),2)
        f1_score_0=round(2*precision_0*recall_0/(precision_0+recall_0),2)
        precision_1=round(TP_1/(TP_1+FP_1),2)
        recall_1=round(TP_1/(TP_1+FN_1),2)
        f1_score_1=round(2*precision_1*recall_1/(precision_1+recall_1),2)

        correct=np.sum(y==y_pred)
        accuracy=round(correct/len(y),2)

        precision=[precision_0, precision_1]
        recall=[recall_0, recall_1]
        f1_score=[f1_score_0, f1_score_1]

        return  precision, recall, f1_score, accuracy