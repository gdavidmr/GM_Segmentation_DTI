import numpy as np
from sklearn.svm import SVC

def train_svm_validation(X,y):
    
    # SVM parameters to test
    C = [1e-2,1e-1,1,1e1,1e2]
    gamma = [1e-2,1e-1,1,1e1,1e2]
    
    # initialization
    accuracy1 = np.zeros((len(C),len(gamma)))
    accuracy2 = np.zeros((len(C),len(gamma)))
      
    # loop through all combinations of parameters and compute the accuracy of prediction
    for i in range(len(C)):
        for j in range(len(gamma)):
            
            print("C = %f, gamma = %f" %(C[i],gamma[j]))
            print("\n")
        
            # create SVM object
            svm = SVC(kernel='rbf', C=C[i], gamma=gamma[j], random_state=42)
        
            # train the model
            y = np.ravel(y)
            svm.fit(X,y)
        
            # test the model
            y_predicted = svm.predict(X)
        
            # test the accuracy of the model
            # compare predicted values to real values         
            # method 1:
            true = 0
            for k in range(len(y)):
                if y_predicted[k] == y[k]:
                    true += 1
            accuracy1[i,j] = true/len(y)
            
            # method 2:
            accuracy2[i,j] = svm.score(X,y)
            
    # Note that both methods for calculating accuracy yield the same result.
                     
    # find the position (x,y) of the highest value in the accuracy matrix          
    x_coor = np.where(accuracy1==np.amax(accuracy1))[0][0]
    y_coor = np.where(accuracy1==np.amax(accuracy1))[1][0]
            
    # parameters yielding the highest accuracy
    C_opt = C[x_coor]
    gamma_opt = gamma[y_coor]
            
    return [C_opt, gamma_opt, np.amax(accuracy1)]