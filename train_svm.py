import pickle

def train_svm(X_train, X_test, y_train, y_test):

    # put aside a small validation set
    from sklearn.model_selection import train_test_split
    X_train2_std, X_val_std, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    from sklearn.svm import SVC
    
    # find the best parameters for SVM (C, gamma) using the validation set
    print("Model validation: find the best SVM model [by varying C and gamma].")
    from train_svm_validation import train_svm_validation
    [C_opt, gamma_opt, accuracy_opt] = train_svm_validation(X_val_std,y_val)
    
    
    # training model
    print("Training the model: fit the model on the training data.")
    #svm = SVC(kernel='rbf', C=C_opt, gamma=gamma_opt, random_state=42)  
    svm = SVC(kernel='linear', C=1, gamma=1, random_state=42)      
    svm.fit(X_train,y_train)
    
    # calculate the generalization error on the test set
    error = svm.score(X_test, y_test)
    print("Generalization error (based on the test set): " + str(error))
    
    # save trained model
    pickle_out = open('classifier_svm.pickle','wb')
    pickle.dump(svm,pickle_out)
    pickle_out.close()
    
    return svm