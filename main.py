import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


if __name__ == "__main__":
    df=pd.read_csv('coronaji.csv')
    train,test = data_split(df,0.2)
    X_train=train[['Age','Fever','BodyPains','RunnyNose','Difficulty_in_Breath']].to_numpy()
    X_test = test[['Age','Fever','BodyPains','RunnyNose','Difficulty_in_Breath']].to_numpy()


    Y_train = train[['infection_Probability']].to_numpy().reshape(5684 ,)
    Y_test = test[['infection_Probability']].to_numpy().reshape(1420 ,)

    clf=LogisticRegression()
    clf.fit(X_train,Y_train)

    file = open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()

   


