import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('hiring.csv')

# Data Cleaaning
# remove all nun values to numeric value 0
dataset.experience.fillna(0, inplace = True)
dataset.test_score.fillna(int(dataset.test_score.mean()), inplace = True)

# convert string to integers for calculation
def w2n(x):    
    word_2_num = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,0:0}
    return word_2_num[x]

dataset.experience = dataset.experience.apply(lambda x: w2n(x))

# Spliting to train and test dataset
X = dataset.iloc[:,:3]
y = dataset.iloc[:,-1]

regressor = LinearRegression()

# Running the modet and fiiting the model
regressor = LinearRegression()
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# saving the model to pc
pickle.dump(regressor, open('model.pkl','wb'))

# loading the model
model = pickle.load(open('model.pkl','rb'))
