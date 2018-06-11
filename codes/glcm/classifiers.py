from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# n_neighbors=10
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import cv2
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


folders = ["anthracnose","mildew","rust","spot"]

path='/mnt/attic/DADA/codes/roseFinal.xlsx'                         #give path where extracted features are saved

abc=pd.read_excel(path)
X=np.array((abc.as_matrix())[1:,0:])

Y=X[:,-1]                                   # Stores labels in Y variable
X=X[:,0:-1]                                  # Stores features in X variable
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=17)      # test train split is done 80% training and 20% testing
y_train = y_train.astype('int')
y_test = y_test.astype('int')                                           #labels must be of integer type and not float. So are converted to int

X_train_path = X_train[:,-1]
X_test_path = X_test[:,-1]

X_train = X_train[:,:-1]
X_test = X_test[:,:-1]

#
# print(X_test_path)
# print(X_test)


saved = []
models = []
results = []
names = []
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
models.append(('GBC',GradientBoostingClassifier(n_estimators=112)))
models.append(('SGD',SGDClassifier()))
models.append(('MLP',MLPClassifier(hidden_layer_sizes=2048,alpha=0.001,activation='relu',solver='lbfgs')))
models.append(("KNN",KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='kd_tree')))

for name,model in models:
    model.fit(X_train,y_train)
    joblib.dump(model,"savedModels/rose"+name+".pkl")
    score = model.score(X_test,y_test)
    names.append(name)
    results.append(score*100)
    saved.append(model)
    print(name,score*100)

i = np.argmax(results)

print("\n\nmax:",names[i],'  ',results[i])

model = saved[i]

y_pred = model.predict(X_test)
print(y_pred,'\n',y_test)
x,y = 10,30
font = cv2.FONT_HERSHEY_PLAIN

for file,pred,actual in zip(X_test_path,y_pred,y_test):
    print(file)
    img = cv2.imread(file[:-4])

    cv2.putText(img, folders[pred], (x, y), font, 3, (0, 255, 0), 2)
        # cv2.putText(frame, 'actual: ' + sFolder, (0, y + h+20), font, 1, (0, 255, 0))

    cv2.putText(img, folders[actual], (x, y + 30), font, 3, (0, 255, 255), 2)
    cv2.imshow(file, img)
    cv2.waitKey(0)
    print(file)
