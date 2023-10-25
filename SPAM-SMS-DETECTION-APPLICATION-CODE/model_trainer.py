from sklearn.metrics import classification_report
from sklearn.svm import SVC

def run_model_training(x_train, x_test, y_train, y_test):
    svm_model = SVC(kernel='linear', C=1.0)
    clf = svm_model.fit(x_train,y_train)
    clf.score(x_test,y_test)
    y_pred=clf.predict(x_test)
    print(classification_report(y_test, y_pred))

    return clf
