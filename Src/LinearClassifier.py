from sklearn.linear_model import LogisticRegression

def linearClassifier(X_train, X_test, y_train):

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    y_pred_proba = logit.predict_proba(X_test)
    y_pred_proba = y_pred_proba[:][:, 1]

    y_pred = logit.predict(X_test)

    return y_pred_proba, y_pred