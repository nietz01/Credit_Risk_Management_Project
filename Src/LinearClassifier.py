from sklearn.linear_model import LogisticRegression

def linearClassifier(X_train, X_test, y_train):

    logit = LogisticRegression()
    logit.fit(X_train, y_train)

    # in-sample
    y_pred_proba_train = logit.predict_proba(X_train)
    y_pred_proba_train = y_pred_proba_train[:][:, 1]
    y_pred_train = logit.predict(X_train)

    # out-of-sample
    y_pred_proba_test = logit.predict_proba(X_test)
    y_pred_proba_test = y_pred_proba_test[:][:, 1]
    y_pred_test = logit.predict(X_test)

    return y_pred_proba_train, y_pred_train, y_pred_proba_test, y_pred_test