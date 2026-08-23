from sklearn.linear_model import LogisticRegression
from preprocess import prepare_data
from metrics import print_report

X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print_report(y_val, y_val_pred, name="baseline - validation")
print_report(y_test, y_test_pred, name="baseline - test")