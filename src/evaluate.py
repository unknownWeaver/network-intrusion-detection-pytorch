import torch
from preprocess import prepare_data
from model import MLP
from metrics import print_report

X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

net = MLP(input_dim=X_train.shape[1])
net.load_state_dict(torch.load("mlp_weights.pt"))
net.eval()


def predict(X):
    X_t = torch.from_numpy(X)
    with torch.no_grad():
        logits = net(X_t)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().squeeze().numpy()
    return preds


y_val_pred = predict(X_val)
y_test_pred = predict(X_test)

print_report(y_val, y_val_pred, name="MLP - validation")
print_report(y_test, y_test_pred, name="MLP - test")