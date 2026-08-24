import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocess import prepare_data
from model import MLP

X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

X_train_t = torch.from_numpy(X_train)
y_train_t = torch.tensor(y_train.values, dtype = torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

net = MLP(input_dim=X_train.shape[1])
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

num_epochs = 20


for epoch in range(num_epochs):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = net(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        
        
    avg_loss = total_loss / len(train_ds)
    print(f"epoch {epoch+1}/{num_epochs}  train loss: {avg_loss:.4f}")


torch.save(net.state_dict(), "mlp_weights.pt")
        