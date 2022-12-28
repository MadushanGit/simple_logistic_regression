# importing depencies

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# load the data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# double to numpy
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y into column vector
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# creating a model
class LogisticRegression(nn.Module):
  def __init__(self, n_input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)

  def forward(self, x):
    y_predicted = torch.sigmoid(self.linear(x))
    return y_predicted

model = LogisticRegression(n_features)

learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100

for epoch in range(epochs):
  y_predicted = model.forward(X_train)

  loss = criterion(y_predicted, y_train)

  loss.backward()

  optimizer.step()

  optimizer.zero_grad()

  if (epoch+1) % 10 == 0:
    print(f"epoch: {epoch+1} loss: {loss.item():.4f}")

# now the model is trained and we can get the model accuracy
with torch.no_grad():
  y_predicted_cls = model.forward(X_test).round()
  acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
  print(f"accuracy: {acc:.4f}")

