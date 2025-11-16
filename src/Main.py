import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import  DataLoader

from Diabetes_AI.src.Dataset import Dataset
from Diabetes_AI.src.Model import Model

data = pd.read_csv("../diabetes.csv")

input = data.iloc[:,0:-1]
y_string = list(data.iloc[:,-1])

y_int = []

for i in range(len(y_string)):
    if y_string[i] == "positive":
        y_int.append(1)
    else:
        y_int.append(0)

y = np.array(y_int)

#normalization
scaler = StandardScaler()
scaler.fit(input)
x = scaler.transform(input)

x = torch.tensor(x)
y = torch.tensor(y)

y = y.unsqueeze(1)


dataset = Dataset(x, y)

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

net = Model()

criterion = torch.nn.BCELoss(reduction='mean')  # 'sum' or 'none' are also valid

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)

epochs = 200

for epoch in range(epochs):
    for inputs, labels in train_loader:
         inputs = inputs.float()
         labels = labels.float()
         outputs = net(inputs)
         loss = criterion(outputs, labels)
         optimizer.zero_grad()
         #backpro
         loss.backward()
         #update weights
         optimizer.step()

    output = (outputs > 0.5).float()

    accuracy = output.eq(labels).sum().float() / output.size()[0]
    print("epochs: " + str(epoch) + " accuracy: " + str((accuracy)) + " loss: "+ str(loss)+"")






