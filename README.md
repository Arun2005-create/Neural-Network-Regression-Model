# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1. The problem involves building a neural network model to predict continuous numerical values rather than categories.
2. A dataset with input features and corresponding target values is required for training.
3. The data must be preprocessed, including cleaning, normalization, and splitting into training and testing sets.
4. A suitable neural network architecture with input, hidden, and output layers is designed.
5. The model learns patterns by adjusting weights using a loss function like Mean Squared Error.
6. Training is performed using an optimization algorithm such as gradient descent or Adam.
7. Finally, the model is evaluated on test data to measure prediction accuracy and generalization.

## Neural Network Model

<img width="930" height="643" alt="image" src="https://github.com/user-attachments/assets/eaebc717-17d1-4762-89a3-b584d2b05979" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: ARUN KUMAR B
### Register Number: 212223230021
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def arun():
    print("Name: ARUN KUMAR B")
    print("Register Number: 212223230021")

dataset1 = pd.read_csv('DL-Exp1 - Sheet1 (1).csv')
print(dataset1)

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
        aakif()
        print("Neural Network Regression Model Initialized")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

ai_arun = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(ai_arun.parameters(), lr=0.01)

def train_model(model, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_arun, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_arun(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_arun.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_arun(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()

arun()
print(f'Prediction for input 9: {prediction}')
```
## Dataset Information
<img width="167" height="416" alt="image" src="https://github.com/user-attachments/assets/baa37749-11a8-4b51-8df4-7ac7c81ed7a5" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="452" height="299" alt="112" src="https://github.com/user-attachments/assets/d13c73a2-477e-4837-bec8-fbd450157c98" />


### New Sample Data Prediction

<img width="758" height="580" alt="113" src="https://github.com/user-attachments/assets/9125f443-7cb2-4102-98c8-ea3869ef065b" />



## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
