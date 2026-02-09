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

<img width="1357" height="874" alt="544129551-f935354f-d0a9-47fa-b725-9957b894329c" src="https://github.com/user-attachments/assets/0287193e-8acd-4bdf-b511-7ca373136d2a" />


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
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x 


# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()


    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information
<img width="754" height="676" alt="543569295-99156218-bd6d-4489-b1be-8e393d6ef39b" src="https://github.com/user-attachments/assets/6dcdf1ac-a871-4c68-83f4-88164fe61987" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="1069" height="685" alt="543569361-45c3572b-4274-45d6-8fd0-e8b3483de70b" src="https://github.com/user-attachments/assets/4cc28a8f-be40-4b12-b3a0-5b712c960ad7" />
<img width="962" height="309" alt="543569389-b430eac8-2dec-4c62-a6e6-9f44691afabd" src="https://github.com/user-attachments/assets/5c0734d0-6e4e-4a5a-99be-ee21cf863e65" />



### New Sample Data Prediction
<img width="855" height="125" alt="543569463-0495a844-058d-428c-8b8b-de6296669ab9" src="https://github.com/user-attachments/assets/31051042-cd77-4e9d-af0f-f84c399c5436" />



## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
