
import torch 
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

# |%%--%%| <4eFd07rzKi|vaAeCG8sHo>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# |%%--%%| <vaAeCG8sHo|FWXxH5s3Ea>

prob_1 = torch.from_numpy(torch.load("test_probs_epoch_1.pt"))
prob_2 = torch.from_numpy(torch.load("test_probs_epoch_5.pt"))
prob = torch.cat([prob_1,prob_2],dim=1).to(device)

# |%%--%%| <FWXxH5s3Ea|pyEhc1nzic>

dataset = load_dataset("imdb")
eval_dataset = dataset["test"].shuffle(seed=42)
labels = eval_dataset['label']

# |%%--%%| <pyEhc1nzic|2V0As3fInx>

labels = torch.tensor(labels,dtype=torch.float32).to(device)

# |%%--%%| <2V0As3fInx|kVRgtsv6ji>

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(NeuralNetwork,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)        
        )
    def forward(self,x):
        return self.layer(x)

# |%%--%%| <kVRgtsv6ji|g69MxKobaP>

model = NeuralNetwork(input_dim=4,hidden_dim=2,output_dim=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# |%%--%%| <g69MxKobaP|y4tpjZq0wH>

def binary_accuracy(logits, true_labels):
    probs = torch.sigmoid(logits)
    
    preds = probs > 0.5  # Threshold of 0.5
    
    correct_predictions = torch.eq(preds.squeeze(), true_labels.bool()).float()  # Ensure true_labels are boolean or 0/1
    
    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(true_labels)
    return accuracy


# |%%--%%| <y4tpjZq0wH|0jqB8fxSoa>

num_epochs = 100000

# |%%--%%| <0jqB8fxSoa|G4xsu1MEwa>

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    predictions = model(prob)
    loss = criterion(predictions.squeeze(), labels)
    accuracy = binary_accuracy(predictions, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')


# |%%--%%| <G4xsu1MEwa|0ZVPA2q9lW>


