import torch 
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset,Dataset
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_experiments = 5
prob_tensors = []

for i in range(num_experiments):
    filename = f"final_layer_predictions/gpt2_f_experiment_{i}_drug_data_train.pt"
    prob_tensors.append(torch.from_numpy(torch.load(filename)))
prob = torch.cat(prob_tensors, dim=1).to(device)


eval_prob_tensors = []

for i in range(num_experiments):
    filename = f"final_layer_predictions/gpt2_f_experiment_{i}_drug_data_test.pt"
    eval_prob_tensors.append(torch.from_numpy(torch.load(filename)))

eval_prob = torch.cat(eval_prob_tensors,dim=1).to(device)

dataset = load_dataset("lewtun/drug-reviews")
eval_dataset = dataset["train"]
reviews_train = eval_dataset['review']
ratings_train = eval_dataset['rating']
drug_data_train = {'label': [int(rating-1) for rating in ratings_train],'text': reviews_train}
drug_data_train = Dataset.from_dict(drug_data_train)
labels = drug_data_train['label']

eval_dataset = dataset["test"]
reviews_eval = eval_dataset['review']
ratings_eval = eval_dataset['rating']
drug_data_eval = {'label': [int(rating-1) for rating in ratings_eval],'text': reviews_eval}
drug_data_eval = Dataset.from_dict(drug_data_eval)
eval_labels = drug_data_eval['label']

# Un comment below if things are not working

# Define the size of the validation set (e.g., 20% of the total data)
validation_size = int(0.2 * len(eval_dataset))
indices = list(range(len(eval_dataset)))
test_indices = indices[validation_size:]
validation_indices = indices[:validation_size]

# Split the shuffled training dataset into training and validation se
eval_dataset = drug_data_eval.map(lambda example, idx: {'index': idx, **example}, with_indices=True).shuffle(42)
#eval_dataset = dataset['test'].shuffle(seed = 42)
test_dataset = eval_dataset.select(test_indices)
validation_dataset = eval_dataset.select(validation_indices)

# EVAL_PROB needs to be shuffled correctly
shuffled_indices = eval_dataset['index']
print(shuffled_indices[0:10])
#print(eval_dataset.list_indexes())
#print(eval_dataset[0])
eval_prob = eval_prob[shuffled_indices]
test_indices_tensor = torch.tensor(test_indices)
validation_indices_tensor = torch.tensor(validation_indices)
test_prob = eval_prob[test_indices_tensor]
validation_prob = eval_prob[validation_indices_tensor]

#test_dataset = dataset["test"]
test_labels = test_dataset['label']
validation_labels = validation_dataset['label']
#print(type(validation_labels))

labels = torch.tensor(labels,dtype=torch.long).to(device)
validation_labels = torch.tensor(validation_labels,dtype=torch.long).to(device)
test_labels = torch.tensor(test_labels,dtype=torch.long).to(device)
eval_labels = torch.tensor(eval_labels,dtype=torch.long).to(device)


class NeuralNetwork(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(NeuralNetwork,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim,50),
            nn.Tanh(),
	    nn.Linear(50, 40),
	    nn.Tanh(),
            nn.Linear(40,20),
            nn.Tanh(),
	    nn.Linear(20, 10),
	    nn.Tanh(),
            nn.Linear(10,output_dim)
                    )
    def forward(self,x):
        return self.layer(x)

model = NeuralNetwork(input_dim=10 * num_experiments,hidden_dim=20,output_dim=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def compute_accuracy(predictions,labels):
    pred_labels = np.argmax(predictions,axis=1)
    accuracy = (pred_labels == labels).mean()
    return accuracy

num_epochs=100

for epoch in range(num_epochs):
    model.train()
    predictions = model(prob)
    loss = loss_fn(predictions,labels)
    predictions_cpu = predictions.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    accuracy = compute_accuracy(predictions_cpu,labels_cpu)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    # Evaluation
    model.eval()
    eval_predictions = model(eval_prob)
    eval_loss = loss_fn(eval_predictions, eval_labels)
    eval_predictions_cpu = eval_predictions.detach().cpu().numpy()
    eval_labels_cpu = eval_labels.detach().cpu().numpy()
    eval_accuracy = compute_accuracy(eval_predictions_cpu, eval_labels_cpu)
    

    model.eval()
    validation_predictions = model(validation_prob)
    validation_loss = loss_fn(validation_predictions, validation_labels)
    validation_predictions_cpu = validation_predictions.detach().cpu().numpy()
    validation_labels_cpu = validation_labels.detach().cpu().numpy()
    validation_accuracy = compute_accuracy(validation_predictions_cpu, validation_labels_cpu)
    
    model.eval()
    test_predictions = model(test_prob)
    test_loss = loss_fn(test_predictions, test_labels)
    test_predictions_cpu = test_predictions.detach().cpu().numpy()
    test_labels_cpu = test_labels.detach().cpu().numpy()
    test_accuracy = compute_accuracy(test_predictions_cpu, test_labels_cpu)
    print(f'Loss {loss} and accuracy {accuracy}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy.item():.4f}, Validation Loss: {validation_loss.item():.4f}, Validation Accuracy: {validation_accuracy.item():.4f}')
    print()



# Reshape test_prob to have 5 sets of 10 columns
test_prob_reshaped = test_prob.reshape(test_prob.shape[0], 5, 10)

# Split test_prob_reshaped along the second axis to get 5 tensors
tensors_list = np.split(test_prob_reshaped, 5, axis=1)

# Calculate average probabilities
avg_probs = sum(tensors_list) / len(tensors_list)
avg_probs = avg_probs.squeeze(dim=1)

# Convert avg_probs to a PyTorch tensor and move it to the appropriate device
avg_probs_tensor = torch.tensor(avg_probs).to(device)

# Calculate predicted labels
pred_labels = torch.argmax(avg_probs_tensor, dim=1)

# Calculate accuracy
correct_predictions = (pred_labels == test_labels).sum().item()
total_predictions = test_labels.size(0)
accuracy = correct_predictions / total_predictions

print(f'Test Accuracy: {accuracy:.4f}')



# Reshape test_prob to have 5 sets of 10 columns
validation_prob_reshaped = validation_prob.reshape(validation_prob.shape[0], 5, 10)

# Split test_prob_reshaped along the second axis to get 5 tensors
tensors_list = np.split(validation_prob_reshaped, 5, axis=1)

# Calculate average probabilities
avg_probs = sum(tensors_list) / len(tensors_list)
avg_probs = avg_probs.squeeze(dim=1)

# Convert avg_probs to a PyTorch tensor and move it to the appropriate device
avg_probs_tensor = torch.tensor(avg_probs).to(device)

# Calculate predicted labels
pred_labels = torch.argmax(avg_probs_tensor, dim=1)

# Calculate accuracy
correct_predictions = (pred_labels == validation_labels).sum().item()
total_predictions = validation_labels.size(0)
accuracy = correct_predictions / total_predictions

print(f'Validation Accuracy: {accuracy:.4f}')
