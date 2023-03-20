## ---------------------------------------------------------------------Useful import-----------------------------------------------------------
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import warnings

warnings.filterwarnings("ignore")

## -------------------------------------------------             speedup GPU -------------------------------------------------------------------------

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## ----------------------------------------------------------------------Loading data---------------------------------------------------------------
inputs = np.load('/home/ubuntu/fedLning/pysyft/data/inputs.npy')
labels = np.load('/home/ubuntu/fedLning/pysyft/data/labels.npy')
# inputs = np.load('/home/ubuntu/fedLning/pysyft/data/inputs.npy').to(device)
# labels = np.load('/home/ubuntu/fedLning/pysyft/data/labels.npy').to(device)

VOCAB_SIZE = int(inputs.max()) + 1

## ----------------------------------------------------------Training model with Federated learning--------------------------------------------

### Training and model hyperparameters
# Training params
EPOCHS = 15
CLIP = 5 # gradient clipping - to avoid gradient explosion (frequent in RNNs)
lr = 0.1
BATCH_SIZE = 32

# Model params
EMBEDDING_DIM = 50
HIDDEN_DIM = 10
DROPOUT = 0.2


### Initiating virtual workers with Pysyft
import syft as sy

labels = torch.tensor(labels)
inputs = torch.tensor(inputs)
# labels = torch.tensor(labels).to(device)
# inputs= torch.tensor(inputs).to(device)

# splitting training and test data
pct_test = 0.2

train_labels = labels[:-int(len(labels)*pct_test)]
train_inputs = inputs[:-int(len(labels)*pct_test)]

test_labels = labels[-int(len(labels)*pct_test):]
test_inputs = inputs[-int(len(labels)*pct_test):]


# Hook that extends the Pytorch library to enable all computations with pointers of tensors sent to other workers
hook = sy.TorchHook(torch)

# Creating 2 virtual workers
bob = sy.VirtualWorker(hook, id="bob")
anne = sy.VirtualWorker(hook, id="anne")

# threshold indexes for dataset split (one half for Bob, other half for Anne)
train_idx = int(len(train_labels)/2)
test_idx = int(len(test_labels)/2)

# Sending toy datasets to virtual workers
bob_train_dataset = sy.BaseDataset(train_inputs[:train_idx], train_labels[:train_idx]).send(bob)
anne_train_dataset = sy.BaseDataset(train_inputs[train_idx:], train_labels[train_idx:]).send(anne)
bob_test_dataset = sy.BaseDataset(test_inputs[:test_idx], test_labels[:test_idx]).send(bob)
anne_test_dataset = sy.BaseDataset(test_inputs[test_idx:], test_labels[test_idx:]).send(anne)

# Creating federated datasets, an extension of Pytorch TensorDataset class
federated_train_dataset = sy.FederatedDataset([bob_train_dataset, anne_train_dataset])
federated_test_dataset = sy.FederatedDataset([bob_test_dataset, anne_test_dataset])

# Creating federated dataloaders, an extension of Pytorch DataLoader class
federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
federated_test_loader = sy.FederatedDataLoader(federated_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
# federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=True, batch_size=BATCH_SIZE).to(device)
# federated_test_loader = sy.FederatedDataLoader(federated_test_dataset, shuffle=False, batch_size=BATCH_SIZE).to(device)

### Creating simple GRU (1-layer) model with sigmoid activation for classification task
from handcrafted_GRU import GRU

# Initiating the model
model = GRU(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT)
# model = GRU(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT).to(device)


## ------------------------------------------------------------------------------------------Training--------------------------------------------------------------------------

# Defining loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
# criterion = nn.BCELoss().to(device)
# optimizer = optim.SGD(model.parameters(), lr=lr).to(device)


for e in range(EPOCHS):
    
    ######### Training ##########
    
    losses = []
    # Batch loop
    for inputs, labels in federated_train_loader:
        # Location of current batch
        worker = inputs.location

        # Initialize hidden state and send it to worker
        h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker)
        
        # Send model to current worker
        model.send(worker)
        # model.send(worker).to(device)

        # Setting accumulated gradients to zero before backward step
        optimizer.zero_grad()
        # Output from the model
        output, _ = model(inputs, h)
        # Calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # Clipping the gradient to avoid explosion
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        # Backpropagation step
        optimizer.step() 
        # Get the model back to the local worker
        model.get()

        losses.append(loss.get())
    
    ######## Evaluation ##########
    
    # Model in evaluation mode
    model.eval()
    # model.eval().to(device)

    with torch.no_grad():
        test_preds = []
        test_labels_list = []
        eval_losses = []

        for inputs, labels in federated_test_loader:
            # get current location
            worker = inputs.location
            # Initialize hidden state and send it to worker
            h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker)   
            # h = torch.Tensor(np.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker).to(device)   

            # Send model to worker
            model.send(worker)
            
            output, _ = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            eval_losses.append(loss.get())
            preds = output.squeeze().get()
            test_preds += list(preds.numpy())
            test_labels_list += list(labels.get().numpy().astype(int))
            # Get the model back to the local worker
            model.get()
        
        score = roc_auc_score(test_labels_list, test_preds)
    
    print("Epoch {}/{}...  \
    AUC: {:.3%}...  \
    Training loss: {:.5f}...  \
    Validation loss: {:.5f}".format(e+1, EPOCHS, score, sum(losses)/len(losses), sum(eval_losses)/len(eval_losses)))
    
    # model = model.to(device)
    model.train()
