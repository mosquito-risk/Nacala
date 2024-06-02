from download import download_data
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm


# Download data
# download_data('dinov2_features', 'datasets')

# Prepare data
train_data = np.load('datasets/dinov2_features/train.npy')
valid_data = np.load('datasets/dinov2_features/valid.npy')

X_train, X_valid = torch.Tensor(train_data[:, :-1]), torch.Tensor(valid_data[:, :-1])
y_train = torch.from_numpy(train_data[:, -1]).long() - 1
y_valid = torch.from_numpy(valid_data[:, -1]).long() - 1
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

device = "cpu"
if torch.cuda.is_available():
    print("CUDA is available..")
    device = "cuda"


# Model
class LogisticRegressionPytorch(torch.nn.Module):
    def __init__(self, d, m):
        super(LogisticRegressionPytorch, self).__init__()
        # LAYER DEFINITION MISSING
        self.linear = torch.nn.Linear(d, m)

    def forward(self, x):
        # RETURN VALUE MISSING
        return self.linear(x)

# Parameters for training
d = 768
m = 5
learning_rate = 0.001
model = LogisticRegressionPytorch(d, m)
print(model)
criterion = torch.nn.functional.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

f1_score = F1Score(task='multiclass', num_classes=5, absent_score=0, average='macro').to(device)
accuracy_m = Accuracy(task='multiclass', top_k=1, num_classes=5).to(device)
out_folder = 'temp'
writer = SummaryWriter(f"./{out_folder}/{'test'}")
best_f1 = -1


no_epochs = 5000  # Number of training steps

for epoch in tqdm(range(no_epochs)):  # Loop over the dataset multiple times
    f1_score.reset()
    accuracy_m.reset()

    model.train()
    optimizer.zero_grad()

    # Forward + backward + optimize
    outputs = model(X_train)
    loss = criterion(torch.squeeze(outputs), y_train, reduction='mean', label_smoothing=0.1)
    loss.backward()
    optimizer.step()

    # update metrics
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    f1_score.update(outputs, y_train)
    accuracy_m.update(outputs, y_train)

    # log metrics
    writer.add_scalar('train/loss', loss.item(), epoch)
    writer.add_scalar('train/f1', f1_score.compute(), epoch)
    writer.add_scalar('train/acc', accuracy_m.compute(), epoch)

    # Validation
    f1_score.reset()
    accuracy_m.reset()

    model.eval()
    outputs = model(X_valid)
    loss = criterion(torch.squeeze(outputs), y_valid, reduction='mean', label_smoothing=0.1)
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    f1_score.update(outputs, y_valid)
    accuracy_m.update(outputs, y_valid)

    # log metrics
    valid_f1 = f1_score.compute()
    writer.add_scalar('valid/loss', loss.item(), epoch)
    writer.add_scalar('valid/f1', valid_f1, epoch)
    writer.add_scalar('valid/acc', accuracy_m.compute(), epoch)
    # writer.add_image('valid/cm', conf.compute().cpu().numpy(), epoch)

    # Save best model
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        torch.save(model.state_dict(), f'./{out_folder}/best_model')

    # scheduler.step()

# print best F1 score
print(f"Best F1 score: {best_f1}")





