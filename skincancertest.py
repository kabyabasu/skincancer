from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Precision,Recall

# Set the device
device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

def test_dataprocessing(location):
    test_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    dataset_test = ImageFolder(
        location,transform=test_transform
    )

    dataloader_test = DataLoader(dataset_test,batch_size=10,shuffle=True)

    return dataloader_test

# Load data
test_data = test_dataprocessing('./skin_cancer/Test')




class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classification = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classification(x)
        return x
    

net = Net(num_classes=9)
net.load_state_dict(torch.load('./skincancermodel.pth'))
net.eval()



def eval_macros():
    metric_precision = Precision(task = "multiclass", num_classes= 9, average="macro")
    metric_recall = Recall(task = "multiclass", num_classes=9,average="macro")




    with torch.no_grad():

        for images,labels in test_data:
            outputs = net(images)
            _,preds = torch.max(outputs,1)
            metric_precision(preds,labels)
            metric_recall(preds,labels)

    precision = metric_precision.compute()
    recall  = metric_recall.compute()


    print(f"Overall Precision : {precision} Overll  Recall : {recall}")


def eval_indigroup():

    metric_precision = Precision(task = "multiclass", num_classes= 9, average=None)
    metric_recall = Recall(task = "multiclass", num_classes=9,average=None)




    with torch.no_grad():

        for images,labels in test_data:
            outputs = net(images)
            _,preds = torch.max(outputs,1)
            metric_precision(preds,labels)
            metric_recall(preds,labels)

    precision = metric_precision.compute()
    recall  = metric_recall.compute()

    precision_per_class = {k: precision[v].item() for k, v in test_data.dataset.class_to_idx.items()}
    recall_per_class = {k: recall[v].item() for k, v in test_data.dataset.class_to_idx.items()}


    print(f"Precision : \n {precision_per_class} \n\n  Recall : \n {recall_per_class}")



eval_macros()
eval_indigroup()
