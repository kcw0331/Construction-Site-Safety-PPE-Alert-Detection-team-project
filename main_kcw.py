import copy
import sys

import matplotlib.pyplot as plt
import torch
import albumentations as A
import os
import pandas as pd
from tqdm import tqdm # 이렇게 해줘야 오류가 없다.
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from customdataset import customDataset
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy # 이번에 이 loss를 사용한다 선생님이 오버피팅이 덜난다고 하심.

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### 0. aug setting -> train val test
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(height=224, width=224),
        # A.RandomShadow(p=0.5),
        # A.RandomFog(p=0.4),
        # A.RandomSnow(p=0.4),
        # A.RandomBrightnessContrast(p=0.5),
        # A.Rotate(25, p=0.7),
        # A.ShiftScaleRotate(shift_limit=5, scale_limit=0.09, rotate_limit=25, p=1),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    ### 1. Loding classification Dataset
    train_dataset = customDataset(".\\dataset\\train\\", transform = train_transform)
    val_dataset = customDataset(".\\dataset\\val\\", transform = val_transform)
    test_dataset = customDataset(".\\dataset\\test\\", transform = test_transform)

    ### def visualize_augmentations()
    def visulize_augmentations(dataset, idx=0, samples=20, cols=5):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([t for t in dataset.transform
                                       if not isinstance(
                t, (A.Normalize, ToTensorV2)
        )])
        rows = samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))
        for i in range(samples):
            image, _ = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()
    # transform한 것을 시각화 해본다.
    # visulize_augmentations(train_dataset)
    # exit()

    ### 2. Data Loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    ## 모델 생성
    # 선생님이 아래 주신 HUB_URL은 라이트한 버전이라고 말씀하심.
    import torch.nn as nn

    # # train model
    # # resnet18(batch_size=32)
    # net = models.resnet18(pretrained=True)
    # net.fc = nn.Linear(in_features=512, out_features=5)
    # net.to(device)

    # test model
    # net = models.resnet18(pretrained=False)
    # net.fc = nn.Linear(in_features=512, out_features=5)
    # net.load_state_dict(torch.load("./hogi/best.pt"))
    # net.to(device)

    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(in_features=2048, out_features=5)
    net.load_state_dict(torch.load("./hogi/best.pt"))
    net.to(device)

    ### 4. epoch. optim, loss
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01) # net의 파라메타를 넣어줘야 한다.
    epochs = 50

    best_val_acc = 0.0
    ## train 모델
    # train_steps = len(train_loader)
    # val_steps = len(val_loader)
    # save_path = "best.pt"
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # # 저장을 해준다.
    # dfForAccuracy = pd.DataFrame(index=list(range(epochs)), columns=["Epoch", "Train_Accuracy", "Train_Loss", "Val_Accuracy", "Val_Loss"])

    # if os.path.exists(save_path):
    #     best_val_acc = max(pd.read_csv("./modelAccuracy.csv")["Accuracy"].tolist())

    # for epoch in range(epochs):
    #     runing_loss = 0
    #     val_acc = 0
    #     train_acc = 0

    #     net.train()
    #     # tqdm은 프로세스 진행 상태를 나타내준다.
    #     train_bar = tqdm(train_loader, file=sys.stdout, colour="green")
    #     for step, data in enumerate(train_bar):
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)

    #         loss = loss_function(outputs, labels)
    #         optimizer.zero_grad()
    #         train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item() # torch.max로 사용해도 된다., 텐서값 뽑을 때, item()을 해준다.
    #         loss.backward()
    #         optimizer.step()
    #         runing_loss += loss.item()

    #         train_bar.desc = f"train epoch [{epoch+1}/{epochs}], loss >> {loss.data:.3f}"

    #     # val을 하기 위해서 eval모드로 전환
    #     net.eval()
    #     with torch.no_grad():
    #         val_loss = 0
    #         valid_bar = tqdm(val_loader, file=sys.stdout, colour="red") # val은 빨간색으로 해준다.
    #         for data in valid_bar:
    #             images, labels = data
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = net(images)
    #             loss = loss_function(outputs, labels)
    #             val_loss += loss.item()

    #             val_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

    #     # val에서는 평가만 할꺼기 때문에 loss를 하지않고 accuracy만 해주었다.
    #     val_accuracy = val_acc / len(val_dataset)
    #     train_accuracy = train_acc / len(train_dataset)

    #     dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
    #     dfForAccuracy.loc[epoch, "Train_Accuracy"] = round(train_accuracy, 3) # round는 반올림해준다.
    #     dfForAccuracy.loc[epoch, "Train_Loss"] = round(runing_loss / train_steps, 3)
    #     dfForAccuracy.loc[epoch, "Val_Accuracy"] = round(val_accuracy, 3)
    #     dfForAccuracy.loc[epoch, "Val_Loss"] = round(val_loss / val_steps, 3)
    #     print(f"epoch [{epoch+1}/{epochs}] trian_loss{(runing_loss / train_steps):.3f} train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")

    #     if val_accuracy > best_val_acc: # best를 loss로하는 경우도 있고 accuracy로 하는 경우도 있다.
    #         best_val_acc = val_accuracy
    #         torch.save(net.state_dict(), save_path)

    #     if epoch == epochs - 1:
    #         dfForAccuracy.to_csv("./modelAccuracy.csv", index=False)

    # torch.save(net.state_dict(), "./last.pt")

    classes = ('down_down', 'down_up', 'sideways', 'up_down', 'up_up')

    def acc_function(correct, total):
        acc = correct / total * 100
        return acc
    
    def test(model, data_loader, device):  # <- 기본 standard이다.
        model.eval()
    
        # 개별 acc를 구하기 위해서 작성
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for i, (image, label) in enumerate(data_loader):
                images, labels = image.to(device), label.to(device)
                output = model(images)  # output에서는 예측된 값이 나온다.
                _, predictions = torch.max(output, 1)  # output에서 max인걸 뽑는다.

                # 각 분류별로 올바른 예측 수를 모아준다.
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

                # {'down_down': 0, 'down_up': 1, 'sideways': 2, 'up_down': 3, 'up_up': 4}
              
        # 각 분류별 정확도(accuracy)를 출력합니다
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    test(net,test_loader, device)

if __name__ == '__main__':
    main()