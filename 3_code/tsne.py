import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.manifold import TSNE


# 이미지 출력함수
def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 그래프 출력 함수
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


# 모델 파라미터 정의
class Config():
    training_dir = "./trainingset"
    testing_dir = "./test/supportset"
    # testing_dir = "./archive/ORL/testing"
    train_batch_size = 50
    train_number_epochs = 2


# 이미지 쌍을 입력으로 받아서
# 이미지 쌍 간의 유사도(similarity)를 예측하는 Siamese Neural Network을 학습시키기 위한 데이터셋 클래스
class SiameseNetworkDataset(Dataset):

    # 클래스의 인스턴스를 초기화하는 메서드
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset  # 이미지 저장된 폴더경로
        self.transform = transform  # 이미지 변환
        self.should_invert = should_invert  # 이미지 쌍 반전시킬지

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # img0 이미지를 무작위로 가져옴
        should_get_same_class = random.randint(0, 1)  # 변수에 0이나 1을 랜덤으로 할당
        if should_get_same_class:  # 1이 선택되면 같은 클래스의 img1 이미지 선택
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:  # 인덱스0은 이미지파일 인덱스1은 이미지파일이 속한 클래스폴더
                    break
        else:  # 0이 선택되면 다른 클래스의 이미지로 img1을 선택
            while True:

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")  # grayscale
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:  # 이미지 데이터 전처리
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # print(img0_tuple[0])
        # print(img1_tuple[0])
        # print(torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)))
        # print()
        # print(img0.size())

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetworkTestDataset(Dataset):
    k = 1
    i = 1
    j = 2
    cnt = 0

    # 클래스의 인스턴스를 초기화하는 메서드
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        # img0_tuple = [Config.testing_dir + '/s{}/img8.jpg'.format(SiameseNetworkTestDataset.k), SiameseNetworkTestDataset.k-1]

        if SiameseNetworkTestDataset.cnt == 5:
            SiameseNetworkTestDataset.cnt = 0
            SiameseNetworkTestDataset.k += 1
            SiameseNetworkTestDataset.i = 1
            SiameseNetworkTestDataset.j = 2

        img0_tuple = [
            './test/supportset/s{}/img{}.jpg'.format(SiameseNetworkTestDataset.k, SiameseNetworkTestDataset.i),
            SiameseNetworkTestDataset.k - 1]
        img1_tuple = [
            './test/supportset/s{}/img{}.jpg'.format(SiameseNetworkTestDataset.k, SiameseNetworkTestDataset.j),
            SiameseNetworkTestDataset.k - 1]

        SiameseNetworkTestDataset.cnt += 1
        SiameseNetworkTestDataset.i += 2
        SiameseNetworkTestDataset.j += 2

        print(img0_tuple[0])
        print(img1_tuple[0])
        print(SiameseNetworkTestDataset.k)
        # print("--------------------")
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # img0 = img0.convert("L") # grayscale
        # img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # print(img0.size())

        return img0, img1, SiameseNetworkTestDataset.k

    def __len__(self):
        return 10000


# folder_dataset = dset.ImageFolder(root=Config.training_dir)  # 트레이닝 데이터 폴더
#
# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
#                                         transform=transforms.Compose(
#                                             [transforms.Resize((128, 64)), transforms.ToTensor()]),
#                                         # 이미지 크기 100, 100 사이즈로 reshape, Numpy -> tensor(0에서 1사이의 값으로 정규화)
#                                         should_invert=False)


class SiameseNetwork(nn.Module):  # Siamese neural network 모델 클래스
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(  # 3개의 레이어를 가진 은닉층
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, kernel_size=3),  # 1개의 input_channel, 4개의 output_channel, 3*3 kernel_size
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, kernel_size=3),  # 4개의 input_channel, 8개의 output_channel, 3*3 kernel_size
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),  # 8개의 input_channel, 8개의 output_channel, 3*3 kernel_size
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
        )

        self.fc1 = nn.Sequential(  # 3개의 레이어를 가지는 완전연결층
            nn.Linear(128 * 16 * 8, 128),  # 500개의 입력을 받아 500개의 출력을 갖는 레이어
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),  # 500개의 입력을 받아 500개의 출력을 갖는 레이어
            nn.ReLU(inplace=True),

            nn.Linear(64, 10))  # 500개의 입력을 받아 5개의 출력을 갖는 레이어

    def forward_once(self, x):  # 한 이미지를 CNN에 입력으로 넣는다
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):  # forward_once 메서드를 두 번 호출하여 두 이미지에 대해 각각 벡터값을 출력하고 반환
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):  # 대조 손실 클래스
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):  # margin은 모델의 임계값
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)  # cnn으로 추출한 벡터의 거리를 유클리드 거리법을 통하여 계산
        # output1과 output2가 같은 클래스인지(0) 다른 클래스인지(1)에 따라 손실을 계산
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

    # batch_size는 training_data 중에 네트워크에 들어가는 이미지 수 (batch는 학습 데이터셋에서 랜덤하게 뽑는다)


# network input = (batch_size, channel, height, width) = (8, 1, 100, 100)
# train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)

net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

# for epoch in range(0,Config.train_number_epochs):
#     for i, data in enumerate(train_dataloader,0):
#         # print(i)
#         # print()
#         img0, img1 , label = data
#         img0, img1 , label = img0, img1 , label
#         # print(img0.size())
#         # print(img0)
#         optimizer.zero_grad() # 옵티마이저 초기화
#         output1,output2 = net(img0,img1) # 모델에 학습 이미지 쌍 넣기
#         # print("output1", output1.size())
#         # print(output1)

#         # print("output2", output2.size())
#         # print(output2)

#         loss_contrastive = criterion(output1,output2,label) # loss 계산
#         loss_contrastive.backward() # 역전파 작업 수행(가중치를 계산)
#         optimizer.step() # 모델의 가중치(weights)를 갱신
#         if i %10 == 0 :
#             print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
#             iteration_number +=10
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())

# show_plot(counter,loss_history)
# torch.save(net.state_dict(), 'model.pth')

net = SiameseNetwork()
device = torch.device('cpu')
net.load_state_dict(torch.load('model0408_3.pth', map_location=device))

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkTestDataset(imageFolderDataset=folder_dataset_test, transform=transforms.Compose(
    [transforms.Resize((128, 64)), transforms.ToTensor()]), should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)

X = np.empty((0, 10), float)
Y = np.empty((0, 1), float)

for i in range(50):
    x0, x1, y = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)
    output1, output2 = net(Variable(x0), Variable(x1))
    out1 = output1.detach().numpy()
    out2 = output2.detach().numpy()
    X = np.vstack([X, out1])
    X = np.vstack([X, out2])
    Y = np.vstack([Y, y])
    Y = np.vstack([Y, y])

    # print(X.shape)
    # print(Y)
    # print(output1)
    # print(output2)
    # print(out1)
    # print(out2)
    print("--------------------")

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
# print(X_tsne.shape)
# print(Y.shape)

# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y)
for i in range(100):
    if (i < 10):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='violet', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='violet', s=200, label='Class1')
    elif (10 <= i < 20):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='red', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='red', s=200, label='Class2')
    elif (20 <= i < 30):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='pink', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='pink', s=200, label='Class3')
    elif (30 <= i < 40):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='orange', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='orange', s=200, label='Class4')
    elif (40 <= i < 50):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='blue', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='blue', s=200, label='Class5')
    elif (50 <= i < 60):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='green', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='green', s=200, label='Class6')
    elif (60 <= i < 70):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='purple', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='purple', s=200, label='Class7')
    elif (70 <= i < 80):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='grey', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='grey', s=200, label='Class8')
    elif (80 <= i < 90):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='yellow', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='yellow', s=200, label='Class9')
    elif (90 <= i < 100):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='brown', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='brown', s=200, label='Class10')
    elif (100 <= i < 110):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='#17becf', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='#17becf', s=200, label='Query')
    else:
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='#e35f62', s=200)
        if (i % 10 == 0):
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color='#e35f62', s=200, label='Class11')
    plt.text(X_tsne[i, 0] - 0.065, X_tsne[i, 1] - 0.1, i % 10 + 1)

plt.legend()
plt.show()