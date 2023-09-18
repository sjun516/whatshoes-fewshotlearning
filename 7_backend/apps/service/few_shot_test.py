import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
    
def find_class():
        class SiameseNetworkTestDataset(Dataset):
            i = 1
            j = 1
            cnt = 0

            # 클래스의 인스턴스를 초기화하는 메서드
            def __init__(self, imageFolderDataset, transform=None, should_invert=True):
                self.imageFolderDataset = imageFolderDataset
                self.transform = transform
                self.should_invert = should_invert

            def __getitem__(self, index):

                if SiameseNetworkTestDataset.cnt == 10:
                    SiameseNetworkTestDataset.cnt = 0
                    SiameseNetworkTestDataset.i += 1
                    SiameseNetworkTestDataset.j = 1

                img0_tuple = ['./apps/resource/query/query.jpg']
                img1_tuple = ['./apps/resource/support/s{}/img{}.jpg'.format(SiameseNetworkTestDataset.i,
                                                                             SiameseNetworkTestDataset.j)]

                SiameseNetworkTestDataset.cnt += 1
                SiameseNetworkTestDataset.j += 1

                print(img0_tuple[0])
                print(img1_tuple[0])
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

                return img0, img1

            def __len__(self):
                return 10000

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

            def __init__(self, margin=2.0):  # margin은 모델의 임계값
                super(ContrastiveLoss, self).__init__()
                self.margin = margin

            def forward(self, output1, output2, label):
                euclidean_distance = F.pairwise_distance(output1, output2,
                                                         keepdim=True)  # cnn으로 추출한 벡터의 거리를 유클리드 거리법을 통하여 계산
                # output1과 output2가 같은 클래스인지(0) 다른 클래스인지(1)에 따라 손실을 계산
                loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                              (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                                  2))

                return loss_contrastive

            # batch_size는 training_data 중에 네트워크에 들어가는 이미지 수 (batch는 학습 데이터셋에서 랜덤하게 뽑는다)
        net = SiameseNetwork()
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0005)

        counter = []
        loss_history = []
        iteration_number = 0

        # print("--------------------------테스트---------------------------------")

        device = torch.device('cpu')
        net.load_state_dict(torch.load('./apps/resource/model0507_1.pth', map_location=device))

        folder_dataset_test = dset.ImageFolder(root="./apps/resource/support")
        siamese_dataset = SiameseNetworkTestDataset(imageFolderDataset=folder_dataset_test, transform=transforms.Compose(
            [transforms.Resize((128, 64)), transforms.ToTensor()]), should_invert=False)

        # 10개 클래스 매칭
        # test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
        # dataiter = iter(test_dataloader)
        # count = 0
        # index = 0
        # arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # for i in range(100):
        #     count += 1
        #     x0, x1 = next(dataiter)
        #     concatenated = torch.cat((x0, x1), 0)
        #     output1, output2 = net(Variable(x0), Variable(x1))
        #     euclidean_distance = F.pairwise_distance(output1, output2).detach().numpy()
        #     arr[index] += euclidean_distance
        #     if(count == 10):
        #         count = 0
        #         index += 1

        #     print("--------------------")

        # print(arr.index(min(arr)) + 1)
        # return arr.index(min(arr)) + 1

        # 13개 클래스 매칭
        test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
        dataiter = iter(test_dataloader)
        count = 0
        index = 0
        arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(130):
            count += 1
            x0, x1 = next(dataiter)
            concatenated = torch.cat((x0, x1), 0)
            output1, output2 = net(Variable(x0), Variable(x1))
            euclidean_distance = F.pairwise_distance(output1, output2).detach().numpy()
            arr[index] += euclidean_distance
            if(count == 10):
                count = 0
                index += 1

            print("--------------------")

        print(arr.index(min(arr)) + 1)
        return arr.index(min(arr)) + 1
