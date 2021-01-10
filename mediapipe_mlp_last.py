import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pytorch_lightning.metrics.functional import accuracy

class HandsDataset(torch.utils.data.Dataset):

    def __init__(self, data_num, transform=None):
        self.transform = transform
        self.data_num = data_num
        self.data = []
        self.label = []
        #df = pd.read_csv('./hands/sample_hands6__.csv', sep=',')
        df = pd.read_csv('./hands/sample_hands7.csv', sep=',')
        print(df.head(3)) #データの確認
        df = df.astype(int)
        x = []
        for j in range(self.data_num):
            x_ = []
            for i in range(0,21,1):
                x__ = [df['{}'.format(2*i)][j],df['{}'.format(2*i+1)][j]]
                x_.append(x__)
            x.append(x_)
        y = df['42'][:self.data_num]
        #y = onehot_convert(y)

        self.data = torch.from_numpy(np.array(x)).float()
        print(self.data)
        self.label = torch.from_numpy(np.array(y)).long()
        print(self.label)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

class LitHands(pl.LightningModule):

    def __init__(self, hidden_size=10, learning_rate=2e-4):
        super().__init__()
        # Set our init args as class attributes
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Hardcode some dataset specific attributes
        self.num_classes = 3
        self.dims = (1, 21, 2)
        channels, width, height = self.dims
        """
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])        
        """
        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, 2*hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(),
            
            #nn.Dropout(0.1),
            nn.Linear(2*hidden_size, self.num_classes)
        )
            
    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        #print(logits,y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    """
    def prepare_data(self):
        data_num=292   
        self.dataset = HandsDataset(data_num, transform=None)
        self.dataset2 = HandsDataset(data_num, transform=None)
    """
    def setup(self, stage=None):
        data_num=1350 #292   
        self.dataset = HandsDataset(data_num, transform=None)
        n_train = int(len(self.dataset)*0.5)
        n_val = int(len(self.dataset)*0.3)
        n_test = len(self.dataset)-n_train-n_val
        print("n_train, n_val, n_test ",n_train, n_val, n_test)

        self.train_data, self.val_data, self.test_data = random_split(self.dataset,[n_train, n_val, n_test])
        print('type(train_data)',type(self.train_data))
        #self.test_data = self.dataset2

    def train_dataloader(self):
        self.trainloader = DataLoader(self.train_data, shuffle=True, drop_last = True, batch_size=32, num_workers=0)
        return self.trainloader

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, batch_size=32, num_workers=0)

    
    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=32)
     
def main():
    model = LitHands()
    print(model)
    trainer = pl.Trainer(max_epochs=2000)
    #trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(model) #, DataLoader(train, batch_size = 32, shuffle= True), DataLoader(val, batch_size = 32))
    trainer.test(model)
    print('training_finished')
    PATH = "hands_mlp.ckpt"
    trainer.save_checkpoint(PATH)

    pretrained_model = model.load_from_checkpoint(PATH)
    pretrained_model.freeze()
    pretrained_model.eval()

    a =  torch.tensor([[315., 420.],
            [409., 401.],
            [485., 349.],
            [534., 302.],
            [574., 279.],
            [418., 205.],
            [442., 126.],
            [462.,  74.],
            [477.,  33.],
            [364., 186.],
            [370.,  89.],
            [379.,  22.],
            [386., -33.],
            [312., 192.],
            [311.,  98.],
            [316.,  37.],
            [321.,  -9.],
            [259., 218.],
            [230., 154.],
            [215., 113.],
            [204.,  77.]])
    print(a[:])
    results = pretrained_model(a[:].reshape(1,21,2))
    print(results)
    preds = torch.argmax(results)
    print(preds)

    df = pd.read_csv('./hands/sample_hands7.csv', sep=',')
    print(df.head(3)) #データの確認
    df = df.astype(int)
    data_num = len(df)
    x = []
    for j in range(data_num):
        x_ = []
        for i in range(0,21,1):
            x__ = [df['{}'.format(2*i)][j],df['{}'.format(2*i+1)][j]]
            x_.append(x__)
        x.append(x_)
    data_ = torch.from_numpy(np.array(x)).float()
    y = df['42'][:data_num]
    label_ = torch.from_numpy(np.array(y)).long()
    count = 0
    for j in range(data_num):
        a = data_[j]
        results =  pretrained_model(a[:].reshape(1,21,2))
        #print(results)
        preds = torch.argmax(results)
        print(j,preds,label_[j])
        if preds== label_[j]:
            count += 1
    acc=count/data_num
    print("acc = ",acc)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))    
