from typing import Dict, Any, Callable, Tuple

import torch
import torch.nn as nn
from models.Transformer import TransformerModel
from models.PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
from models.IntmdSequential import IntermediateSequential
import  math


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity_data = x
        output = self.prelu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output, identity_data)
        output = self.prelu(output)
        return output


class backboneRES(nn.Module):
    def __init__(self):
        super(backboneRES, self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.PReLU()
        )
        self.block2=ResidualBlock(64)
        self.block3=ResidualBlock(64)
        self.block4=ResidualBlock(64)
        self.block5=ResidualBlock(64)
        self.block6=ResidualBlock(64)

        self.block7=nn.Sequential(
            nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward (self,x):
        b1=self.block1(x)
        b2=self.block2(b1)
        b3=self.block3(b2)
        b4=self.block4(b3)
        b5=self.block5(b4)
        b6=self.block6(b5)
        b7=self.block7(b6)
        return b7


class backboneSimple_CNN(nn.Module):
    def __init__(self):
        super(backboneSimple_CNN, self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1,padding=1),
            nn.PReLU()
        )
        self.block2=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
            )
        self.block3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
        )
        self.block4=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
        )
        self.block5=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
            nn.PReLU()
        )
        self.block6=nn.Sequential(
            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward (self,x):
        b1=self.block1(x)
        b2=self.block2(b1)
        b3=self.block3(b2)
        b4=self.block4(b3)
        b5=self.block5(b4)
        b6=self.block6(b5)
        return b6




def _get_padding(padding_type,kernel_size):
    assert padding_type in ['SAME','VALID']
    if padding_type == 'SAME':
        _list = [(k-1)//2 for k in kernel_size]
        return tuple(_list)
    return tuple(0 for _ in kernel_size)


def _reshape_output(x, img_dim, patch_dim, embedding_dim):
    x = x.view(
        x.size(0),
        int(img_dim / patch_dim),
        int(img_dim / patch_dim),
        embedding_dim
    )
    x = x.permute(0, 3, 1, 2).contiguous()
    return x



class Transfusion_encode(nn.Module):
    def __init__(self,
                 img_dim,
                 patch_dim,
                 num_channels,
                 embedding_dim,
                 num_heads,
                 num_layers,
                 hidden_dim,
                 dropout_rate,
                 attn_droupt_rate,
                 conv_patch_repersentation,
                 positional_encoding_type):
        super(Transfusion_encode,self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim ==0

        self.img_dim = img_dim
        self.backbone = backboneRES()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_droupt_rate
        self.conv_patch_repersentation = conv_patch_repersentation
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        #self._get_padding = _get_padding()

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels #3x3x2

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type =="fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim
            )

        self.linear_encoding = nn.Linear(self.flatten_dim,self.embedding_dim)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate
        )

        self.pre_head_ln = nn.LayerNorm(self.embedding_dim)

        if self.conv_patch_repersentation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size= (self.patch_dim,self.patch_dim),
                stride=(self.patch_dim,self.patch_dim),
                padding=_get_padding('VALID',(self.patch_dim,self.patch_dim))
            )
        else:
            self.conv_x = None

    def forward(self,x):
        n, c, h, w = x.shape  #(1,2,60,60)
        if self.conv_patch_repersentation:
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim).unfold(3, self.patch_dim, self.patch_dim).contiguous() #(1,2,20,20,2,2)
            )
            x = x.view(n, c, -1, self.patch_dim**2)  #(1,2,)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        #apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x, intmd_x


class Transfusion_decoder(nn.Module):
    def __init__(self,
                 img_dim,
                 patch_dim,
                 num_channels,
                 embedding_dim,
                 num_heads,
                 num_layers,
                 hidden_dim,
                 dropout_rate,
                 attn_droupt_rate,
                 conv_patch_repersentation,
                 positional_encoding_type
                 ):
        super(Transfusion_decoder,self).__init__()
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_droupt_rate
        self.conv_patch_repersentation = conv_patch_repersentation
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=_get_padding('VALID',(1,1))
        )
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels = self.embedding_dim,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=_get_padding('VALID',(1,1))
        )
        self.upsample = nn.Upsample(scale_factor=self.patch_dim, mode='bilinear', align_corners=True)

    def forward(self,x):
        x = _reshape_output(x, self.img_dim, self.patch_dim, self.embedding_dim)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class Transfusion_conv1_1(nn.Module):
    def __init__(self):
        super(Transfusion_conv1_1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n =m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Transfusion_conv1_2(nn.Module):
    def __init__(self):
        super(Transfusion_conv1_2,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n =m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x



class Transfusion_conv1(nn.Module):
    def __init__(self):
        super(Transfusion_conv1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n =m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Transfusion_conv2(nn.Module):
    def __init__(self):
        super(Transfusion_conv2,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n =m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Transfusion_conv3(nn.Module):
    def __init__(self):
        super(Transfusion_conv3,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=5,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n =m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x



class Transfusion_conv4(nn.Module):
    def __init__(self):
        super(Transfusion_conv4,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.PReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n =m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x




class Transfusion15(nn.Module):
    def __init__(self,
                 img_dim=60,#60
                 patch_dim=3,#3
                 num_channels=2,
                 num_channels_2=4,
                 num_channels_3=5,
                 num_channels_4=6,
                 embedding_dim=512,
                 num_heads=4,
                 num_layers=6,
                 hidden_dim=4,
                 dropout_rate=0.0,
                 attn_droupt_rate=0.0,
                 conv_patch_repersentation=False,
                 positional_encoding_type="learned"
                 ):
        super(Transfusion15,self).__init__()

        self.img_dim = img_dim
        self.backbone = backboneRES()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.num_channels_2 = num_channels_2
        self.num_channels_3 = num_channels_3
        self.num_channels_4 = num_channels_4
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_droupt_rate
        self.conv_patch_repersentation = conv_patch_repersentation
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.positional_encoding_type = positional_encoding_type

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length,self.embedding_dim,self.seq_length
            )
        elif positional_encoding_type =="fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim
            )

        self.encoder1 = Transfusion_encode(self.img_dim,self.patch_dim,self.num_channels,self.embedding_dim,
                                          self.num_heads,self.num_layers,self.hidden_dim,self.dropout_rate,
                                          self.attn_dropout_rate,self.conv_patch_repersentation,self.positional_encoding_type)

        self.encoder2 = Transfusion_encode(self.img_dim,self.patch_dim,self.num_channels_2,self.embedding_dim,
                                          self.num_heads,self.num_layers,self.hidden_dim,self.dropout_rate,
                                          self.attn_dropout_rate,self.conv_patch_repersentation,self.positional_encoding_type)

        self.encoder3 = Transfusion_encode(self.img_dim, self.patch_dim, self.num_channels_3, self.embedding_dim,
                                           self.num_heads, self.num_layers, self.hidden_dim, self.dropout_rate,
                                           self.attn_dropout_rate, self.conv_patch_repersentation,
                                           self.positional_encoding_type)

        self.encoder4 = Transfusion_encode(self.img_dim, self.patch_dim, self.num_channels_4, self.embedding_dim,
                                           self.num_heads, self.num_layers, self.hidden_dim, self.dropout_rate,
                                           self.attn_dropout_rate, self.conv_patch_repersentation,
                                           self.positional_encoding_type)

        self.decoder = Transfusion_decoder(self.img_dim,self.patch_dim,self.num_channels,self.embedding_dim,
                                          self.num_heads,self.num_layers,self.hidden_dim,self.dropout_rate,
                                          self.attn_dropout_rate,self.conv_patch_repersentation,self.positional_encoding_type)

        self.conv_module1 = Transfusion_conv1()
        self.conv_module2 = Transfusion_conv2()
        self.conv_module3 = Transfusion_conv3()
        self.conv_module4 = Transfusion_conv4()

    def forward(self,x):
        source = x
        #x = self.backbone(x)
        x_Te1, intmd_x = self.encoder1(x)
        x_Td1 = self.decoder(x_Te1)
        x1 = self.conv_module1(torch.cat((x_Td1,source),1))
        x_Te2, intmd_x2 = self.encoder2(torch.cat((x1,source,x_Td1),1))
        x_Td2 = self.decoder(x_Te2)
        x2 = self.conv_module2(torch.cat((x_Td2,x1,source),1))
        x_Te3, intmd_x3 = self.encoder3(torch.cat((x2,source,x_Td1,x_Td2),1))
        x_Td3 = self.decoder(x_Te3)
        x3 = self.conv_module3(torch.cat((x_Td3,x2,x1,source),1))
        x_Te4, intmd_x4 = self.encoder4(torch.cat((x3,source,x_Td1,x_Td2,x_Td3),1))
        x_Td4 = self.decoder(x_Te4)
        x4 =self.conv_module4(torch.cat((x_Td4,x3,x2,x1,source),1))
        return x1, x2, x3, x4


