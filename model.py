import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DownConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels , padding ,
    ):
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels
                      ,out_channels=out_channels
                      ,kernel_size=kernel
                      ,padding=padding),

            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # properties of class

    def forward(self, x):
        return self.down_conv(x)
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        # None


class UpConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels , padding ,
    ):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels
                      , out_channels=out_channels
                      , kernel_size=kernel
                      , padding=padding),

            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # properties of class

    def forward(self, x):
        return self.up_conv(x)
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        # None


class Bottleneck(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, padding,
    ):
        super(Bottleneck, self).__init__()
        self.bottle = nn.Sequential(
            nn.Conv2d(in_channels=in_channels
                      , out_channels=out_channels
                      , kernel_size=kernel
                      , padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # properties of class

    def forward(self, x):
        return self.bottle(x)
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        # None


class BaseModel(nn.Module):
    def __init__(
            self, kernel, num_filters, num_colors, in_channels=1, padding=1
    ):
        super(BaseModel, self).__init__()

        self.downs      = nn.ModuleList([DownConv(kernel,in_channels,num_filters,padding),
                                    DownConv(kernel,num_filters,2*num_filters,padding),])

        self.bottleneck = Bottleneck(kernel,2*num_filters,2*num_filters,padding)

        self.ups        = nn.ModuleList([UpConv(kernel, 2 * num_filters, num_filters, padding),
                                 UpConv(kernel, num_filters, num_colors, padding),])

        self.conv2d     = nn.Conv2d(in_channels=num_colors
                      , out_channels=num_colors
                      , kernel_size=kernel
                      , padding=padding)
        # Other properties if needed

        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################

    def forward(self, x):
        out = self.downs[0](x)
        out = self.downs[1](out)
        out = self.bottleneck(out)
        out = self.ups[0](out)
        out = self.ups[1](out)
        out = self.conv2d(out)
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        return out

##############################################################################

class CustomUNET(nn.Module):
    def __init__(
            self, kernel, num_filters, num_colors, in_channels=1, padding=1
    ):
        super(CustomUNET, self).__init__()

        self.downs = nn.ModuleList([DownConv(kernel, in_channels, num_filters, padding),
                                    DownConv(kernel, num_filters, 2 * num_filters, padding), ])

        self.bottleneck = Bottleneck(kernel, 2 * num_filters, 2 * num_filters, padding)

        self.ups = nn.ModuleList([UpConv(kernel, 2*2 * num_filters, num_filters, padding),
                                  UpConv(kernel, 2*num_filters, num_colors, padding), ])

        self.conv2d = nn.Conv2d(in_channels=num_colors+1
                                , out_channels=num_colors
                                , kernel_size=kernel
                                , padding=padding)
        # Other properties if needed

        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################

    def forward(self, x):
        firstOut=self.downs[0](x)
        secondOut=self.downs[1](firstOut)
        thirdOut = self.bottleneck(secondOut)
        fourthOut=self.ups[0](torch.cat([secondOut,thirdOut],dim=1))
        fifthOut=self.ups[1](torch.cat([firstOut,fourthOut],dim=1))
        return self.conv2d(torch.cat([x,fifthOut],dim=1))

##############################################################################

class DownConvResidual(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels , padding ,
    ):

        super(DownConvResidual, self).__init__()
        self.r   = nn.ReLU()
        self.b1  = nn.BatchNorm2d(in_channels)
        self.c1  = nn.Conv2d(in_channels=in_channels
                               , out_channels=out_channels
                               , kernel_size=kernel
                               , padding=padding)
        self.b2  = nn.BatchNorm2d(out_channels)
        self.c2  = nn.Conv2d(in_channels=out_channels
                               , out_channels=out_channels
                               , kernel_size=kernel
                               , padding=padding
                               , stride=1)
        self.s   = nn.Conv2d(in_channels=in_channels
                               , out_channels=out_channels
                               , kernel_size=1
                               , padding=0)
        self.max = nn.MaxPool2d(kernel_size=2)
        self.b3  = nn.BatchNorm2d(out_channels)
    def forward(self, inputs):
        x=self.b1(inputs)
        x=self.r(x)
        x=self.c1(x)
        x=self.b2(x)
        x=self.r(x)
        x=self.c2(x)
        s=self.s(inputs)
        skip = x+s
        skip = self.max(skip)
        skip = self.b3(skip)
        skip = self.r(skip)
        return skip

    ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        # None


class UpConvResidual(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels , padding ,
    ):
        super(UpConvResidual, self).__init__()
        self.r = nn.ReLU()

        self.b1 = nn.BatchNorm2d(in_channels)

        self.c1 = nn.Conv2d(in_channels=in_channels
                            , out_channels=out_channels
                            , kernel_size=kernel
                            , padding=padding)

        self.b2 = nn.BatchNorm2d(out_channels)

        self.c2 = nn.Conv2d(in_channels=out_channels
                            , out_channels=out_channels
                            , kernel_size=kernel
                            , padding=padding
                            , stride=1)

        self.s = nn.Conv2d(in_channels=in_channels
                           , out_channels=out_channels
                           , kernel_size=1
                           , padding=0)

        self.ups = nn.Upsample(scale_factor=2)
        self.b3  = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.r(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.r(x)
        x = self.c2(x)
        s = self.s(inputs)
        skip = x + s
        skip = self.ups(skip)
        skip = self.b3(skip)
        skip = self.r(skip)
        return skip
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################


class BottleneckResidual(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels, padding,
    ):
        super(BottleneckResidual, self).__init__()
        self.r = nn.ReLU()

        self.b1 = nn.BatchNorm2d(in_channels)

        self.c1 = nn.Conv2d(in_channels=in_channels
                            , out_channels=out_channels
                            , kernel_size=kernel
                            , padding=padding)

        self.b2 = nn.BatchNorm2d(out_channels)

        self.c2 = nn.Conv2d(in_channels=out_channels
                            , out_channels=out_channels
                            , kernel_size=kernel
                            , padding=padding
                            , stride=1)

        self.s = nn.Conv2d(in_channels=in_channels
                           , out_channels=out_channels
                           , kernel_size=1
                           , padding=0)

        self.b3  = nn.BatchNorm2d(out_channels)
    def forward(self, inputs):
        x = self.b1(inputs)
        r = self.r(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.r(x)
        x = self.c2(x)
        s = self.s(inputs)
        skip = x + s
        skip = self.b3(skip)
        skip = self.r(skip)
        return skip
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        # None


class CustomUNETResidual(nn.Module):
    def __init__(
            self, kernel, num_filters, num_colors, in_channels=1, padding=1
    ):
        super(CustomUNETResidual, self).__init__()

        self.downs = nn.ModuleList([DownConvResidual(kernel, in_channels, num_filters, padding),
                                    DownConvResidual(kernel, num_filters, 2 * num_filters, padding), ])

        self.bottleneck = BottleneckResidual(kernel, 2 * num_filters, 2 * num_filters, padding)

        self.ups = nn.ModuleList([UpConvResidual(kernel, 2 * 2 * num_filters, num_filters, padding),
                                  UpConvResidual(kernel, 2 * num_filters, num_colors, padding), ])

        self.conv2d = nn.Conv2d(in_channels=num_colors + 1
                                , out_channels=num_colors
                                , kernel_size=kernel
                                , padding=padding)
        # Other properties if needed

        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################

    def forward(self, x):
        firstOut = self.downs[0](x)
        secondOut = self.downs[1](firstOut)
        thirdOut = self.bottleneck(secondOut)
        fourthOut = self.ups[0](torch.cat([secondOut, thirdOut], dim=1))
        fifthOut = self.ups[1](torch.cat([firstOut, fourthOut], dim=1))
        return self.conv2d(torch.cat([x, fifthOut], dim=1))


##############################################################################


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, num_filters, num_colors, in_channels=1,
    ):
        super(UNET, self).__init__()

        self.ups   = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Other properties if needed
        self.pool  = nn.MaxPool2d(kernel_size=2 , stride =2)
        # Down part of the model, bottleneck, Up part of the model, final conv
        # in_channel=in_channels
        # out_channel=num_filters
        features=[]
        for i in range(4):
            features.append(num_filters*(2**i))
        in_channel=in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channel=feature
            # in_channel=out_channel
            # out_channel=out_channel*2
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels=num_colors, kernel_size=3)
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################

    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        skip_connections = []

        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x= self.bottleneck(x)

        skip_connections=skip_connections[::-1]

        for index in range(0,len(self.ups),2):
            x = self.ups[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape)

            concat_skip=torch.cat((skip_connection,x),dim=1)
            x= self.ups[index+1](concat_skip)

        return self.final_conv(x)