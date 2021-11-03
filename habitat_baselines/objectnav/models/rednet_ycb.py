import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# from torch.utils.checkpoint import checkpoint

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class RedNet(nn.Module):
    def __init__(self, cfg):

        super(RedNet, self).__init__()

        num_classes = cfg["n_classes"]
        pretrained = cfg["resnet_pretrained"]

        block = Bottleneck
        transblock = TransBasicBlock
        layers = [3, 4, 6, 3]
        # original resnet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # resnet for depth channel
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64 * 4, 64)
        self.agant2 = self._make_agant_layer(128 * 4, 128)
        self.agant3 = self._make_agant_layer(256 * 4, 256)
        self.agant4 = self._make_agant_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv_custom = nn.ConvTranspose2d(
            self.inplanes,
            num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

        self.out5_conv_custom = nn.Conv2d(
            256, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out4_conv_custom = nn.Conv2d(
            128, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out3_conv_custom = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, bias=True
        )
        self.out2_conv_custom = nn.Conv2d(
            64, num_classes, kernel_size=1, stride=1, bias=True
        )

        if pretrained:
            self._load_resnet_pretrained()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes,
                    kernel_size=2,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        return layers

    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(utils.model_urls["resnet50"])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith("conv1"):  # the first conv_op
                    model_dict[k] = v
                    model_dict[k.replace("conv1", "conv1_d")] = torch.mean(
                        v, 1
                    ).data.view_as(state_dict[k.replace("conv1", "conv1_d")])

                elif k.startswith("bn1"):
                    model_dict[k] = v
                    model_dict[k.replace("bn1", "bn1_d")] = v
                elif k.startswith("layer"):
                    model_dict[k] = v
                    model_dict[k[:6] + "_d" + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward_downsample(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        fuse0 = x + depth

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        depth = self.layer1_d(depth)
        fuse1 = x + depth
        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        fuse2 = x + depth
        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        fuse3 = x + depth
        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        fuse4 = x + depth

        return fuse0, fuse1, fuse2, fuse3, fuse4

    def forward_upsample(self, fuse0, fuse1, fuse2, fuse3, fuse4):

        agant4 = self.agant4(fuse4)
        # upsample 1
        x = self.deconv1(agant4)
        # if self.training:
        #     out5 = self.out5_conv_custom(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        # if self.training:
        #     out4 = self.out4_conv_custom(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        # if self.training:
        #     out3 = self.out3_conv_custom(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        # if self.training:
        #     out2 = self.out2_conv_custom(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)
        out = self.final_deconv_custom(x)

        # if self.training:
        #     return out, out2, out3, out4, out5

        return out

    def forward(self, rgb, depth):
        # phase_checkpoint=False

        # if phase_checkpoint:
        #     depth.requires_grad_()
        #     fuses = checkpoint(self.forward_downsample, rgb, depth)
        #     out = checkpoint(self.forward_upsample, *fuses)
        # else:
        fuses = self.forward_downsample(rgb, depth)
        out = self.forward_upsample(*fuses)
        return out

    @classmethod
    def load_pretrained_model(
        cls,
        model_path="/checkpoint/maksymets/checkpoints/object_nav/perception/rednet_mp3d_best_model.pkl",
        model_config=None,
        device=torch.device("cuda"),
        num_classes=40,
    ):
        if model_config is None:
            model_config = {
                "arch": "rednet",
                "resnet_pretrained": False,
                "finetune": True,
                "SUNRGBD_pretrained_weights": "custom_rednet_ckpt_40.pth",
                "n_classes": num_classes,
                "upsample_prediction": True,
            }

        # Create model
        model = cls(model_config)
        model.cuda()
        print(f"torch.__version__={torch.__version__}")
        print(f"torch.version.cuda={torch.version.cuda}")
        print(f"torch.backends.cudnn.version={torch.backends.cudnn.version()}")

        # if device.type == 'cuda':
        #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        # print("Loading pre-trained weights: ", model_path)
        state = torch.load(model_path, map_location="cpu")
        model_state = state["model_state"]
        model_state = convert_weights_cuda_cpu(
            model_state, "cpu"
        )  # device.type
        model.load_state_dict(model_state)
        del model_state
        model.cuda()
        print(f"Rendet model loaded from: {model_path}")
        print(f"Rendet cuda device: {next(model.parameters()).device}")
        print(
            "Rendet number of trainable parameters: {}".format(
                sum(
                    param.numel()
                    for param in model.parameters()
                    if param.requires_grad
                )
            )
        )
        return model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1,
                bias=False,
            )
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


"""
Compute ReNet encoder features
"""


def convert_weights_cuda_cpu(weights, device):
    names = list(weights.keys())
    is_module = names[0].split(".")[0] == "module"
    if device == "cuda" and not is_module:
        new_weights = {"module." + k: v for k, v in weights.items()}
    elif device == "cpu" and is_module:
        new_weights = {
            ".".join(k.split(".")[1:]): v for k, v in weights.items()
        }
    else:
        new_weights = weights
    return new_weights


def upsample_prediction(input, size):
    # size = (ht, wt)
    return F.interpolate(input, size=size, mode="bilinear", align_corners=True)


class RedNet_Encoder(nn.Module):
    """
    Encoder extracted from the RedNet model to
    compute Memory tensors.
    Not meant for training.
    """

    def __init__(self, cfg, device):
        super(RedNet_Encoder, self).__init__()

        self.device = device

        self.upsample_prediction = cfg["upsample_prediction"]
        self.size = (cfg["img_rows"], cfg["img_cols"])

        self.rednet = RedNet(cfg)

        ckpt = torch.load(cfg["model_path"])
        model_state = convert_weights_cuda_cpu(ckpt["model_state"], "cpu")
        self.rednet.load_state_dict(model_state)

        print("loaded state checkpoint from: ", cfg["model_path"])

    def forward(self, x, depth):

        embedding = torch.zeros((1, 512, 15, 20))

        def copy_data(m, i, o):
            embedding.copy_(o.data)

        h = self.rednet.agant4.register_forward_hook(copy_data)

        self.rednet(x, depth)
        h.remove()

        if self.upsample_prediction:
            embedding = upsample_prediction(embedding, self.size)

        embedding = embedding.to(self.device)

        return embedding


import numpy as np


class ConfusionMatrix:
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert (
            predicted.shape[0] == target.shape[0]
        ), "number of targets and predicted outputs do not match"

        if np.ndim(predicted) != 1:
            assert (
                predicted.shape[1] == self.num_classes
            ), "number of predictions does not match size of confusion matrix"
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (
                predicted.min() >= 0
            ), "predicted values are not between 0 and k-1"

        if np.ndim(target) != 1:
            assert (
                target.shape[1] == self.num_classes
            ), "Onehot target does not match size of confusion matrix"
            assert (target >= 0).all() and (
                target <= 1
            ).all(), "in one-hot encoding, target values should be 0 or 1"
            assert (
                target.sum(1) == 1
            ).all(), "multi-label setting is not supported"
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (
                target.min() >= 0
            ), "target values are not between 0 and k-1"

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes ** 2
        )
        assert bincount_2d.size == self.num_classes ** 2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class IoU:
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.
        """
        # Dimensions check
        # assert predicted.size(0) == target.size(0), \
        #    'number of targets and predicted outputs do not match'
        # assert predicted.dim() == 3 or predicted.dim() == 4, \
        #    "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        # assert target.dim() == 3 or target.dim() == 4, \
        #    "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        # if predicted.dim() == 4:
        #    _, predicted = predicted.max(1)
        # if target.dim() == 4:
        #    _, target = target.max(1)

        # self.conf_metric.add(predicted.view(-1), target.view(-1))
        # -- data is already flatten
        # -- preprocessed in the train() pipeline
        self.conf_metric.add(predicted, target)

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / (
                true_positive + false_positive + false_negative
            )

        acc = np.sum(true_positive) / np.sum(conf_matrix)
        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            acc_cls = true_positive / np.sum(conf_matrix, 1)
        mean_acc_cls = np.nanmean(acc_cls)

        return iou, np.nanmean(iou), acc, mean_acc_cls