"""ImageNet 사전학습 CNN 특징.

    net = resnet()                              # 한 번 만들어 재사용
    f = cnn_features(net, images)               # (n, 32, 32, 384)

PaDiM, PatchCore 가 쓰는 특징임. 역전파 학습이 없어서 CPU 로도 돌아감.

가중치는 처음 한 번 내려받아 `artifacts/` 에 저장하고 그 뒤로는 거기서 읽음.
"""

import numpy as np

from ..paths import ARTIFACTS

# ImageNet 정규화 값. 사전학습 때와 같은 분포로 넣어야 특징이 의미가 있음.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WEIGHTS = ARTIFACTS / "resnet18.pt"


def resnet(path=WEIGHTS):
    """ImageNet 사전학습 ResNet18. 평가 모드로 돌려줌.

    `path` 에 가중치가 없으면 torchvision 에서 내려받아 저장함.
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18()
    if path.exists():
        net.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), path)
    return net.eval()


def cnn_features(net, images, layers=("layer2", "layer3"), batch=32):
    """(n, 256, 256, 3) uint8 -> (n, 32, 32, d).

    layer2 는 256 -> 32 (stride 8), layer3 는 256 -> 16 (stride 16) 임.
    layer3 를 32 로 키워 layer2 와 이어붙임. 얕은 층은 질감, 깊은 층은 의미를 담음.
    ResNet18 기준 d = 128 + 256 = 384.
    """
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = (np.asarray(images[i:i + batch], np.float32) / 255.0 - MEAN) / STD
            x = torch.from_numpy(x).permute(0, 3, 1, 2)

            feats = {}
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            for name in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(net, name)(x)
                if name in layers:
                    feats[name] = x
                if name == layers[-1]:
                    break

            size = feats[layers[0]].shape[-2:]
            f = torch.cat([F.interpolate(feats[k], size=size, mode="bilinear",
                                         align_corners=False) for k in layers], dim=1)
            out.append(f.permute(0, 2, 3, 1).numpy())
    return np.concatenate(out)
