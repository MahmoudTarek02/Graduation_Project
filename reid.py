import os
from pathlib import Path

_CACHE_ROOT = Path("/tmp/codex_retail_tracking_cache")
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))

from PIL import Image
import torch
import torchreid
from torchreid import metrics
from torchreid.data.transforms import build_transforms

from config import (
    REID_DISTANCE_METRIC,
    REID_IMAGE_HEIGHT,
    REID_IMAGE_WIDTH,
    REID_MODEL_NAME,
    REID_WEIGHTS_PATH,
)


class REID:
    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.model = torchreid.models.build_model(
            name=REID_MODEL_NAME,
            num_classes=1,  # human
            loss="softmax",
            pretrained=False,
            use_gpu=self.use_gpu
        )
        torchreid.utils.load_pretrained_weights(self.model, REID_WEIGHTS_PATH)
        if self.use_gpu:
            self.model = self.model.cuda()
        _, self.transform_te = build_transforms(
            height=REID_IMAGE_HEIGHT, width=REID_IMAGE_WIDTH,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )
        self.dist_metric = REID_DISTANCE_METRIC
        self.model.eval()

        if True:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _features(self, imgs):
        f = []
        for img in imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            if self.use_gpu:
                img = img.cuda()
            features = self._extract_features(img)
            features = features.data.cpu()  # tensor shape=1x2048
            f.append(features)
        f = torch.cat(f, 0)
        return f
    
    def _feature(self, img): # Process single image to work on real time
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = self.transform_te(img)
        img = torch.unsqueeze(img, 0)
        if self.use_gpu:
            img = img.cuda()
        features = self._extract_features(img)
        features = features.data.cpu()  # tensor shape=1x2048
        
        return features

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()


if __name__ == '__main__':
    reid = REID()
