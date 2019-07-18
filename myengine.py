from torchreid.data.transforms import build_transforms
import cv2
from PIL import Image
import torchreid
import torch
import os
from torchreid import metrics

class REID:
    def __init__(self):
        self.model = torchreid.models.build_model(
                name='resnet50',
                num_classes=1,#unusable
                loss='softmax',
                pretrained=True,
                use_gpu = True
            )
        torchreid.utils.load_pretrained_weights(self.model, 'model_data/model.pth.tar-60')
        self.model = self.model.cuda()
        self.optimizer = torchreid.optim.build_optimizer(
                self.model,
                optim='adam',
                lr=0.0003
            )
        self.scheduler = torchreid.optim.build_lr_scheduler(
                self.optimizer,
                lr_scheduler='single_step',
                stepsize=20
            )
        _, self.transform_te = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )
        self.dist_metric = 'euclidean'
        self.model.eval()

    @torch.no_grad()
    def get_features(self, query_imgs, gallery_imgs):
        qf = []
        for img in query_imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            features = self._extract_features(img)
            features = features.data.cpu() #tensor shape=1x2048
            qf.append(features)
        qf = torch.cat(qf, 0)

        gf = []
        for img in gallery_imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            features = self._extract_features(img)
            features = features.data.cpu() #tensor shape=1x2048
            gf.append(features)
        gf = torch.cat(gf, 0)
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()
    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

if __name__ == '__main__':
    query_imgs = []
    query_imgs.append(cv2.imread('1.jpg'))
    query_imgs.append(cv2.imread('timg.jpg'))
    gallery_imgs = []
    gallery_imgs.append(cv2.imread('timg.jpg'))
    gallery_imgs.append(cv2.imread('3.jpg'))

    reid = REID()
    reid.get_features(query_imgs, gallery_imgs)


