import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot

class ImageTransformer:
    def __init__(self, lenth, image_size):
        self.image_size = image_size
        self.gasf = GramianAngularField(image_size=self.image_size, method='summation')
        self.gadf = GramianAngularField(image_size=self.image_size, method='difference')
        self.mtf = MarkovTransitionField(image_size=self.image_size)
        self.rp = RecurrencePlot(threshold='point',
                                 percentage=20,
                                 dimension=lenth - image_size + 1,
                                 time_delay=1,
                                 #color_method='distance'
                                 )

        # TODO:形状不一致24*24

    def fit_transform(self, X):
        gasf_imgs = []
        gadf_imgs = []
        mtf_imgs = []
        rp_imgs = []

        for i in range(X.shape[0]):
            gasf_img = self.gasf.fit_transform(X[i].reshape(1, -1))[0]
            gadf_img = self.gadf.fit_transform(X[i].reshape(1, -1))[0]
            mtf_img = self.mtf.fit_transform(X[i].reshape(1, -1))[0]
            rp_img = self.rp.fit_transform(X[i].reshape(1, -1))[0]

            #TODO

            gasf_imgs.append(gasf_img)
            gadf_imgs.append(gadf_img)
            mtf_imgs.append(mtf_img)
            rp_imgs.append(rp_img)

        return np.array(gasf_imgs), np.array(gadf_imgs), np.array(mtf_imgs), np.array(rp_imgs)
