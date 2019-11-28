import numpy as np
from errors import *

print('===============METRIC===============')

#
preds = np.load('./npy/mae_compare/predict_in_0_taxibj_GeoMan_day0405.npy')
truth = np.load('./npy/mae_compare/truth_in_0_taxibj_GeoMan_day0405.npy')
print(Mae(truth, preds))
print(Mape(truth, preds))
print(Made(truth, preds))
