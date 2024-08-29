import torch
from torch import nn

from models import cnnlstm
from models.vision_transformer.vivit import ViViT
from models.i3d.pytorch_i3d import InceptionI3d

def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm', "vivit", '3dcnn'
	]

	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	elif opt.model == 'vivit':
		model = ViViT(224, 16, opt.n_classes, 16)
	elif opt.model == '3dcnn':
		model = InceptionI3d(opt.n_classes, in_channels=3)
		# model.load_state_dict(torch.load('models/rgb_imagenet.pt'))

	return model.to(device)