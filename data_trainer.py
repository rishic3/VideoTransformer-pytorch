import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from dataset import Sinus
import data_transform as T

class Collator(object):

	def __init__(self, objective):
		self.objective = objective
	
	def collate(self, minibatch):
		image_list = []
		label_list = []
		mask_list = []
		marker_list = []
		for record in minibatch:
			image_list.append(record[0])
			label_list.append(record[1])
			if self.objective == 'mim':
				mask_list.append(record[2])
				marker_list.append(record[3])
		minibatch = []
		minibatch.append(torch.stack(image_list))
		if self.objective == 'mim':
			minibatch.append(torch.stack(label_list))
			minibatch.append(torch.stack(mask_list))
			minibatch.append(marker_list)
		else:
			label = np.stack(label_list)
			minibatch.append(torch.from_numpy(label))
		
		return minibatch

class SinusDataModule(pl.LightningDataModule):
	def __init__(self, configs, data_directory):
		super().__init__()
		self.data_directory = data_directory
		self.configs = configs
	
	def get_dataset(self, csv_filename, transform, temporal_sample):
		csv_path = os.path.join(self.data_directory, csv_filename)
		dataset = Sinus(
			self.configs,
			csv_path,
			transform=transform,
			temporal_sample=temporal_sample)
		return dataset

	def setup(self, stage):
		if self.configs.objective == 'mim':
			scale = (0.5, 1.0)
			color_jitter = None
		else:
			color_jitter = 0.4
			scale = None
		
		if self.configs.data_statics == 'imagenet':
			mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
		elif self.configs.data_statics == 'sinus':
			mean, std = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225)
		else:
			mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
		
		train_transform = T.create_video_transform(
			objective=self.configs.objective,
			input_size=self.configs.img_size,
			is_training=True,
			scale=scale,
			hflip=0.5,
			color_jitter=color_jitter,
			auto_augment=self.configs.auto_augment,
			interpolation='bicubic',
			mean=mean,
			std=std)
		train_temporal_sample = T.TemporalRandomCrop(
			self.configs.num_frames * self.configs.frame_interval)

		self.train_dataset = self.get_dataset('train.csv', train_transform, train_temporal_sample)
	
		val_transform = T.create_video_transform(
			input_size=self.configs.img_size,
			is_training=False,
			interpolation='bicubic',
			mean=mean,
			std=std)
		val_temporal_sample = T.TemporalRandomCrop(
			self.configs.num_frames * self.configs.frame_interval)
		self.val_dataset = self.get_dataset('valid.csv', val_transform, val_temporal_sample)

		test_transform = T.Compose([
			T.Resize(scale_range=(-1, 256)),
			T.ThreeCrop(size=self.configs.img_size),
			T.ToTensor(),
			T.Normalize(mean, std),
			])
		test_temporal_sample = T.TemporalRandomCrop(
			self.configs.num_frames * self.configs.frame_interval)
		self.test_dataset = self.get_dataset('test.csv', test_transform, test_temporal_sample)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.configs.batch_size,
			num_workers=self.configs.num_workers,
			collate_fn=Collator(self.configs.objective).collate,
			shuffle=True,
			drop_last=True, 
			pin_memory=True
		)
	
	def val_dataloader(self):
		if hasattr(self, 'val_dataset'):
			return DataLoader(
				self.val_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				collate_fn=Collator(self.configs.objective).collate,
				shuffle=False,
				drop_last=False,
			)

	def test_dataloader(self):
		if hasattr(self, 'test_dataset'):
			return DataLoader(
				self.test_dataset,
				batch_size=self.configs.batch_size,
				num_workers=self.configs.num_workers,
				collate_fn=Collator(self.configs.objective).collate,
				shuffle=False,
				drop_last=False,
			)