import os.path as osp
import math
import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchmetrics import Accuracy
from timm.loss import SoftTargetCrossEntropy

import utils
from mixup import Mixup
from optimizer import build_optimizer
from transformer import ClassificationHead
from video_transformer import TimeSformer, ViViT, MaskFeat

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, base_lr, objective, min_lr=5e-5, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases following the
	values of the cosine function between 0 and `pi * cycles` after a warmup
	period during which it increases linearly between 0 and base_lr.
	"""
	# step means epochs here
	def lr_lambda(current_step):
		current_step += 1
		if current_step <= num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps)) # * base_lr 
		progress = min(float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)), 1)
		if objective == 'mim':
			return 0.5 * (1. + math.cos(math.pi * progress))
		else:
			factor = 0.5 * (1. + math.cos(math.pi * progress))
			return factor*(1 - min_lr/base_lr) + min_lr/base_lr

	return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class VideoTransformer(pl.LightningModule):

	def __init__(self, 
				 configs,
				 trainer,
				 ckpt_dir,
				 do_eval,
				 do_test,
				 n_crops=3):
		super().__init__()
		self.configs = configs
		self.trainer = trainer

		# build models
		if self.configs.objective =='mim': 
			self.model = MaskFeat(pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], feature_dim=2*2*2*3*9)
		else: # supervised
			# load pretrain weights from pretrained weight path and model.init_weights method
			if self.configs.arch == 'vivit':
				self.model = ViViT(
					pretrain_pth=self.configs.pretrain_pth,
					weights_from=self.configs.weights_from,
					img_size=self.configs.img_size,
					num_frames=self.configs.num_frames,
					attention_type=self.configs.attention_type)
			elif self.configs.arch == 'timesformer':
				self.model = TimeSformer(
					pretrain_pth=self.configs.pretrain_pth,
					weights_from=self.configs.weights_from,
					img_size=self.configs.img_size,
					num_frames=self.configs.num_frames,
					attention_type=self.configs.attention_type)
			else: # arch-mvit
				self.model = MaskFeat(
					pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2]], 
					feature_dim=2*2*2*3*9,
					pretrain_pth=self.configs.pretrain_pth,
					img_size=self.configs.img_size,
					num_frames=self.configs.num_frames)
				for name, param in self.model.decoder_pred.named_parameters():
					param.requires_grad = False
		
			self.cls_head = ClassificationHead(
				self.configs.num_class, self.model.embed_dims, eval_metrics=self.configs.eval_metrics)
			
			self.max_top1_acc = 0
			self.train_top1_acc = Accuracy(task='multiclass', num_classes=self.configs.num_class)
			# init running counters of tp, tn, fp, fn
			self.reset_metrics()

			if self.configs.mixup:
				self.mixup_fn = Mixup(num_classes=self.configs.num_class)
				self.loss_fn = SoftTargetCrossEntropy()
			else:
				# weight positive classes more heavily
				device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
				class_weights = torch.tensor([1.0 / 0.7, 1.0 / 0.3], dtype=torch.float32).to(device) 
				self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

		# common
		self.iteration = 0
		self.data_start = 0
		self.ckpt_dir = ckpt_dir
		self.do_eval = do_eval
		self.do_test = do_test
		if self.do_eval:
			self.val_top1_acc = Accuracy(task='multiclass', num_classes=self.configs.num_class)
		if self.do_test:
			self.n_crops = n_crops
			self.test_top1_acc = Accuracy(task='multiclass', num_classes=self.configs.num_class)

	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def reset_metrics(self):
		# reset sensitivity and specificity for each epoch
		self.tp = 0.0
		self.tn = 0.0
		self.fp = 0.0
		self.fn = 0.0

	def configure_optimizers(self):
		# build optimzer
		is_pretrain = not (self.configs.objective == 'supervised')
		if self.configs.objective == 'supervised' and self.configs.eval_metrics == 'linear_prob':
			model = self.cls_head.module if hasattr(self.cls_head, 'module') else self.cls_head
			optimizer = build_optimizer(self.configs, model, is_pretrain=is_pretrain)
		else:
			optimizer = build_optimizer(self.configs, self, is_pretrain=is_pretrain)
		
		# lr schedule
		lr_scheduler = None
		lr_schedule = self.configs.lr_schedule 
		if lr_schedule == 'multistep':
			lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
														  milestones=[5, 11],
														  gamma=0.1)
		elif lr_schedule == 'cosine':
			lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
														  num_warmup_steps=self.configs.warmup_epochs, 
														  num_training_steps=self.trainer.max_epochs,
														  base_lr=self.configs.lr,
														  min_lr=self.configs.min_lr,
														  objective=self.configs.objective)
		return [optimizer], [lr_scheduler]

	def parse_batch(self, batch, train):
		if self.configs.objective == 'mim':
			inputs, labels, mask, cube_marker, =  *batch,
			return inputs, labels, mask, cube_marker
		else:
			inputs, labels, = *batch,
			if self.configs.mixup and train:
				inputs, labels = self.mixup_fn(inputs, labels)
			return inputs, labels

	# epoch schedule
	def _get_momentum(self, base_value, final_value):
		return final_value - (final_value - base_value) * (math.cos(math.pi * self.trainer.current_epoch / self.trainer.max_epochs) + 1) / 2

	def _weight_decay_update(self):
		for i, param_group in enumerate(self.optimizers().optimizer.param_groups):
			if i == 1:  # only the first group is regularized
				param_group["weight_decay"] = self._get_momentum(base_value=self.configs.weight_decay, final_value=self.configs.weight_decay_end)

	def log_step_state(self, data_time, top1_acc=0):
		self.log("time",float(f'{time.perf_counter()-self.data_start:.3f}'),prog_bar=True)
		self.log("data_time", data_time, prog_bar=True)
		if self.configs.objective == 'supervised':
			self.log("top1_acc",top1_acc,on_step=True,on_epoch=False,prog_bar=True)

		return None

	def get_progress_bar_dict(self):
		# don't show the version number
		items = super().get_progress_bar_dict()
		items.pop("v_num", None)
		
		return items

	# Trainer Pipeline
	def training_step(self, batch, batch_idx):
		data_time = float(f'{time.perf_counter() - self.data_start:.3f}')
		if self.configs.objective == 'mim':
			inputs, labels, mask, cube_marker = self.parse_batch(batch, train=True)
			preds, loss = self.model(inputs, labels, mask, cube_marker)
			self.log_step_state(data_time)
			return {'loss': loss, 'data_time': data_time}
		else:
			inputs, labels = self.parse_batch(batch, train=True)
			if self.configs.eval_metrics == 'linear_prob':
				with torch.no_grad():
					self.model.eval()
					preds = self.model(inputs)
			else:
				if self.configs.arch == 'mvit':
					preds = self.model.forward_features(inputs)[:, 0]
				else:
					preds = self.model(inputs)
			preds = self.cls_head(preds)
			loss = self.loss_fn(preds, labels)
			predicted_classes = preds.argmax(dim=1)

			# sensitivity / specificity
			self.tp += ((predicted_classes == 1) & (labels == 1)).sum().item()
			self.tn += ((predicted_classes == 0) & (labels == 0)).sum().item()
			self.fp += ((predicted_classes == 1) & (labels == 0)).sum().item()
			self.fn += ((predicted_classes == 0) & (labels == 1)).sum().item()

			top1_acc = self.train_top1_acc(predicted_classes, labels)
			self.log_step_state(data_time, top1_acc)
			return {'loss': loss, 'data_time': data_time}
	
	def on_after_backward(self):
		self._weight_decay_update()
		# log learning daynamic
		lr = self.optimizers().optimizer.param_groups[0]['lr']
		self.log("lr",lr,on_step=True,on_epoch=False,prog_bar=True)

	def on_train_epoch_end(self):
		timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
		if self.configs.objective == 'supervised':
			mean_top1_acc = self.train_top1_acc.compute()
			self.print(f'{timestamp} - Evaluating mean train',
					   f'top1_acc:{mean_top1_acc:.3f},')
			self.train_top1_acc.reset()

			# sensitivity / specificity
			sensitivity = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
			specificity = self.tn / (self.tn + self.fp) if self.tn + self.fp > 0 else 0
			self.log('train_sensitivity', sensitivity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
			self.log('train_specificity', specificity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
			print(f'         Mean train sensitivity: {sensitivity:.3f}')
			print(f'         Mean train specificity: {specificity:.3f}')
			self.reset_metrics()

		# save last checkpoint
		save_path = osp.join(self.ckpt_dir, 'last_checkpoint.pth')
		self.trainer.save_checkpoint(save_path)

		if self.configs.objective != 'supervised' and (self.trainer.current_epoch+1) % self.configs.save_ckpt_freq == 0:
			save_path = osp.join(self.ckpt_dir,
								 f'{timestamp}_'+
								 f'ep_{self.trainer.current_epoch}.pth')
			self.trainer.save_checkpoint(save_path)

	def validation_step(self, batch, batch_indx):
		if self.do_eval:
			inputs, labels = self.parse_batch(batch, train=False)
			if self.configs.eval_metrics == 'linear_prob':
				with torch.no_grad():
					preds = self.model(inputs)
			else:
				if self.configs.arch == 'mvit':
					preds = self.model.forward_features(inputs)[:, 0]
				else:
					preds = self.model(inputs)
			preds = self.cls_head(preds)
			predicted_classes = preds.argmax(dim=1)

			# sensitivity / specificity
			self.tp = ((predicted_classes == 1) & (labels == 1)).sum().item()
			self.tn = ((predicted_classes == 0) & (labels == 0)).sum().item()
			self.fp = ((predicted_classes == 1) & (labels == 0)).sum().item()
			self.fn = ((predicted_classes == 0) & (labels == 1)).sum().item()

			self.val_top1_acc(predicted_classes, labels)
			self.data_start = time.perf_counter()
	
	def on_validation_epoch_end(self):
		if self.do_eval:
			mean_top1_acc = self.val_top1_acc.compute()
			timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
			self.print(f'{timestamp} - Evaluating mean valid',
					   f'top1_acc:{mean_top1_acc:.3f}, ')
			self.val_top1_acc.reset()

			# sensitivity / specificity
			sensitivity = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
			specificity = self.tn / (self.tn + self.fp) if self.tn + self.fp > 0 else 0
			self.log('val_sensitivity', sensitivity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
			self.log('val_specificity', specificity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
			print(f'         Mean val sensitivity: {sensitivity:.3f}')
			print(f'         Mean val specificity: {specificity:.3f}')
			self.reset_metrics()

			# save best checkpoint
			if mean_top1_acc > self.max_top1_acc:
				save_path = osp.join(self.ckpt_dir,
									 f'{timestamp}_'+
									 f'ep_{self.trainer.current_epoch}_'+
									 f'top1_acc_{mean_top1_acc:.3f}.pth')
				self.trainer.save_checkpoint(save_path)
				self.max_top1_acc = mean_top1_acc
			
	def test_step(self, batch, batch_idx):
		if self.do_test:
			inputs, labels = self.parse_batch(batch)
			preds = self.cls_head(self.model(inputs))
			preds = preds.view(-1, self.n_crops, self.configs.num_class).mean(1)
			predicted_classes = preds.argmax(dim=1)

			# sensitivity / specificity
			self.tp += ((predicted_classes == 1) & (labels == 1)).sum().item()
			self.tn += ((predicted_classes == 0) & (labels == 0)).sum().item()
			self.fp += ((predicted_classes == 1) & (labels == 0)).sum().item()
			self.fn += ((predicted_classes == 0) & (labels == 1)).sum().item()

			self.test_top1_acc(predicted_classes, labels)
			self.data_start = time.perf_counter()
	
	def on_test_epoch_end(self):
		if self.do_test:
			mean_top1_acc = self.test_top1_acc.compute()
			timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
			self.print(f'{timestamp} - Evaluating mean ',
					   f'top1_acc:{mean_top1_acc:.3f}, ')
			self.test_top1_acc.reset()

			# sensitivity / specificity
			sensitivity = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0
			specificity = self.tn / (self.tn + self.fp) if self.tn + self.fp > 0 else 0
			self.log('test_sensitivity', sensitivity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
			self.log('test_specificity', specificity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
			print(f'         Mean test sensitivity: {sensitivity:.3f}')
			print(f'         Mean test specificity: {specificity:.3f}')
			self.reset_metrics()
