import os
import time
import logging
from itertools import chain, cycle
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import shutil
import math
import numpy as np
from .utils.log import ResultsLog
from .utils.optim import OptimRegime
from .utils.meters import AverageMeter
from .utils.cross_entropy import CrossEntropyLoss

from .config import PAD
from torch.nn.utils.rnn import PackedSequence

__all__ = ['Seq2SeqTrainer', 'MultiSeq2SeqTrainer', 'Img2SeqTrainer']


class AddLossModule(nn.Module):
    """adds a loss to module for easy parallelization"""

    def __init__(self, module, criterion, get_accuracy=True, ignore_index=PAD):
        super(AddLossModule, self).__init__()
        self.module = module
        self.criterion = criterion
        self.get_accuracy = get_accuracy
        self.ignore_index = ignore_index

    def forward(self, module_inputs, target):
        output = self.module(*module_inputs)
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        loss = self.criterion(output, target).view(1, 1)
        if self.get_accuracy:
            _, argmax = output.max(-1)
            invalid_targets = target.eq(self.ignore_index)
            accuracy = argmax.eq(target).masked_fill_(
                invalid_targets, 0).long().sum()
            return loss, accuracy.view(1, 1)
        else:
            return loss


class Seq2SeqTrainer(object):
    """class for Trainer.

     regime is an ordered list by epochs
     (can be a float indicating relative progress)"""

    def __init__(self, model, regime=None,
                 criterion=None,
                 label_smoothing=0,
                 print_freq=10,
                 eval_freq=1000,
                 save_freq=1000,
                 grad_clip=None,
                 embedding_grad_clip=None,
                 limit_num_tokens=None,
                 save_info={},
                 save_path='.',
                 checkpoint_filename='checkpoint%s.pth',
                 keep_checkpoints=5,
                 avg_loss_time=True,
                 distributed=False,
                 local_rank=0,
                 dtype=torch.float,
                 device_ids=None,
                 device="cuda"):
        super(Seq2SeqTrainer, self).__init__()
        self.model = model
        self.criterion = criterion or CrossEntropyLoss(
            ignore_index=PAD, smooth_eps=label_smoothing, reduction='sum')

        self.optimizer = OptimRegime(self.model, regime=regime)
        self.grad_clip = grad_clip
        self.embedding_grad_clip = embedding_grad_clip
        self.epoch = 0
        self.training_steps = 0
        self.save_info = save_info
        self.device = device
        self.dtype = dtype
        self.limit_num_tokens = limit_num_tokens
        self.print_freq = print_freq
        self.eval_freq = eval_freq
        self.perplexity = float('inf')
        self.device_ids = device_ids
        self.avg_loss_time = avg_loss_time
        self.model_with_loss = AddLossModule(self.model, self.criterion)
        self.distributed = distributed
        self.local_rank = local_rank
        if distributed:
            self.model_with_loss = DistributedDataParallel(
                self.model_with_loss,
                device_ids=[local_rank],
                output_device=local_rank)
        else:
            if isinstance(self.device_ids, tuple):
                self.model_with_loss = DataParallel(self.model_with_loss,
                                                    self.device_ids,
                                                    dim=0 if self.batch_first else 1)
        self.save_path = save_path
        self.save_freq = save_freq
        self.checkpoint_filename = checkpoint_filename
        self.keep_checkpoints = keep_checkpoints
        results_file = os.path.join(save_path, 'results')
        self.results = ResultsLog(results_file,
                                  params=save_info.get('config', None))

    @property
    def batch_first(self):
        return getattr(self.model.decoder, 'batch_first', False)

    def _batch_limit_tokens(self, src, target, limit_num=None):
        limit_num = limit_num or self.limit_num_tokens
        src, src_length = src
        target, target_length = target
        num_tokens = src.size(0) * src.size(1) + \
            target.size(0) * target.size(1)
        if num_tokens <= limit_num:
            return (src, src_length), (target, target_length)
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        src_max_length = np.maximum.accumulate(src_length)
        target_max_length = np.maximum.accumulate(target_length)
        sum_max_length = src_max_length + target_max_length
        num_tokens_batch = sum_max_length * \
            (np.arange(src.size(batch_dim)) + 1)
        B = int((num_tokens_batch > limit_num).argmax() - 1)
        Tsrc = int(src_max_length[B - 1])
        Ttarget = int(target_max_length[B - 1])

        src = (src.narrow(batch_dim, 0, B).narrow(time_dim, 0, Tsrc),
               src_length[:B])
        target = (target.narrow(batch_dim, 0, B).narrow(time_dim, 0, Ttarget),
                  target_length[:B])
        logging.debug(
            'Trimmed batch to %s as number of tokens was > %s, T = (%s, %s)' %
            (B, limit_num, Tsrc, Ttarget))
        return src, target

    def iterate(self, src_tuple, target_tuple, training=True):
        # limit number of tokens to avoid gpu overload
        if self.limit_num_tokens is not None:
            src_tuple, target_tuple = self._batch_limit_tokens(
                src_tuple, target_tuple)
        src, src_length = src_tuple
        target, target_length = target_tuple
        batch_dim, time_dim = (0, 1) if self.batch_first else (1, 0)
        num_words = sum(target_length) - target.size(batch_dim)

        if isinstance(src, PackedSequence) or \
                not isinstance(self.model_with_loss, DataParallel):
            if isinstance(src, PackedSequence):
                src = PackedSequence(src.data.to(self.device),
                                     src.batch_sizes.to(self.device))
            else:
                src = src.to(self.device)
            target = target.to(self.device)

        if self.batch_first:
            inputs = (src, target[:, :-1])
            target_labels = target[:, 1:].contiguous()
        else:
            inputs = (src, target[:-1])
            target_labels = target[1:]

        # compute output
        loss, accuracy = self.model_with_loss(inputs, target_labels)

        loss = loss.sum()
        loss_measure = float(loss / num_words)
        if self.avg_loss_time:
            loss /= num_words
        else:
            loss /= target.size(batch_dim)
        accuracy = float(accuracy.sum().float() / num_words)

        if training:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                if isinstance(self.grad_clip, dict):
                    clip_encoder = self.grad_clip.get('encoder', 0)
                    clip_decoder = self.grad_clip.get('decoder', 0)
                    if clip_encoder > 0:
                        clip_grad_norm_(
                            self.model.encoder.parameters(), clip_encoder)
                    if clip_decoder > 0:
                        clip_grad_norm_(
                            self.model.decoder.parameters(), clip_decoder)
                elif self.grad_clip > 0:  # grad_clip is a number
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self.embedding_grad_clip is not None and self.embedding_grad_clip > 0:
                if hasattr(self.model.encoder, 'embedder'):
                    clip_grad_norm_(self.model.encoder.embedder.parameters(),
                                    self.embedding_grad_clip)
                if hasattr(self.model.decoder, 'embedder'):
                    clip_grad_norm_(self.model.decoder.embedder.parameters(),
                                    self.embedding_grad_clip)
            self.optimizer.step()
        return loss_measure, accuracy, num_words

    def _feed_data(self, data_loader, num_iterations=None, training=True):
        if training:
            counter = cycle(range(self.keep_checkpoints))
            assert self.optimizer is not None

        num_iterations = num_iterations or len(data_loader) - 1
        batch_time = AverageMeter()
        data_time = AverageMeter()
        tok_time = AverageMeter()
        losses = AverageMeter()
        perplexity = AverageMeter()
        accuracy = AverageMeter()

        end = time.time()
        for i, (src, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if training:
                self.epoch += 1. / len(data_loader)
                if self.distributed:
                    data_loader.sampler.set_epoch(int(math.floor(self.epoch)))
                self.training_steps += 1
                # update optimizer according to epoch and steps
                self.optimizer.update(self.epoch, self.training_steps)

            try:
                # do a train/evaluate iteration
                loss, acc, num_words = self.iterate(src, target,
                                                    training=training)

                # measure accuracy and record loss
                losses.update(loss, num_words)
                perplexity.update(math.exp(loss), num_words)
                accuracy.update(acc, num_words)

                # measure elapsed time
                elapsed = time.time() - end
                batch_time.update(elapsed)
                tok_time.update(num_words / elapsed, num_words)

                end = time.time()
            except RuntimeError as err:
                if training and 'out of memory' in str(err):
                    logging.info(
                        'WARNING: ran out of memory, skipping batch')
                    torch.cuda.empty_cache()
                else:
                    raise err

            last_iteration = (i == len(data_loader) - 1)
            if i > 0 or last_iteration:
                if i % self.print_freq == 0 or last_iteration:
                    logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                 'Tok/sec {tok_time.val:.3f} ({tok_time.avg:.3f})\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                 'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                                 'Perplexity {perplexity.val:.4f} ({perplexity.avg:.4f})'.format(
                                     int(self.epoch), i, len(data_loader),
                                     phase='TRAINING' if training else 'EVALUATING',
                                     batch_time=batch_time, data_time=data_time, tok_time=tok_time,
                                     loss=losses, acc=accuracy, perplexity=perplexity))

                if training and (i % self.save_freq == 0 or last_iteration):
                    self.save(identifier=next(counter))

                if i % num_iterations == 0 or last_iteration:
                    yield {'loss': losses.avg, 'accuracy': accuracy.avg, 'perplexity': perplexity.avg}
                    losses.reset()
                    accuracy.reset()
                    perplexity.reset()

    def optimize(self, data_loader):
        # switch to train mode
        self.model.train()
        for result in self._feed_data(
                data_loader,
                num_iterations=self.eval_freq,
                training=True):
            yield result
            self.model.train()

    def evaluate(self, data_loader):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            for r in self._feed_data(data_loader, training=False):
                result = r
        return result

    def run(self, train_loader, val_loader=None):
        for train_results in self.optimize(train_loader):
            results = {'epoch': self.epoch,
                       'training steps': self.training_steps,
                       'training loss': train_results['loss'],
                       'training accuracy': train_results['accuracy'],
                       'training perplexity': train_results['perplexity']}
            plot_loss = ['training loss']
            plot_perplexity = ['training perplexity']
            plot_accuracy = ['training accuracy']
            if val_loader is not None:
                # evaluate on validation set
                val_results = self.evaluate(val_loader)

                # remember best prec@1 and save checkpoint
                is_best = val_results['perplexity'] < self.perplexity
                if is_best:
                    self.perplexity = val_results['perplexity']
                    self.save(is_best=True)

                results['validation loss'] = val_results['loss']
                results['validation perplexity'] = val_results['perplexity']
                results['validation accuracy'] = val_results['accuracy']
                plot_loss += ['validation loss']
                plot_perplexity += ['validation perplexity']
                plot_accuracy += ['validation accuracy']

            if self.distributed and self.local_rank > 0:
                continue
            self.results.add(**results)
            self.results.plot(x='training steps', y=plot_perplexity,
                              title='Perplexity', ylabel='perplexity')
            self.results.plot(x='training steps', y=plot_accuracy,
                              title='Accuracy', ylabel='accuracy')
            self.results.plot(x='training steps', y=plot_loss,
                              title='Loss', ylabel='loss')
            self.results.plot(x='epoch', y=plot_perplexity,
                              title='Perplexity', ylabel='perplexity')
            self.results.plot(x='epoch', y=plot_accuracy,
                              title='Accuracy', ylabel='accuracy')
            self.results.plot(x='epoch', y=plot_loss,
                              title='Loss', ylabel='loss')
            self.results.save()

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.epoch = checkpoint['epoch']
            self.training_steps = checkpoint['training_steps']
            self.perplexity = checkpoint['perplexity']
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         filename, self.epoch)
        else:
            logging.error('invalid checkpoint: {}'.format(filename))

    def save(self, filename=None, identifier=None, is_best=False, save_all=False):
        state = {
            'epoch': self.epoch,
            'training_steps': self.training_steps,
            'state_dict': self.model.state_dict(),
            'perplexity': getattr(self, 'perplexity', None),
        }
        state = dict(list(state.items()) + list(self.save_info.items()))
        identifier = identifier or ''
        filename = filename or self.checkpoint_filename % identifier
        filename = os.path.join(self.save_path, filename)
        logging.info('saving model to %s' % filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(
                self.save_path, 'model_best.pth'))
        if save_all:
            shutil.copyfile(filename, os.path.join(
                self.save_path, 'checkpoint_epoch_%s.pth' % self.epoch))


class MultiSeq2SeqTrainer(Seq2SeqTrainer):
    """class for Trainer."""

    def iterate(self, src, target, training=True):
        batch_dim = 0 if self.batch_first else 1
        time_dim = 1 if self.batch_first else 0

        def pad_copy(x, length):
            if x.size(time_dim) == length:
                return x
            else:
                sz = (x.size(0), length) if self.batch_first \
                    else (length, x.size(1))
                padded = x.new().resize_(*sz).fill_(0)
                padded.narrow(time_dim, 0, x.size(time_dim)).copy_(x)
                return padded

        src, src_length = src
        target, target_length = target
        max_length = max(src.size(time_dim), target.size(time_dim))
        src = pad_copy(src, max_length)
        target = pad_copy(target, max_length)

        src_full = torch.cat([src, target], batch_dim)
        src_length_full = src_length + target_length

        target_full = torch.cat([target, src], batch_dim)
        target_length_full = target_length + src_length
        src = (src_full, src_length_full)
        target = (target_full, target_length_full)
        return super(MultiSeq2SeqTrainer, self).iterate(src, target, training)


class Img2SeqTrainer(Seq2SeqTrainer):
    """class for Trainer."""

    def iterate(self, src_img, target, training=True):
        src = (src_img, None)
        return super(Img2SeqTrainer, self).iterate(src, target, training)
