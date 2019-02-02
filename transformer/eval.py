from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

import streamlit as st
from PIL import Image

from captions_pred import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=0,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='',
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1,
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0,
                help='if we need to calculate loss.')
parser.add_argument('--num_tags', type=int, default=30,
                help='number of hashtags')


opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = cPickle.load(f, encoding='latin1')


# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    #opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
#model.cuda()
model.to(device)
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
    vars(opt))

#print ('preds: ', split_predictions)
#print ('preds caption:', split_predictions[0]['caption'])

#print('loss: ', loss)
if lang_stats:
    print(lang_stats)

caption = split_predictions[0]['caption'].split()
# Remove stopWords
caption_tags = remove_stopwords(caption)

#Download glove - http://nlp.stanford.edu/data/glove.6B.zip
# Save glove file as word2vec file - only need to do this once
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = './models/glove/glove.6B/glove.6B.100d.txt'
word2vec_output_file = './models/glove/glove.6B/glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

# Load the popular instagram hashtag file and extract the categories
with open('./models/glove/hashtag.pkl', 'rb') as f:
    hashtag = cPickle.load(f)

# Load the banned hashtags
with open('./models/glove/banned.pkl', 'rb') as f:
    banned = set(cPickle.load(f))

# Get the relevant popular hashtags after removing the banned tags
insta_tags = list(set(get_hashtags(caption_tags, hashtag)) - banned)

# Load generic tags for images without relevant categories
with open('./models/glove/generic_hashtags.pkl', 'rb') as f:
    generic_tags = list(set(cPickle.load(f)) - banned)


# Choose random tags from insta_tags and generic_tags
if len(insta_tags)<(opt.num_tags - len(caption_tags)):
    random_tags = insta_tags
    random_tags.extend(random.sample(generic_tags, (opt.num_tags-len(insta_tags)-len(caption_tags))))
else:
    random_tags = random.sample(insta_tags, opt.num_tags- len(caption_tags))

# Display image
image = Image.open('./images/baby.jpg')
st.image(image)
#st.image(image, caption=' '.join(hash_cap))

# Display the caption
#st.write('actual caption: ', caption)

# Display the caption tags and the hashtags
hash_caption_tags = ["#" + word for word in caption_tags]
hash_instatags = ["#" + word for word in random_tags] + hash_caption_tags
st.write(' '.join(tag for tag in hash_instatags))




if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
