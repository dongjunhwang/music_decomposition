import os
import pickle
import torch

from torch.utils.data import Dataset

import torch
from trainer import MuLaNTrainer
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import einops
from tqdm import tqdm

import spacy
import re

# Set the CUDA_VISIBLE_DEVICES environment variable to specify which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 256,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)
mulan.to('cuda')


class MusicDataset(Dataset):
    def __init__(self, musiccaps_dataset_pkl_path: Path, sdd_dataset_pkl_path: Path):

        self.musiccaps_dataset = pickle.load(open(musiccaps_dataset_pkl_path, 'rb'))
        self.num_musiccaps = len(self.musiccaps_dataset)

        self.sdd_dataset = pickle.load(open(sdd_dataset_pkl_path, 'rb'))
        self.num_sdd = len(self.sdd_dataset)

        self.num_data = self.num_musiccaps + self.num_sdd

        self.wav_duration = 16000 * 10  # 10 seconds

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if idx < self.num_musiccaps:
            wav = self.musiccaps_dataset[idx][0]
            cap = self.musiccaps_dataset[idx][1]

        else:
            # get the sdd_dataset_index
            idx = idx - self.num_musiccaps
            real_wav_len = self.sdd_dataset[idx][2]

            # randomly select a starting point
            start_point = torch.randint(0, real_wav_len - self.wav_duration, (1,)).item()
            wav = self.sdd_dataset[idx][0][start_point:start_point + self.wav_duration]
            cap = self.sdd_dataset[idx][1]

        return wav, cap

training_data = MusicDataset(
    musiccaps_dataset_pkl_path=Path('pkls/musiccaps_dataset.pkl'),
    sdd_dataset_pkl_path=Path('pkls/sdd_dataset.pkl')
)

mulan_trainer = MuLaNTrainer(mulan=mulan, dataset=training_data, num_train_steps=1000, batch_size=16, grad_accum_every=16)
min_loss_path = 'results/mulan_min_loss.pt'
mulan_trainer.load(min_loss_path)


dataloader = DataLoader(training_data, batch_size=16)
# Read and preprocess the text
with open('../text_descriptions/music_characteristics.txt', 'r') as file:
    query_text = file.readlines()

special_characters = ['.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '~', '`']
get_lemma = spacy.load('en_core_web_sm')
for caption in query_text:
    for special_character in special_characters:
        caption = caption.replace(special_character, ' ')
        caption = caption.lower()

    words = re.findall(r'\w+|\.', caption)
    normalized_caption = ' '.join([token.lemma_ for word in words for token in get_lemma(word)])

_, txt_embeds = mulan_trainer.mulan.get_text_latents(raw_texts=query_text)
with torch.no_grad():
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        wav = data[0].to(mulan_trainer.device)
        _, audio_layers, img_embeds = mulan_trainer.mulan.get_audio_latents(wav)
        remove_spatial_head = img_embeds.mean(dim=2) # [Num of Layer, Batch Size, ?, Num of Head, Dim]
        remove_spatial_layer = audio_layers.mean(dim=2)
        remove_spatial_head, remove_spatial_layer = remove_spatial_head.transpose(1, 0), remove_spatial_layer.transpose(1, 0)


