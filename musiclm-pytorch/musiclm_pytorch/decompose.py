import os
from musiclm_pytorch import MuLaNTrainer, MuLaN

import pickle
import torch

from torch.utils.data import Dataset

import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import einops
from tqdm import tqdm

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


class MuLanDataset(Dataset):
    def __init__(self, txt_pickle_path: Path, wav_pickle_path: Path):
                
        with open(wav_pickle_path, 'rb') as f:
            self.wavs = pickle.load(f)
        
        with open(txt_pickle_path, 'rb') as f:
            self.txts = pickle.load(f)

        self.num_data = len(self.txts)
                
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        # read wav from pt file, read txt from list
        return self.wavs[idx], self.txts[idx]
        
    
training_data = MuLanDataset(
    txt_pickle_path=Path('pkls/txts.pkl'),
    wav_pickle_path=Path('pkls/wavs.pkl'))

mulan_trainer = MuLaNTrainer(mulan=mulan, dataset=training_data, num_train_steps=1000, batch_size=16, grad_accum_every=16)
min_loss_path = 'results/mulan_min_loss.pt'
mulan_trainer.load(min_loss_path)


dataloader = DataLoader(training_data, batch_size=16)
# Read and preprocess the text
with open('../text_descriptions/music_characteristics.txt', 'r') as file:
    query_text = file.readlines()



_, txt_embeds = mulan_trainer.mulan.get_text_latents(raw_texts=query_text)
for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    wav = data[0].to(mulan_trainer.device)
    _, img_embeds = mulan_trainer.mulan.get_audio_latents(wav)
    remove_spatial = img_embeds.sum(dim=2)
    remove_spatial.transpose(1, 0)
