import os
import pickle
import torch
import spacy
import re
import argparse
import logging

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from musiclm_pytorch.trainer import MuLaNTrainer
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from src.test import pca_dimension_reduction, lda_dimension_reduction
from src.decompose_text import replace_with_iterative_removal

# Set the CUDA_VISIBLE_DEVICES environment variable to specify which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def embed_decompose(audio_head_features, text_embed, text_descriptions,
                    reduction='pca', texts_per_head=2, w_ov_rank=80, device='cuda',
                    only_layer=False):
    if only_layer:
        num_layers = audio_head_features.shape[1]
        results_per_layer = []
        for layer_idx in range(num_layers):
            reconstruct, results = replace_with_iterative_removal(audio_head_features[:, layer_idx].detach().cpu().numpy(),
                                                                  text_embed.detach().cpu().numpy(),
                                                                  text_descriptions,
                                                                  iters=texts_per_head, rank=w_ov_rank,
                                                                  device=device)
            results_per_layer.append({'text_set': results})
    else:
        num_layers = audio_head_features.shape[1]
        num_heads = audio_head_features.shape[2]

        reduced_text_embed = None
        if reduction == 'pca':
            reduced_text_embed = pca_dimension_reduction(text_embed[:, None, None, :], audio_head_features)
        elif reduction == 'lda':
            reduced_text_embed = lda_dimension_reduction(text_embed[:, None, None, :], audio_head_features)

        reduced_text_embed = reduced_text_embed.squeeze().numpy()
        results_per_layer = []
        for layer_idx in range(num_layers):
            results_per_head = []
            for head_idx in range(num_heads):
                reconstruct, results = replace_with_iterative_removal(audio_head_features[:, layer_idx, head_idx].numpy(),
                                                                      reduced_text_embed, text_descriptions,
                                                                      iters=texts_per_head, rank=w_ov_rank, device=device)
                results_per_head.append({'text_set': results})
            results_per_layer.append(results_per_head)

    return results_per_layer


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='/home/music/Data/genres_original/')
    parser.add_argument('--text_descriptions_file', type=str, default='../text_descriptions/music_characteristics.txt')
    parser.add_argument('--ckpt_path', type=str, default='music_audioset_epoch_15_esc_90.14.pt')
    parser.add_argument('--save_folder', type=str, default='./output')
    parser.add_argument('--ext_format', type=str, default='**/*.wav')
    parser.add_argument('--texts_per_head', type=int, default=10)
    parser.add_argument('--reduction', type=str, default='pca')
    parser.add_argument('--amodel', type=str, default='HTSAT-base')
    parser.add_argument('--num_of_samples', type=int, default=900)
    parser.add_argument('--divide_samples', type=int, default=20)

    return parser.parse_args()


if __name__ == '__main__':
    args = config()
    fh = logging.FileHandler(os.path.join(args.save_folder, 'log_mulan.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info("Preparing the dataset and model...")


    training_data = MusicDataset(
        musiccaps_dataset_pkl_path=Path('musiclm_pytorch/pkls/musiccaps_dataset.pkl'),
        sdd_dataset_pkl_path=Path('musiclm_pytorch/pkls/sdd_dataset.pkl')
    )

    mulan_trainer = MuLaNTrainer(mulan=mulan, dataset=training_data, num_train_steps=1000, batch_size=16, grad_accum_every=16)
    min_loss_path = 'musiclm_pytorch/results/mulan_min_loss.pt'
    mulan_trainer.load(min_loss_path)


    dataloader = DataLoader(training_data, batch_size=16)
    # Read and preprocess the text
    with open('text_descriptions/music_characteristics.txt', 'r') as file:
        query_text = file.readlines()
        query_text = list(map(str.rstrip, query_text))

    special_characters = ['.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '~', '`']
    get_lemma = spacy.load('en_core_web_sm')
    for caption in query_text:
        for special_character in special_characters:
            caption = caption.replace(special_character, ' ')
            caption = caption.lower()

        words = re.findall(r'\w+|\.', caption)
        normalized_caption = ' '.join([token.lemma_ for word in words for token in get_lemma(word)])

    _, txt_embeds = mulan_trainer.mulan.get_text_latents(raw_texts=query_text)
    logger.info("Decompose the text per head.")
    only_layer = True
    with torch.no_grad():
        concat_features = []
        for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            wav = data[0].to(mulan_trainer.device)
            _, audio_layers, img_embeds = mulan_trainer.mulan.get_audio_latents(wav)
            remove_spatial_head = img_embeds.mean(dim=2) # [Num of Layer, Batch Size, ?, Num of Head, Dim]
            remove_spatial_layer = audio_layers.mean(dim=2)
            remove_spatial_head, remove_spatial_layer = remove_spatial_head.transpose(1, 0), remove_spatial_layer.transpose(1, 0)
            if only_layer:
                concat_features.append(remove_spatial_layer.cpu())
            else:
                concat_features.append(remove_spatial_head.cpu())
        concat_features = torch.cat(concat_features, dim=0)
        results = embed_decompose(concat_features, txt_embeds, query_text,
                                  texts_per_head=args.texts_per_head, reduction=args.reduction,
                                  only_layer=only_layer)

    with open(os.path.join(args.save_folder, 'log_mulan_only_layer.txt'), "w") as f:
        if only_layer:
            for i, res in enumerate(results):
                f.write(f"---------Layer {i}---------\n")
                for r in res['text_set']:
                    f.write(r+"\n")
                f.write(f"--------------------------\n")
        else:
            for i, res in enumerate(results):
                for j, res_head in enumerate(res):
                    f.write(f"---------Layer {i}, Head {j}---------\n")
                    for r in res_head['text_set']:
                        f.write(r+"\n")
                    f.write(f"-------------------------------------\n")