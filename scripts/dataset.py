import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

class RIRDataset(Dataset):
    def __init__(self, data_dir, sample_rate=48000, duration=1.0):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.total_samples = int(sample_rate * duration)
        self.head_samples = int(sample_rate * 0.05) # 50ms = 2400 muestras
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_list)

    def preprocess(self, audio):
        # 1. Eliminar el retraso inicial (buscando el pico máximo)
        peak_idx = torch.argmax(torch.abs(audio))
        audio = audio[peak_idx:]
        
        # 2. Normalizar amplitud absoluta a 1.0 
        max_amp = torch.max(torch.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp
            
        # 3. Ajustar a la longitud deseada (1 segundo)
        if len(audio) < self.total_samples:
            audio = torch.nn.functional.pad(audio, (0, self.total_samples - len(audio)))
        else:
            audio = audio[:self.total_samples]
            
        return audio

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        audio, sr = torchaudio.load(file_path)
        
        # Convertir a mono si es necesario
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)
        else:
            audio = audio.squeeze(0)
            
        # Resamplear si la frecuencia no es 48kHz
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
            
        audio = self.preprocess(audio)
        
        # Separar en Head (0-50ms) y Tail (50ms-1s) [cite: 199]
        head = audio[:self.head_samples]
        tail = audio[self.head_samples:]
        
        return head.unsqueeze(0), tail.unsqueeze(0) # Formato (Canal, Muestras)