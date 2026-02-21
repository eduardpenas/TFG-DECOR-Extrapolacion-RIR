import os
import torch
import torchaudio
from torch.utils.data import Dataset


def schroeder_edc_torch(signal: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if signal.numel() == 0:
        return torch.zeros(1, dtype=torch.float32)

    energy = signal.to(torch.float64).pow(2)
    reverse_energy = torch.flip(energy, dims=(0,))
    reverse_cumsum = torch.cumsum(reverse_energy, dim=0)
    edc = torch.flip(reverse_cumsum, dims=(0,))
    edc = edc / (edc[0] + eps)
    return edc.to(torch.float32)


class RIRDataset(Dataset):
    def __init__(
        self,
        data_dir,
        sample_rate=48000,
        duration=1.0,
        head_ms=50.0,
        compute_edc_gt=False,
        return_dict=False,
    ):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.total_samples = int(sample_rate * duration)
        self.head_samples = int(sample_rate * (head_ms / 1000.0))
        self.compute_edc_gt = compute_edc_gt
        self.return_dict = return_dict
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])

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

    def split_head_tail(self, rir):
        head = rir[:self.head_samples]
        tail = rir[self.head_samples:]
        return head, tail

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

        # Separar en Head (0-50ms por defecto) y Tail
        head, tail = self.split_head_tail(audio)
        head = head.unsqueeze(0)
        tail = tail.unsqueeze(0)

        if not self.compute_edc_gt:
            if self.return_dict:
                return {
                    "head": head,
                    "tail": tail,
                    "path": file_path,
                }
            return head, tail

        edc_tail = schroeder_edc_torch(tail.squeeze(0)).unsqueeze(0)

        if self.return_dict:
            return {
                "head": head,
                "tail": tail,
                "edc_tail_gt": edc_tail,
                "path": file_path,
            }

        return head, tail, edc_tail