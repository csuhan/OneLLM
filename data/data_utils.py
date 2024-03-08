from PIL import Image
import torch
import torchaudio
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

T_random_resized_crop = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


# image transform
transform_img_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(
        0.75, 1.3333), interpolation=3, antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


class PairRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias) for img in imgs]


class PairToTensor(transforms.ToTensor):
    def __call__(self, pics):
        return [F.to_tensor(pic) for pic in pics]


class PairNormalize(transforms.Normalize):
    def forward(self, tensors):
        return [F.normalize(tensor, self.mean, self.std, self.inplace) for tensor in tensors]


transform_pairimg_train = transforms.Compose([
    PairRandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(
        0.75, 1.3333), interpolation=3, antialias=None),  # 3 is bicubic
    PairToTensor(),
    PairNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = torch.mean(xyz, dim=0)
    xyz = xyz - centroid
    m = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=1)))
    xyz = xyz / m

    pc = torch.cat((xyz, other_feature), dim=1)
    return pc


def make_audio_features(wav_name, mel_bins=128, target_length=1024, aug=False):
    waveform, sr = torchaudio.load(wav_name)
    # assert sr == 16000, 'input audio sampling rate must be 16kHz'
    if sr != 16000:
        trans = torchaudio.transforms.Resample(sr, 16000)
        waveform = trans(waveform)

    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if aug:
        freqm = torchaudio.transforms.FrequencyMasking(48)
        timem = torchaudio.transforms.TimeMasking(192)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        fbank = freqm(fbank)
        fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank