"""WavLM-large + ECAPA-TDNN speaker embeddings: UniSpeech's speaker-verification model (the one Seed-TTS-Eval's SIM uses), for
telling apart voices that Nemotron's diarizer hears as one (two young women, two old men). Written from the published
architecture; the weights are the safetensors of prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native. One unpadded
16 kHz clip at a time, so no attention masks."""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import WavLMConfig, WavLMModel


def wavlm_large_config():
    return WavLMConfig(conv_bias=False, conv_dim=(512,) * 7, conv_kernel=(10, 3, 3, 3, 3, 2, 2), conv_stride=(5, 2, 2, 2, 2, 2, 2),
                       do_stable_layer_norm=True, feat_extract_norm="layer", hidden_size=1024, intermediate_size=4096,
                       num_attention_heads=16, num_hidden_layers=24, num_conv_pos_embedding_groups=16, num_conv_pos_embeddings=128,
                       num_buckets=320, max_bucket_distance=800, layer_norm_eps=1e-5, apply_spec_augment=False, layerdrop=0.0)


class ConvReluBn(nn.Module):
    def __init__(self, cin, cout, k=1, p=0):
        super().__init__()
        self.conv, self.bn = nn.Conv1d(cin, cout, k, padding=p), nn.BatchNorm1d(cout)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2(nn.Module):
    def __init__(self, p, d):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(64, 64, 3, padding=p, dilation=d) for _ in range(7))
        self.bns = nn.ModuleList(nn.BatchNorm1d(64) for _ in range(7))

    def forward(self, x):
        pieces, out, prev = torch.split(x, 64, dim=1), [], None
        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            prev = b(F.relu(c(pieces[i] if prev is None else prev + pieces[i])))
            out.append(prev)
        return torch.cat(out + [pieces[-1]], dim=1)


class SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(512, 128), nn.Linear(128, 512)

    def forward(self, x):
        return x * torch.sigmoid(self.linear2(F.relu(self.linear1(x.mean(dim=2))))).unsqueeze(2)


class SERes2Block(nn.Module):
    def __init__(self, p, d):
        super().__init__()
        self.Conv1dReluBn1, self.Res2Conv1dReluBn, self.Conv1dReluBn2, self.SE_Connect = ConvReluBn(512, 512), Res2(p, d), ConvReluBn(512, 512), SE()

    def forward(self, x):
        return x + self.SE_Connect(self.Conv1dReluBn2(self.Res2Conv1dReluBn(self.Conv1dReluBn1(x))))


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1, self.linear2 = nn.Conv1d(1536, 128, 1), nn.Conv1d(128, 1536, 1)

    def forward(self, x):
        a = torch.softmax(self.linear2(torch.tanh(self.linear1(x))), dim=2)
        mean = (a * x).sum(2)
        return torch.cat([mean, ((a * x.square()).sum(2) - mean.square()).clamp_min(1e-9).sqrt()], dim=1)


class ECAPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(1024)
        self.layer1 = ConvReluBn(1024, 512, 5, 2)
        self.layer2, self.layer3, self.layer4 = SERes2Block(2, 2), SERes2Block(3, 3), SERes2Block(4, 4)
        self.conv, self.pooling, self.bn, self.linear = nn.Conv1d(1536, 1536, 1), Pool(), nn.BatchNorm1d(3072), nn.Linear(3072, 256)

    def forward(self, f):
        o2 = self.layer2(self.layer1(self.instance_norm(f)))
        o3 = self.layer3(o2)
        o4 = self.layer4(o3)
        return self.linear(self.bn(self.pooling(F.relu(self.conv(torch.cat([o2, o3, o4], dim=1))))))


class WavLMLargeECAPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel(wavlm_large_config())
        self.feature_weight = nn.Parameter(torch.empty(25))
        self.ecapa = ECAPA()

    def forward(self, wave16k):
        """One 16 kHz mono clip [T] -> its 256-d embedding."""
        w = F.layer_norm(wave16k, wave16k.shape)[None]
        hs = self.wavlm(w, output_hidden_states=True).hidden_states
        feats = sum(k * h for k, h in zip(F.softmax(self.feature_weight, dim=0), hs))
        return self.ecapa(feats.transpose(1, 2) + 1e-6)[0]


def is_wavlm_large_ecapa(state_dict: dict) -> bool:
    return "ecapa.linear.weight" in state_dict and "feature_weight" in state_dict
