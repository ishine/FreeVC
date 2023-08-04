import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

# import commons
# import modules
import numpy as np
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding

import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
# from modules.commons import init_weights, get_padding
from utils import f0_to_coarse, energy_to_coarse

def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)

class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1)
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                                 device=f0.device)
            # fundamental component
            fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

            # generate sine waveforms
            sine_waves = self._f02sine(fn) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
    

class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv

class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x


class Encoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class ProsodyEncoder(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      kernel_size,
      n_layers,
      gin_channels=0,
      filter_channels=None,
      n_heads=None,
      p_dropout=None):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.f0_emb = nn.Embedding(256, hidden_channels)

    self.enc_ =  attentions.Encoder(
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)

  def forward(self, x, x_mask, f0=None, noice_scale=1):
    x = x + self.f0_emb(f0).transpose(1,2)
    x = self.enc_(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask

    return z, m, logs, x_mask

# If energy
class ProsodyEncoder_energy(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      kernel_size,
      n_layers,
      gin_channels=0,
      filter_channels=None,
      n_heads=None,
      p_dropout=None):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.f0_emb = nn.Embedding(256, hidden_channels)

    self.energy_emb = nn.Embedding(256, hidden_channels)

    self.enc_ =  attentions.Encoder(
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)

  def forward(self, x, x_mask, f0=None, energy = None, noice_scale=1):

    # print(energy.shape)
    # print(self.energy_emb(energy).shape)
    # print(x.shape)

    x = x + self.f0_emb(f0).transpose(1,2)
    x = x + self.energy_emb(energy).transpose(1,2)

    x = self.enc_(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs) * noice_scale) * x_mask

    return z, m, logs, x_mask

class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, 
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels,
                sampling_rate, energy_agg_type, energy_linear_dim, use_energy):
        
        super(Generator, self).__init__()

        self.use_energy_convs = False # default value
        self.use_energy = use_energy
        self.energy_linear_dim = energy_linear_dim

        ## Energy args
        if(self.use_energy):
            if(energy_agg_type == 'one_step'):
                self.energy_emb = nn.Embedding(256, upsample_initial_channel)
            elif(energy_agg_type == 'all_step'):
                self.energy_noise_convs = nn.ModuleList()
                if(energy_linear_dim == 1):
                    print('Using raw energy!')
                    self.energy_emb = None
                else:
                    self.energy_emb = nn.Linear(1, energy_linear_dim)
                self.use_energy_convs = True 
            else:
                print(f'''energy_agg_type = {energy_agg_type} does not exit.''')
            ## End energy args


        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        if(self.use_energy):
            self.energy_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=8)
        self.noise_convs = nn.ModuleList()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))
            
            if i + 1 < len(upsample_rates):  #
                stride_f0 = np.prod(upsample_rates[i + 1:])# Same for energy

                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
                
                if(self.use_energy):
                    if(energy_agg_type == 'all_step'):
                        self.energy_noise_convs.append(Conv1d(energy_linear_dim, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
                if(self.use_energy):
                    if(energy_agg_type == 'all_step'):
                        self.energy_noise_convs.append(Conv1d(energy_linear_dim, c_cur, kernel_size=1))


        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, f0, energy, g=None):

        if(self.use_energy):
        # print(x.shape, f0.shape,energy.shape)
            energy = torch.clamp(energy, min=0)

        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,L_upsampled,1

        if(self.use_energy):
            if(self.use_energy_convs):
                if(self.energy_linear_dim == 1):
                    energy = self.energy_upsamp(energy[:, None]) # bs,1, L_upsampled  
                else:
                    energy = self.energy_emb(energy.unsqueeze(-1)) # bs,L,linear_dim
                    energy = self.energy_upsamp(energy.transpose(1, 2)) # bs, linear_dim, L_upsampled

            else:
                energy = self.energy_emb(energy)


        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2) # bs, 1, L_upsampled

        x = self.conv_pre(x)

        if(self.use_energy):
            if(self.use_energy_convs):
                if g is not None:
                    x = x + self.cond(g)      
                else:
                    x = x
            else:
                if g is not None:
                    x = x + self.cond(g) + energy.transpose(1,2)  
                else:
                    x = x + energy.transpose(1,2)
        else:
            if g is not None:
                x = x + self.cond(g)      
            else:
                x = x
            # x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)


            x_source = self.noise_convs[i](har_source)

            if(self.use_energy):
                if(self.use_energy_convs):
                    x_energy = self.energy_noise_convs[i](energy)

                    # print(x.shape, x_source.shape, x_energy.shape)
                    x = x + x_source + x_energy
                else:
                    x = x + x_source
            else:
                x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        
        
class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames-partial_frames, partial_hop):
            mel_range = torch.arange(i, i+partial_frames)
            mel_slices.append(mel_range)
            
        return mel_slices
    
    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:,-partial_frames:]
        
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:,s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
        
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            #embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        
        return embed


class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    gin_channels,
    ssl_dim,
    use_spk,
    sampling_rate=44100,
    use_local_max = False,
    energy_use_log = False,
    energy_agg_type = 'one_step', 
    energy_linear_dim = 32,
    use_energy = True,
    energy_type = 'linear',
    energy_max = 500,
    **kwargs):

    super().__init__()

    self.energy_max = energy_max
    self.use_energy = use_energy
    self.energy_type = energy_type
    self.use_local_max = use_local_max
    self.energy_use_log = energy_use_log
    self.energy_agg_type = energy_agg_type
    self.energy_linear_dim = energy_linear_dim
    print(f'''Running energy embedding:
          \tusing energy = {self.use_energy}
          \tusing_local_max = {self.use_local_max}
          \tenergy_agg_type = {self.energy_agg_type}
          \tenergy_linear_dim = {self.energy_linear_dim}
          \tenergy_type = {self.energy_type}''')


    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.ssl_dim = ssl_dim
    self.use_spk = use_spk

    # self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16)

    self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

    if(self.use_energy):
        self.enc_p = ProsodyEncoder_energy(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
    else:
        self.enc_p = ProsodyEncoder(
        inter_channels,
        hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout
    )


    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, 
                         upsample_rates, upsample_initial_channel, upsample_kernel_sizes, 
                         gin_channels=gin_channels, sampling_rate = sampling_rate, 
                         energy_agg_type=energy_agg_type,energy_linear_dim=energy_linear_dim, 
                         use_energy = self.use_energy)
    
    self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels) 
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    
    if self.use_spk:
      self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels, model_embedding_size=gin_channels)

    self.emb_uv = nn.Embedding(2, hidden_channels)
    
  def forward(self, c, f0, uv, spec, energy =None, g=None, mel = None, c_lengths=None, spec_lengths=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    if spec_lengths == None:
      spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)

    x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)  
    
    if self.use_spk:
      g = self.enc_spk(mel.transpose(1,2))
      g = g.unsqueeze(-1)
    
    x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1,2)

    # encoder
    if(self.use_energy):
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), energy = energy_to_coarse(energy, self.use_local_max, energy_max = self.energy_max))
    else:
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))

    # _, m_p, logs_p, _ = self.enc_p(c, c_lengths)

    z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g) 
    z_p = self.flow(z, spec_mask, g=g)

    # print(z.shape, f0.shape, energy.shape, self.segment_size)
    z_slice, pitch_slice, energy_slice, ids_slice = commons.rand_slice_segments_with_pitch_and_energy(z, f0, energy, spec_lengths, self.segment_size)
    # print(z_slice.shape, pitch_slice.shape, energy_slice.shape, self.segment_size)
    if(self.energy_use_log):
        energy_ = torch.log10(energy_slice) #default by apple paper https://arxiv.org/pdf/2009.06775.pdf
    
    if(self.energy_type == 'quantized'):
        energy_ = energy_to_coarse(energy_slice, self.use_local_max, energy_max = self.energy_max)

    o = self.dec(z_slice, g=g, f0=pitch_slice, energy = energy_)
    
    return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

  def infer(self, c, f0, uv, energy = None, g=None, mel = None, noice_scale=0.35):
    c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
    x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1,2)

    if(self.use_energy):
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), energy = energy_to_coarse(energy, self.use_local_max, energy_max = self.energy_max), noice_scale=noice_scale)
    else:
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)

    z = self.flow(z_p, c_mask, g=g, reverse=True)

    if(self.energy_use_log):
        energy_ = torch.log10(energy) #default by apple paper https://arxiv.org/pdf/2009.06775.pdf
    
    if(self.energy_type == 'quantized'):
        energy_ = energy_to_coarse(energy, self.use_local_max, energy_max = self.energy_max)

    # if not self.use_spk:
    #   g = self.enc_spk.embed_utterance(mel.transpose(1,2))
    # g = g.unsqueeze(-1)

    # z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
    # z = self.flow(z_p, c_mask, g=g, reverse=True)
    o = self.dec(z * c_mask, g=g, f0=f0, energy=energy_)
    
    return o
