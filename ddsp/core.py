import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import numpy as np

def MaskedAvgPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    ones_kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)

    # sum 풀링 수행
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )

    # 각 풀링 창에서 마스킹된(유효한) 요소 수 계산
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )
    valid_count = valid_count.clamp(min=1)  # 0으로 나누는 것을 피하기 위해

    # 마스킹된 평균 풀링 수행
    avg_pooled = sum_pooled / valid_count

    return avg_pooled.squeeze(1)

def MedianPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    x = x.squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]
    
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True):
  """효율적인 FFT의 최종 크기 계산.
  Args:
    frame_size: 오디오 프레임의 크기.
    ir_size: 합성 임펄스 응답의 크기.
    power_of_2: 2의 거듭제곱으로 제한할지 여부. False이면 다른 5-smooth 숫자를 허용.
      TPU는 2의 거듭제곱이 필요하지만 GPU는 더 유연함.
  Returns:
    fft_size: 효율적인 FFT 크기.
  """
  convolved_frame_size = ir_size + frame_size - 1
  if power_of_2:
    # 다음 2의 거듭제곱.
    fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
  else:
    fft_size = convolved_frame_size
  return fft_size


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal,signal[:,:,-1:]),2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:,:,:-1]
    return signal.permute(0, 2, 1)


def remove_above_fmax(amplitudes, pitch, fmax, level_start=1):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(level_start, n_harm + level_start).to(pitch)
    aa = (pitches < fmax).float() + 1e-7
    return amplitudes * aa


def crop_and_compensate_delay(audio, audio_size, ir_size,
                              padding = 'same',
                              delay_compensation = -1):
  """그룹 지연을 보상하기 위해 합성 결과 오디오를 자르기.
  Args:
    audio: 컨벌루션 후 오디오. Tensor of shape [batch, time_steps].
    audio_size: 컨벌루션 이전의 오디오 초기 크기.
    ir_size: 컨벌루션 임펄스 응답의 크기.
    padding: 'valid' 또는 'same'. 'same'은 최종 출력을 입력 오디오와 동일한 크기로 설정합니다.
      'valid'는 오디오를 연장하여 임펄스 응답의 꼬리를 포함합니다 (audio_timesteps + ir_timesteps - 1).
    delay_compensation: 임펄스 응답의 그룹 지연을 보상하기 위해 출력 오디오의 시작부터 잘라낼 샘플 수입니다.
      delay_compensation < 0이면 frequency_impulse_response()에서 윈도우화된 선형 위상 필터의 일정한 그룹 지연을 자동으로 계산합니다.
  Returns:
    잘린 오디오의 Tensor.
  Raises:
    ValueError: padding이 'valid' 또는 'same'이 아닌 경우.
  """
  # 출력 자르기.
  if padding == 'valid':
    crop_size = ir_size + audio_size - 1
  elif padding == 'same':
    crop_size = audio_size
  else:
    raise ValueError('패딩은 \'valid\' 또는 \'same\'이어야 합니다. 대신 {}입니다.'
                     .format(padding))

  # 필터의 그룹 지연을 보상하기 위해 앞부분을 잘라냅니다.
  # frequency_impulse_response()에서 생성된 임펄스 응답의 경우 필터가 선형 위상입니다.
  total_size = int(audio.shape[-1])
  crop = total_size - crop_size
  start = (ir_size // 2 if delay_compensation < 0 else delay_compensation)
  end = crop - start
  return audio[:, start:-end]


def fft_convolve(audio,
                 impulse_response): # B, n_frames, 2*(n_mags-1)
    """시간 변화하는 임펄스 응답 프레임으로 오디오 필터링.
    시간에 따라 변하는 필터. 주어진 오디오 [batch, n_samples]와 일련의 임펄스
    응답 [batch, n_frames, n_impulse_response]을 받아, 오디오를 프레임으로 나누고,
    필터를 적용한 다음, 오디오를 겹쳐서 다시 합칩니다.
    큰 임펄스 응답 크기에 대한 효율적인 컨볼루션을 계산하기 위해 비윈도우, 비겹치는 STFT/ISTFT 적용.
    Args:
        audio: 입력 오디오. Tensor of shape [batch, audio_timesteps].
        impulse_response: 컨볼루션할 유한 임펄스 응답. 2-D Tensor of shape [batch, ir_size]
        또는 3-D Tensor of shape [batch, ir_frames, ir_size]일 수 있습니다. 2-D 텐서는 오디오에
        단일 선형 시불변 필터를 적용합니다. 3-D 텐서는 선형 시간 변화 필터를 적용합니다.
        오디오를 일치하도록 동일한 모양의 블록으로 잘라낼 수 있습니다.
    Returns:
        오디오 아웃: 컨볼루션된 오디오. Tensor of shape [batch, audio_timesteps].
    """
    # 임펄스 응답에 프레임 차원 추가 (있는 경우).
    ir_shape = impulse_response.size() 
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)
        ir_shape = impulse_response.size()

    # 오디오와 임펄스 응답의 모양 가져오기.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.size() # B, T

    # 배치 크기가 일치하는지 확인합니다.
    if batch_size != batch_size_ir:
        raise ValueError('오디오 ({})와 임펄스 응답 ({})의 배치 크기가 동일해야 합니다.'
                        .format(batch_size, batch_size_ir))

    # 오디오를 50% 겹쳐진 프레임(가운데 패딩)으로 자릅니다.
    hop_size = int(audio_size / n_ir_frames)
    frame_size = 2 * hop_size    
    audio_frames = F.pad(audio, (hop_size, hop_size)).unfold(1, frame_size, hop_size)
    
    # Bartlett(삼각형) 창 적용
    window = torch.bartlett_window(frame_size).to(audio_frames)
    audio_frames = audio_frames * window
    
    # 오디오와 임펄스 응답을 패딩하고 FFT합니다.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=False)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(torch.cat((impulse_response,impulse_response[:,-1:,:]),1), fft_size)
    
    # FFT를 곱합니다 (시간에 따른 컨볼루션).
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    # 오디오를 재합성하기 위해 IFFT를 취합니다.
    audio_frames_out = torch.fft.irfft(audio_ir_fft, fft_size)
    
    # 겹쳐서 추가합니다
    batch_size, n_audio_frames, frame_size = audio_frames_out.size() # # B, n_frames+1, 2*(hop_size+n_mags-1)-1
    fold = torch.nn.Fold(output_size=(1, (n_audio_frames - 1) * hop_size + frame_size),kernel_size=(1, frame_size),stride=(1, hop_size))
    output_signal = fold(audio_frames_out.transpose(1, 2)).squeeze(1).squeeze(1)
    
    # 출력 오디오를 자르고 이동시킵니다.
    output_signal = crop_and_compensate_delay(output_signal[:,hop_size:], audio_size, ir_size)
    return output_signal
    

def apply_window_to_impulse_response(impulse_response, # B, n_frames, 2*(n_mag-1)
                                     window_size: int = 0,
                                     causal: bool = False):
    """임펄스 응답에 window 적용 및 window 형태로 설정.
    Args:
        impulse_response: window을 적용할 일련의 임펄스 응답 프레임, 형태는 [batch, n_frames, ir_size]입니다.
        window_size: 시간 영역에 적용할 window 크기입니다. window_size가 1보다 작으면 임펄스 응답 크기로 기본 설정됩니다.
        causal: 임펄스 응답 입력이 casual form인지 여부입니다 (가운데에 피크).
    Returns:
        impulse_response: window가 적용된 임펄스 응답, 마지막 차원이 window_size보다 큰 경우에만 자르기.
    """
    
    # IR이 casual form인 경우, 제로 페이즈 형태로 변경합니다.
    if causal:
        impulse_response = torch.fftshift(impulse_response, axes=-1)
    
    # 더 나은 시간/주파수 해상도를 위한 window 가져오기.
    # window는 IR 크기로 기본 설정되며 더 크지 않습니다.
    ir_size = int(impulse_response.size(-1))
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = nn.Parameter(torch.hann_window(window_size), requires_grad = False).to(impulse_response)
    
    # window을 제로 패딩하고 제로 페이즈 형태로 변경합니다.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:],
                            torch.zeros([padding]),
                            window[:half_idx]], axis=0)
    else:
        window = window.roll(window.size(-1)//2, -1)
        
    # window 적용하여 새 IR 가져오기 (둘 다 제로 페이즈 형태).
    window = window.unsqueeze(0)
    impulse_response = impulse_response * window
    
    # IR을 casual form으로 변경하고 제로 패딩 자르기.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                    impulse_response[..., :second_half_end]],
                                    dim=-1)
    else:
        impulse_response = impulse_response.roll(impulse_response.size(-1)//2, -1)

    return impulse_response


def apply_dynamic_window_to_impulse_response(impulse_response,  # B, n_frames, 2*(n_mag-1) or 2*n_mag-1
                                             half_width_frames):        # B，n_frames, 1
    ir_size = int(impulse_response.size(-1)) # 2*(n_mag -1) or 2*n_mag-1
    
    window = torch.arange(-(ir_size // 2), (ir_size + 1) // 2).to(impulse_response) / half_width_frames 
    window[window > 1] = 0
    window = (1 + torch.cos(np.pi * window)) / 2 # B, n_frames, 2*(n_mag -1) or 2*n_mag-1
    
    impulse_response = impulse_response.roll(ir_size // 2, -1)
    impulse_response = impulse_response * window
    
    return impulse_response
    
        
def frequency_impulse_response(magnitudes,
                               hann_window = True,
                               half_width_frames = None):
                               
    # IR 가져오기
    impulse_response = torch.fft.irfft(magnitudes) # B, n_frames, 2*(n_mags-1)
    
    # window 적용하고 casual form으로 변경.
    if hann_window:
        if half_width_frames is None:
            impulse_response = apply_window_to_impulse_response(impulse_response)
        else:
            impulse_response = apply_dynamic_window_to_impulse_response(impulse_response, half_width_frames)
    else:
        impulse_response = impulse_response.roll(impulse_response.size(-1) // 2, -1)
       
    return impulse_response


def frequency_filter(audio,
                     magnitudes,
                     hann_window=True,
                     half_width_frames=None):

    impulse_response = frequency_impulse_response(magnitudes, hann_window, half_width_frames)
    
    return fft_convolve(audio, impulse_response)
