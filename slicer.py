import librosa
import torch
import torchaudio


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    #실행시간 측정
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = librosa.to_mono(waveform)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
        rms_list = librosa.feature.rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):

            if rms < self.threshold:
                # silent frames의 시작을 기록
                if silence_start is None:
                    silence_start = i
                continue
            # frame이 무음이 아닌 경우 및 무음 시작이 기록되지 않은 경우 계속 반복
            if silence_start is None:
                continue
            # 만약 간격이 충분하지 않거나 클립이 너무 짧다면 기록된 무음 시작을 지움
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # 제거될 무음 프레임의 범위를 기록
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # 음원 뒷부분 정적 삭제
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # 음원조각 적용 및 반환
        if len(sil_tags) == 0:
            return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
        else:
            chunks = []
            # 첫 번째 조용한 섹션이 시작 부분에서 시작되지 않는 경우, 소리가 있는 섹션을 보완
            if sil_tags[0][0]:
                chunks.append(
                    {"slice": False, "split_time": f"0,{min(waveform.shape[0], sil_tags[0][0] * self.hop_size)}"})
            for i in range(0, len(sil_tags)):
                # 소리가 있는 섹션을 식별(첫 번째는 건너뜀)
                if i:
                    chunks.append({"slice": False,
                                   "split_time": f"{sil_tags[i - 1][1] * self.hop_size},{min(waveform.shape[0], sil_tags[i][0] * self.hop_size)}"})
                # 모든 조용한 섹션을 식별
                chunks.append({"slice": True,
                               "split_time": f"{sil_tags[i][0] * self.hop_size},{min(waveform.shape[0], sil_tags[i][1] * self.hop_size)}"})
            # 마지막 조용한 섹션이 끝이 아니라면 마지막 조용한 섹션을 보완
            if sil_tags[-1][1] * self.hop_size < len(waveform):
                chunks.append({"slice": False, "split_time": f"{sil_tags[-1][1] * self.hop_size},{len(waveform)}"})
            chunk_dict = {}
            for i in range(len(chunks)):
                chunk_dict[str(i)] = chunks[i]
            return chunk_dict


def cut(audio_path, db_thresh=-30, min_len=5000, flask_mode=False, flask_sr=None):
    if not flask_mode:
        audio, sr = librosa.load(audio_path, sr=None)
    else:
        audio = audio_path
        sr = flask_sr
    slicer = Slicer(
        sr=sr,
        threshold=db_thresh,
        min_length=min_len
    )
    chunks = slicer.slice(audio)
    return chunks


def chunks2audio(audio_path, chunks):
    chunks = dict(chunks)
    audio, sr = torchaudio.load(audio_path)
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = audio.cpu().numpy()[0]
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            result.append((v["slice"], audio[int(tag[0]):int(tag[1])]))
    return result, sr