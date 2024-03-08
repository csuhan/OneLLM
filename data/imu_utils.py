import string
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import json
from collections import defaultdict
from bisect import bisect_left
import os
import torch
import torchaudio
torchaudio.set_audio_backend("sox_io")


def load_json(json_path: str):
    """
    Load a json file
    """
    with open(json_path, "r", encoding="utf-8") as f_name:
        data = json.load(f_name)
    return data


def check_window_signal(info_t, w_s, w_e):
    length = w_e - w_s
    frame_offset = int(w_s * info_t.sample_rate)
    num_frames = int(length * info_t.sample_rate)
    if frame_offset + num_frames > int(info_t.num_frames):
        return False
    else:
        return True


def index_narrations(ann_path):
    narration_raw = load_json(ann_path)

    narration_dict = defaultdict(list)
    summary_dict = defaultdict(list)
    avg_len = []
    for v_id, narr in narration_raw.items():
        narr_list = []
        summ_list = []
        if "narration_pass_1" in narr:
            narr_list += narr["narration_pass_1"]["narrations"]
            summ_list += narr["narration_pass_1"]["summaries"]
        if "narration_pass_2" in narr:
            narr_list += narr["narration_pass_2"]["narrations"]
            summ_list += narr["narration_pass_2"]["summaries"]

        if len(narr_list) > 0:
            narration_dict[v_id] = [
                (
                    float(n_t["timestamp_sec"]),
                    n_t["narration_text"],
                    n_t["annotation_uid"],
                    n_t["timestamp_frame"],
                )
                for n_t in narr_list
            ]
            avg_len.append(len(narration_dict[v_id]))
        else:
            narration_dict[v_id] = []
        if len(summ_list) > 0:
            summary_dict[v_id] = [
                (
                    float(s_t["start_sec"]),
                    float(s_t["end_sec"]),
                    s_t["summary_text"],
                )
                for s_t in summ_list
            ]
        else:
            summary_dict[v_id] = []
    # print(f"Number of Videos with narration {len(narration_dict)}")
    # print(f"Avg. narration length {np.mean(avg_len)}")
    # print(f"Number of Videos with summaries {len(summary_dict)}")
    return narration_dict, summary_dict


def get_signal_info(signal_fn: str):
    return torchaudio.info(signal_fn)


def get_signal_frames(signal_fn: str, video_start_sec: float, video_end_sec: float):
    """
    Given a signal track return the frames between video_start_sec and video_end_sec
    """
    info_t = get_signal_info(signal_fn)

    length = video_end_sec - video_start_sec
    aframes, _ = torchaudio.load(
        signal_fn,
        normalize=True,
        frame_offset=int(video_start_sec * info_t.sample_rate),
        num_frames=int(length * info_t.sample_rate),
    )
    return {"signal": aframes, "meta": info_t}


def tosec(value):
    return value / 1000


def toms(value):
    return value * 1000


def delta(first_num: float, second_num: float):
    """Compute the absolute value of the difference of two numbers"""
    return abs(first_num - second_num)


def padIMU(signal, duration_sec):
    """
    Pad the signal if necessary
    """
    expected_elements = round(duration_sec) * 200

    if signal.shape[0] > expected_elements:
        signal = signal[:expected_elements, :]
    elif signal.shape[0] < expected_elements:
        padding = expected_elements - signal.shape[0]
        padded_zeros = np.zeros((padding, 6))
        signal = np.concatenate([signal, padded_zeros], 0)
        # signal = signal[:expected_elements, :]
    return signal


def resample(
    signals: np.ndarray,
    timestamps: np.ndarray,
    original_sample_rate: int,
    resample_rate: int,
):
    """
    Resamples data to new sample rate
    """
    signals = torch.as_tensor(signals)
    timestamps = torch.from_numpy(timestamps).unsqueeze(-1)
    signals = torchaudio.functional.resample(
        waveform=signals.data.T,
        orig_freq=original_sample_rate,
        new_freq=resample_rate,
    ).T.numpy()

    nsamples = len(signals)

    period = 1 / resample_rate

    # timestamps are expected to be shape (N, 1)
    initital_seconds = timestamps[0] / 1e3

    ntimes = (torch.arange(nsamples) * period).view(-1, 1) + initital_seconds

    timestamps = (ntimes * 1e3).squeeze().numpy()
    return signals, timestamps


def resampleIMU(signal, timestamps):
    sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
    # resample all to 200hz
    if sampling_rate != 200:
        signal, timestamps = resample(signal, timestamps, sampling_rate, 200)
    return signal, timestamps


def get_imu_frames(
    imu_path,
    uid: str,
    video_start_sec: float,
    video_end_sec: float,
):
    """
    Given a IMU signal return the frames between video_start_sec and video_end_sec
    """
    signal = np.load(os.path.join(imu_path, f"{uid}.npy"))
    signal = signal.transpose()
    timestamps = np.load(os.path.join(imu_path, f"{uid}_timestamps.npy"))

    if toms(video_start_sec) > timestamps[-1] or toms(video_end_sec) > timestamps[-1]:
        return None

    start_id = bisect_left(timestamps, toms(video_start_sec))
    end_id = bisect_left(timestamps, toms(video_end_sec))

    # make sure the retrieved window interval are correct by a max of 1 sec margin
    if (
        delta(video_start_sec, tosec(timestamps[start_id])) > 4
        or delta(video_end_sec, tosec(timestamps[end_id])) > 4
    ):
        return None

    # get the window
    if start_id == end_id:
        start_id -= 1
        end_id += 1
    signal, timestamps = signal[start_id:end_id], timestamps[start_id:end_id]

    if len(signal) < 10 or len(timestamps) < 10:
        return None
    # resample the signal at 200hz if necessary
    signal, timestamps = resampleIMU(signal, timestamps)

    # pad  the signal if necessary
    signal = padIMU(signal, video_end_sec - video_start_sec)

    sample_dict = {
        "timestamp": timestamps,
        "signal": torch.tensor(signal.T),
        "sampling_rate": 200,
    }

    return sample_dict


def display_animation(frames, title, save_path_gif):
    fig, ax = plt.subplots()
    frames = [[ax.imshow(frames[i])] for i in range(len(frames))]
    plt.title(title)
    ani = animation.ArtistAnimation(fig, frames)
    ani.save(save_path_gif, writer="imagemagick")
    plt.close()


def display_animation_imu(frames, imu, title, save_path_gif):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_title(title)
    ax2.set_title("Acc.")
    ax3.set_title("Gyro.")
    frames = [[ax1.imshow(frames[i])] for i in range(len(frames))]
    ani = animation.ArtistAnimation(fig, frames)

    ax2.plot(imu[0].cpu().numpy(), color="red")
    ax2.plot(imu[1].cpu().numpy(), color="blue")
    ax2.plot(imu[2].cpu().numpy(), color="green")
    ax3.plot(imu[3].cpu().numpy(), color="red")
    ax3.plot(imu[4].cpu().numpy(), color="blue")
    ax3.plot(imu[5].cpu().numpy(), color="green")
    plt.tight_layout()
    ani.save(save_path_gif, writer="imagemagick")
    plt.close()


def filter_narration(narration_text: str) -> bool:
    if "#c" in narration_text.lower():
        return True
    return False


def clean_narration_text(narration_text: str) -> str:
    return (
        narration_text.replace("#C C ", "")
        .replace("#C", "")
        .replace("#unsure", "something")
        .strip()
        .strip(string.punctuation)
        .lower()[:128]
    )
