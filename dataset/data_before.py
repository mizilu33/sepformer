
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np


class AudioDataset(Dataset):

    def __init__(self, data_dir, json_dir, batch_size, sample_rate, segment):

        super(AudioDataset, self).__init__()

        self.data_dir = data_dir
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.segment = segment
        # self.cv_max_len = cv_max_len

        file = ["mix", "s1", "s2"]

        self.mix_dir = os.path.join(data_dir, file[0])
        self.mix_list = os.listdir(os.path.abspath(self.mix_dir))
        # self.json_dir = os.path.join(data_dir, file[0])

        self.s1_dir = os.path.join(data_dir, file[1])
        self.s1_list = os.listdir(os.path.abspath(self.s1_dir))
        # self.json_dir = os.path.join(data_dir, file[1])

        self.s2_dir = os.path.join(data_dir, file[2])
        self.s2_list = os.listdir(os.path.abspath(self.s2_dir))

    def __getitem__(self, item):

        mix_path = os.path.join(self.mix_dir, self.mix_list[item])
        mix_data = librosa.load(path=mix_path,
                                sr=self.sample_rate,
                                mono=True,  # 单通道
                                offset=0,  # 音频读取起始点
                                duration=None,  # 获取音频时长
                                dtype=np.float32,
                                res_type="kaiser_best",
                                )[0]
        length = len(mix_data)

        s1_path = os.path.join(self.s1_dir, self.s1_list[item])
        s1_data = librosa.load(path=s1_path,
                               sr=self.sample_rate,
                               mono=True,  # 单通道
                               offset=0,  # 音频读取起始点
                               duration=None,  # 获取音频时长
                               )[0]

        s2_path = os.path.join(self.s2_dir, self.s2_list[item])
        s2_data = librosa.load(path=s2_path,
                               sr=self.sample_rate,
                               mono=True,  # 单通道
                               offset=0,  # 音频读取起始点
                               duration=None,  # 获取音频时长
                               )[0]

        s_data = np.stack((s1_data, s2_data), axis=0)

        return mix_data, length, s_data

    def __len__(self):

        return len(self.mix_list)


if __name__ == "__main__":

    dataset = AudioDataset(data_dir="/home/mzlu/lungsound/data/HF_Lung_V1/tr",
                        sample_rate=8000)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             drop_last=True)

    for (i, data) in enumerate(data_loader):

        if i >= 1:
            break

        mix, length, s = data
        print(mix.shape, length, s.shape)
