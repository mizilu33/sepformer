

# 混合肺音和噪声生成mix文件夹
import os
from pydub import AudioSegment

def mix(s1_path, s2_path, out_path):

    # path = '/home/mzlu/data/lsdata/SPRSound/train_mix_wav'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # output_path = '/home/mzlu/data/lsdata/SPRSound/train_mix_wav'
    for file1, file2 in zip(os.listdir(s1_path),os.listdir(s2_path)):  #遍历两个文件路径下的文件
        path1 = s1_path + "/" + file1  # 拼接第一个输入路径和对应文件名
        path2 = s2_path + "/" + file2  # 拼接第一个输入路径和对应文件名
        path3 = out_path + "/" + file1 # 拼接输出路径和输出文件名，这里以第一个输入的文件名命名

        sound1 = AudioSegment.from_wav(path1)  # 第一个文件
        sound2 = AudioSegment.from_wav(path2)  # 第二个文件
        output = sound1.overlay(sound2)  # 把sound2叠加到sound1上面
        output.export(path3, format="wav")  # 保存文件

if __name__ == "__main__":

    root_dir = '/home/mzlu/lungsound/data/HF_Lung_V1'

    for data_type in ['tr', 'cv', 'tt']:
        s1_path = root_dir + '/' + data_type + '/s1'
        s2_path = root_dir + '/' + data_type + '/s2'
        out_path = root_dir + '/' + data_type + '/mix'
        mix(s1_path, s2_path, out_path)
