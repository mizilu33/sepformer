import argparse
import torch
# from dataset.data import AudioDataLoader, AudioDataset
from dataset.data_segment import AudioDataset
# from dataset.data import AudioDataset

from torch.utils.data import Dataset, DataLoader
import os

from src.trainer import Trainer
from model.sepformer import Sepformer
import json5
import numpy as np
from adamp import AdamP, SGDP

from const import CUDA_ID, MAIN_DIR, LUNGSOUND_DIR

import wandb
from random import randint

os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"


'''
dataloader  load出来的东西就是重写的collate_fn函数最后return出来的
这里传入的data, 应该是[[data1, label1], [data2, label2], ...]
这里需要返回 data 和 label 以及二者的长度序列

————————————————————等待修改——————————————
'''
# mix_data, length, s_data, s1_path

def collate_fn(batch_data):
    # data = [i[0] for i in batch_dic]
    # label = [i[1] for i in batch_dic]

    batch_len = len(batch_data)

    mix_data = [i[0] for i in batch_data]
    length = [i[1] for i in batch_data]
    s1_data = [i[2] for i in batch_data]
    s2_data = [i[3] for i in batch_data]

    s1_path = [i[4] for i in batch_data]

    # mix_data_lengths = torch.LongTensor([len(x) for x in mix_data])
    # s_data_lengths = torch.LongTensor([len(x) for x in s_data])
  
    # data_lengths = torch.LongTensor([len(x) for x in data])
    # label_lengths = torch.LongTensor([len(x) for x in label])

    # s1_path = torch.nn.utils.rnn.pad_sequence(s1_path, batch_first=True, padding_value=-1)
    # label = torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=-1)

    mix_data = torch.nn.utils.rnn.pad_sequence(mix_data, batch_first=True, padding_value=0)

    # s_data_shape = np.array(s_data).shape
    # print(s_data_shape)
    # s_data = torch.nn.utils.rnn.pad_sequence(s_data, batch_first=True, padding_value=0)
    # pad_sequence接受targets 2层嵌套list,故不能pad_sequences 原s_data[2,4000],n 个batch则为[n, 2, 4000]

    s1_data = torch.nn.utils.rnn.pad_sequence(s1_data, batch_first=True, padding_value=0)
    s2_data = torch.nn.utils.rnn.pad_sequence(s2_data, batch_first=True, padding_value=0)

    s_data = torch.tensor(np.stack((s1_data, s2_data), axis=1))

    max_length = max(length)
    length = []
    for i in range(batch_len):
        length.append(max_length)

    length = torch.tensor(length)

    # return mix_data, length, s_data, s1_path, mix_data_lengths, s_data_lengths
    return mix_data, length, s_data, s1_path

    # return data, label, data_lengths, label_lengths
# 传入的参数是dataset返回的参数吗
# 返回的参数是所有传入参数的pad_sequence和和他们的长度序列吗
"""
def collate_func(batch_dic):
    from torch.nn.utils.rnn import pad_sequence
    batch_len=len(batch_dic)
    max_seq_length=max([dic['length'] for dic in batch_dic])
    mask_batch=torch.zeros((batch_len,max_seq_length))
    fea_batch=[]
    label_batch=[]
    id_batch=[]
    for i in range(len(batch_dic)):
        dic=batch_dic[i]
        fea_batch.append(dic['feature'])
        label_batch.append(dic['label'])
        id_batch.append(dic['id'])
        mask_batch[i,:dic['length']]=1
    res={}
    res['feature']=pad_sequence(fea_batch,batch_first=True)
    res['label']=pad_sequence(label_batch,batch_first=True)
    res['id']=id_batch
    res['mask']=mask_batch
    return res
"""



def main(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    # start_time = randint(0,10)
    # 数据
    tr_dataset = AudioDataset(json_dir=config["train_dataset"]["train_dir"],  # 目录下包含 mix.json, s1.json, s2.json
                              data_dir=config["train_dataset"]["data_dir"],
                              batch_size=config["train_dataset"]["batch_size"],
                              sample_rate=config["train_dataset"]["sample_rate"],  # 采样率
                              segment=config["train_dataset"]["segment"],
                            #   start_time=start_time
                              )  # 语音时长
                    
    cv_dataset = AudioDataset(json_dir=config["validation_dataset"]["validation_dir"],
                              data_dir=config["validation_dataset"]["data_dir"],
                              batch_size=config["validation_dataset"]["batch_size"],
                              sample_rate=config["validation_dataset"]["sample_rate"],
                              segment=config["validation_dataset"]["segment"],
                            #   start_time=start_time
                              ) # cv_max_len=config["validation_dataset"]["cv_max_len"])

    # tr_loader = AudioDataLoader(tr_dataset,
    #                             batch_size=config["train_loader"]["batch_size"],
    #                             shuffle=config["train_loader"]["shuffle"],
    #                             num_workers=config["train_loader"]["num_workers"])

    # cv_loader = AudioDataLoader(cv_dataset,
    #                             batch_size=config["validation_loader"]["batch_size"],
    #                             shuffle=config["validation_loader"]["shuffle"],
    #                             num_workers=config["validation_loader"]["num_workers"])

    tr_loader = DataLoader(tr_dataset,
                                batch_size=config["train_loader"]["batch_size"],
                                shuffle=config["train_loader"]["shuffle"],
                                num_workers=config["train_loader"]["num_workers"],
                                collate_fn=collate_fn)

    cv_loader = DataLoader(cv_dataset,
                                batch_size=config["validation_loader"]["batch_size"],
                                shuffle=config["validation_loader"]["shuffle"],
                                num_workers=config["validation_loader"]["num_workers"],
                                collate_fn=collate_fn)


    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    # # wandb
    # wandb_logger = wandb.init(
    #     project= "sepformer-ls",
    #     name="test",
    #     resume="allow"
    # )
    # wandb_logger.config.update(dict(lr=0.01, batch_size=1))

    # 模型
    if config["model"]["type"] == "sepformer":
        model = Sepformer(N=config["model"]["sepformer"]["N"],
                          C=config["model"]["sepformer"]["C"],
                          L=config["model"]["sepformer"]["L"],
                          H=config["model"]["sepformer"]["H"],
                          K=config["model"]["sepformer"]["K"],
                          Global_B=config["model"]["sepformer"]["Global_B"],
                          Local_B=config["model"]["sepformer"]["Local_B"])
    else:
        print("No loaded model!")

    if config["optimizer"]["type"] == "sgd":
        optimize = torch.optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"]["sgd"]["lr"],
            momentum=config["optimizer"]["sgd"]["momentum"],
            weight_decay=config["optimizer"]["sgd"]["l2"])
    elif config["optimizer"]["type"] == "adam":
        optimize = torch.optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["adam"]["lr"],
            betas=(config["optimizer"]["adam"]["beta1"], config["optimizer"]["adam"]["beta2"]))
    elif config["optimizer"]["type"] == "sgdp":
        optimize = SGDP(
            params=model.parameters(),
            lr=config["optimizer"]["sgdp"]["lr"],
            weight_decay=config["optimizer"]["sgdp"]["weight_decay"],
            momentum=config["optimizer"]["sgdp"]["momentum"],
            nesterov=config["optimizer"]["sgdp"]["nesterov"],
        )
    elif config["optimizer"]["type"] == "adamp":
        optimize = AdamP(
            params=model.parameters(),
            lr=config["optimizer"]["adamp"]["lr"],
            betas=(config["optimizer"]["adamp"]["beta1"], config["optimizer"]["adamp"]["beta2"]),
            weight_decay=config["optimizer"]["adamp"]["weight_decay"],
        )
    else:
        print("Not support optimizer")
        return
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0,1])   
        model.cuda()
        # optimize = torch.nn.DataParallel(optimize, device_ids=[0,1])
        # optimize.cuda()
    # wandb.watch(model, log="all")
    trainer = Trainer(data, model, optimize, config)

    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Speech Separation")

    parser.add_argument("-C",
                        "--configuration",
                        default=f"{MAIN_DIR}/config/train/train.json5",
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration['train_dataset']['data_dir'] = f'{MAIN_DIR}/{configuration["train_dataset"]["data_dir"]}'
    configuration['validation_dataset']['data_dir'] = f'{MAIN_DIR}/{configuration["validation_dataset"]["data_dir"]}'
    configuration["train_dataset"]["train_dir"] = f'{MAIN_DIR}/{configuration["train_dataset"]["train_dir"]}'
    configuration["validation_dataset"]["validation_dir"] = f'{MAIN_DIR}/{configuration["validation_dataset"]["validation_dir"]}'
    configuration["save_load"]["save_folder"] = f'{MAIN_DIR}/{configuration["save_load"]["save_folder"]}'

    print(configuration)

    main(configuration)
