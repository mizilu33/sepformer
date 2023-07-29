import os
import time
import torch
from src.pit_criterion import cal_loss_pit, cal_loss_no, MixerMSE
from torch.utils.tensorboard import SummaryWriter
import gc

from const import CUDA_ID

import wandb

# # cuda.outofmemory
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,0,1,2"

import torch.distributed as dist
class Trainer(object):
    def __init__(self, data, model, optimizer, config):

        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = config["train"]["use_cuda"]  # 是否使用 GPU
        self.epochs = config["train"]["epochs"]  # 训练批次
        self.half_lr = config["train"]["half_lr"]  # 是否调整学习率
        self.early_stop = config["train"]["early_stop"]  # 是否早停
        self.max_norm = config["train"]["max_norm"]  # L2 范数
        self.batch_size=config["validation_loader"]["batch_size"]

        # save and load model
        self.save_folder = config["save_load"]["save_folder"]  # 模型保存路径
        self.checkpoint = config["save_load"]["checkpoint"]  # 是否保存每一个训练模型
        self.continue_from = config["save_load"]["continue_from"]  # 是否接着原来训练进度进行
        self.model_path = config["save_load"]["model_path"]  # 模型保存格式

        # logging
        self.print_freq = config["logging"]["print_freq"]

        # loss
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        # 生成保存模型的文件夹
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_improve = 0

        # 可视化
        self.write = SummaryWriter("./lungsound/knowledge/sepformer/logs")
        # self.log = wandb_logger
        # self.log.config.update(dict(epoch=self.epochs, lr=self.half_lr, batch_size=self.batch_size))

        self._reset()

        self.MixerMSE = MixerMSE()

        # # wandb
        # self.wandb_logger = wandb.init(
        #     project= "sepformer-ls",
        #     name="test",
        #     config=config
        # )

    def _reset(self):
        if self.continue_from:
            # 接着原来进度训练
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.save_folder + self.continue_from)

            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
            self.model.load_state_dict(package['state_dict'])
            self.model = torch.nn.DataParallel(self.model)
            self.optimizer.load_state_dict(package['optim_dict'])

            self.start_epoch = int(package.get('epoch', 1))

            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            # 重新训练
            self.start_epoch = 0

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()  # 将模型设置为训练模式

            start_time = time.time()  # 训练起始时间

            tr_loss = self._run_one_epoch(epoch)  # 训练模型

            gc.collect()
            torch.cuda.empty_cache()

            self.write.add_scalar("train loss", tr_loss, epoch+1)

            end_time = time.time()  # 训练结束时间
            run_time = end_time - start_time  # 训练时间

            print('-' * 85)
            print('End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(epoch+1, run_time, tr_loss))
            print('-' * 85)
            # self.wandb_logger.log('train_loss', tr_loss, on_step=True, on_epoch=True, prog_bar=True,logger=True)
            # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
            #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
            # 2. 我只需要在进程0上保存一次就行了，避免多次保存重复的东西
            if self.checkpoint and dist.get_rank() == 0:
                # 保存每一个训练模型
                # 这里没考虑不是dist分布式训练的情况, 之后改
                file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.module.serialize(self.model.module,
                                                self.optimizer,
                                                epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss), file_path)
                print('Saving checkpoint model to %s' % file_path)

            print('Cross validation Start...')

            self.model.eval()  # 将模型设置为验证模式

            start_time = time.time()  # 验证开始时间

            with torch.no_grad():
                val_loss = self._run_one_epoch(epoch, cross_valid=True)  # 验证模型

            self.write.add_scalar("validation loss", val_loss, epoch+1)

            
            self.write.close()

            end_time = time.time()  # 验证结束时间
            run_time = end_time - start_time  # 训练时间

            print('-' * 85)
            print('End of Epoch {0} | Time {1:.2f}s | ''Valid Loss {2:.3f}'.format(epoch+1, run_time, val_loss))
            print('-' * 85)
            # self.wandb_logger.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True,logger=True)

            # 是否调整学习率
            if self.half_lr:
                # 验证损失是否提升
                if val_loss >= self.prev_val_loss:
                    self.val_no_improve += 1  # 统计没有提升的次数

                    # 如果训练 3 个 epoch 没有提升，学习率减半
                    if self.val_no_improve >= 3:
                        self.halving = True

                    # 如果训练 10 个 epoch 没有提升, 结束训练
                    if self.val_no_improve >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_improve = 0

            if self.halving:
                optime_state = self.optimizer.state_dict()
                optime_state['param_groups'][0]['lr'] = optime_state['param_groups'][0]['lr']/2.0
                self.optimizer.load_state_dict(optime_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(lr=optime_state['param_groups'][0]['lr']))
                self.halving = False

            self.prev_val_loss = val_loss  # 当前损失

            self.tr_loss[epoch] = tr_loss
            self.cv_loss[epoch] = val_loss

            # 保存最好的模型
            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss  # 最小的验证损失值

                file_path = os.path.join(self.save_folder, self.model_path)

                if isinstance(self.model, torch.nn.DataParallel):
                    torch.save(self.model.module.serialize(self.model.module,
                                                    self.optimizer,
                                                    epoch + 1,
                                                    tr_loss=self.tr_loss,
                                                    cv_loss=self.cv_loss), file_path)
                else:
                    torch.save(self.model.serialize(self.model,
                                self.optimizer,
                                epoch + 1,
                                tr_loss=self.tr_loss,
                                cv_loss=self.cv_loss), file_path)
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start_time = time.time()

        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader  # 数据集切换
        # 新增2：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
        data_loader.sampler.set_epoch(epoch)
        for i, (data) in enumerate(data_loader):

            padded_mixture, mixture_lengths, padded_source, s1_path = data
            # print("padded_mixture, mixture_lengths, padded_source:", padded_mixture, mixture_lengths, padded_source)
            # print("padded_mixture.shape", padded_mixture.shape)
            # print("mixture_lengths.shape", mixture_lengths.shape)
            # print("padded_source.shape", padded_source.shape)

            # return xs_pad, ilens, ys_pad
            # 是否使用 GPU 训练
            if torch.cuda.is_available():
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            # padded_mixture = padded_mixture.data.cpu().numpy()
            # mixture_lengths = mixture_lengths.data.cpu().numpy()
            # padded_source = padded_source.data.cpu().numpy()
            

            estimate_source = self.model(padded_mixture)  # 将数据放入模型

            loss, max_snr, estimate_source, reorder_estimate_source = cal_loss_pit(padded_source,
                                                                                   estimate_source,
                                                                                   mixture_lengths)

            # loss, max_snr, estimate_source, reorder_estimate_source = cal_loss_no(padded_source,
            #                                                                       estimate_source,
            #                                                                       mixture_lengths)

            # loss = self.MixerMSE(estimate_source, padded_source)

            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            end_time = time.time()
            run_time = end_time - start_time

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | Current Loss {3:.6f} | {4:.1f} s/batch'.format(
                    epoch+1,
                    i+1,
                    total_loss/(i+1),
                    loss.item(),
                    run_time/(i+1)),
                    flush=True)

            if loss.item() > total_loss/(i+1):
                # print(s1_path)
                # file_name_s1 = os.path.split(s1_path)[1]
                # print(file_name_s1)
                print("s1_path:{0}--loss:{1:3f}".format(s1_path, loss.item()))

        return total_loss/(i+1)
