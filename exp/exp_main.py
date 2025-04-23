from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visualize_data
from utils.metrics import metric, MSE, MSE_NEW
from torch import Tensor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        print("self.args ",self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        #from tsdm.tasks.mimic_iii_debrouwer2019 import mimic_collate as task_collate_fn
        #data_set, data_loader = data_provider(self.args, flag)
        from tsdm.tasks.physionet2012 import physionet_collate as task_collate_fn


        if self.args.data == "mimiciii":
            from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

            TASK = MIMIC_III_DeBrouwer2019(
                condition_time=self.args.observation_time, forecast_horizon=self.args.forecast_horizon
            )
            INPUT_DIM = 96

        if self.args.data == "mimiciv":
            from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

            TASK = MIMIC_IV_Bilos2021(
                condition_time=self.args.observation_time, forecast_horizon=self.args.forecast_horizon
            )
            INPUT_DIM = 102

        if self.args.data == "p12":
            print("In p12")
            from tsdm.tasks.physionet2012 import Physionet2012

            TASK = Physionet2012(
                condition_time=self.args.observation_time, forecast_horizon=self.args.forecast_horizon
            )
            INPUT_DIM = 37
            print("data downloaded")

        if self.args.data == "ushcn":
            from tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019

            TASK = USHCN_DeBrouwer2019(
                condition_time=self.args.observation_time, forecast_horizon=self.args.forecast_horizon
            )
            INPUT_DIM = 5

        dloader_config_train = {
            "batch_size": self.args.batch_size,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": task_collate_fn,
        }

        dloader_config_infer = {
            "batch_size": 64,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": task_collate_fn,
        }

        #TRAIN_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_train)
        #INFER_LOADER = TASK.get_dataloader((ARGS.fold, "train"), **dloader_config_infer)
        #VALID_LOADER = TASK.get_dataloader((ARGS.fold, "valid"), **dloader_config_infer)
        #TEST_LOADER = TASK.get_dataloader((ARGS.fold, "test"), **dloader_config_infer)
        #EVAL_LOADERS = {"train": INFER_LOADER, "valid": VALID_LOADER, "test": TEST_LOADER}

        print("flag ",flag)
        data_loader = TASK.get_dataloader((self.args.fold, flag), **dloader_config_train)
        print("data_loader ",data_loader)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def MSE(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
        err = torch.sum(mask * ((y - yhat) ** 2)) / torch.sum(mask)
        return err



    def vali(self,  vali_loader, criterion):
        print("Validation started")
        print("****************************")
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (x_time,batch_x, x_mask,y_time, batch_y,y_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.forecast_horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.forecast_horizon:, f_dim:]
                batch_y = batch_y[:, -self.args.forecast_horizon:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        print("Validation done")
        return total_loss

    def train(self, setting):
        print("*******Training started********")
        train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_loader = self._get_data(flag='valid')
            test_loader = self._get_data(flag='test')
        print("After data is downloaded")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        batch = next(iter(train_loader))
        print("batch ",batch)
        torch.set_printoptions(threshold=float('inf'), linewidth=200)
        train_steps = len(train_loader)
        print("train_steps",train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            print("*******************EPOCH************",epoch)
            iter_count = 0
            print("iter_count",iter_count)
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (x_time,batch_x, x_mask,y_time, batch_y,y_mask) in enumerate(train_loader):
                print("============i==========",i)
                iter_count += 1
                print("iter_count ",iter_count)
                model_optim.zero_grad()
                print("x_time", x_time.shape)
                print("batch_x", batch_x.shape)
                print("NaNs in batch_x:", torch.isnan(batch_x).sum())


                #visualize_data(x_time,batch_x, x_mask)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)


                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.forecast_horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #print("self.args.use_amp",self.args.use_amp)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.forecast_horizon:, f_dim:]
                        batch_y = batch_y[:, -self.args.forecast_horizon:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    #f_dim = -1 if self.args.features == 'MS' else 0
                    #outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    #batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    #print("outputs",outputs)
                    #print("batch_y",batch_y)
                    #print("y_mask",y_mask)
                    loss = MSE_NEW(outputs, batch_y,y_mask)
                    print("loss ",loss)
                    train_loss.append(loss.item())
                    #print("train_loss",train_loss)
                print("Train iteration done")
                print("***************************")
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali( vali_loader, criterion)
                test_loss = self.vali( test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        print("Test Started")
        print("********************************")
        test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (x_time,batch_x, x_mask,y_time, batch_y,y_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.forecast_horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.forecast_horizon, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.forecast_horizon:, f_dim:]
                batch_y = batch_y[:, -self.args.forecast_horizon:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        #print('mse:{}, mae:{}'.format(mse, mae))
        test_mse = MSE_NEW(batch_y,outputs,y_mask)
        print("test_mse ",test_mse)
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('test_mse, corr:{}'.format(test_mse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                print(batch_x)
                batch_y = batch_y.float()
                print(batch_y)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.forecast_horizon, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
