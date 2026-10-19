from Sentinel.dataset.hpc_dataset.data_provider.data_factory import data_provider
from Sentinel.dataset.hpc_dataset.exp.exp_basic import Exp_Basic
from Sentinel.dataset.hpc_dataset.utils.tools import EarlyStopping, adjust_learning_rate, visual
from Sentinel.dataset.hpc_dataset.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from Sentinel.dataset.hpc_dataset.utils.dtw_metric import dtw, accelerated_dtw
from Sentinel.dataset.hpc_dataset.utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_MMD(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_MMD, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_hostname_id = batch_x_hostname_id.int().to(self.device)
                batch_x_job_id = batch_x_job_id.int().to(self.device)
                batch_x_rack_id = batch_x_rack_id.int().to(self.device)
                batch_x_cabinet_id = batch_x_cabinet_id.int().to(self.device)
                batch_x_node_in_rack_id = batch_x_node_in_rack_id.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, :,  -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :,  :self.args.label_len, :], dec_inp], dim=2).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :,  -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, :,  -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_hostname_id = batch_x_hostname_id.int().to(self.device)
                batch_x_job_id = batch_x_job_id.int().to(self.device)
                batch_x_rack_id = batch_x_rack_id.int().to(self.device)
                batch_x_cabinet_id = batch_x_cabinet_id.int().to(self.device)
                batch_x_node_in_rack_id = batch_x_node_in_rack_id.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, :,  -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :,  :self.args.label_len, :], dec_inp], dim=2).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, :,  -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, :,-self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    # Unscale before clipping
                    scaler.unscale_(model_optim)
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # Read feature names
        features_file_path = os.path.join(self.args.root_path, 'features_mmd_sample.txt')
        if os.path.exists(features_file_path):
            with open(features_file_path, 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
        else:
            # Fallback if the feature names file is not found
            # Assuming c_out is the number of features if the file is not present
            feature_names = [f'feature_{i}' for i in range(self.args.c_out)]
            print(f"Warning: {features_file_path} not found. Using generic feature names: {feature_names}")


        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_hostname_id = batch_x_hostname_id.int().to(self.device)
                batch_x_job_id = batch_x_job_id.int().to(self.device)
                batch_x_rack_id = batch_x_rack_id.int().to(self.device)
                batch_x_cabinet_id = batch_x_cabinet_id.int().to(self.device)
                batch_x_node_in_rack_id = batch_x_node_in_rack_id.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, :,  -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :,  :self.args.label_len, :], dec_inp], dim=2).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_hostname_id, batch_x_job_id, batch_x_rack_id, batch_x_cabinet_id, batch_x_node_in_rack_id)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, :]
                batch_y = batch_y[:, :, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1] * shape[2], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1] * shape[2], -1)).reshape(shape)

                outputs = outputs[:, :,  :, f_dim:]
                batch_y = batch_y[:, :,  :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1] * shape[2], -1)).reshape(shape)
                    # gt = np.concatenate((input[0, :, :, -1], true[0, :, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, :, -1], pred[0, :, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast_mmd.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        f = open("result_long_term_forecast_mmd.csv", 'a')
        if f.tell() == 0:
            f.write("model, mse, mae, dtw, pred_len, seq_len, label_len, feature, data\n")
        f.write(f"{self.args.model}, {mse}, {mae}, {dtw}, {self.args.pred_len}, {self.args.seq_len}, {self.args.label_len}, {self.args.features}, {self.args.data}\n")
        f.close()

        f = open("result_long_term_forecast_mmd_per_variable.csv", 'a')
        # Calculate per-variable MSE
        per_variable_mse = []
        if preds.shape[-1] == len(feature_names):
            for i in range(preds.shape[-1]):
                var_mse = np.mean((preds[:, :, i] - trues[:, :, i]) ** 2)
                per_variable_mse.append(var_mse)
        else:
            print(f"Warning: Number of features in preds ({preds.shape[-1]}) does not match number of feature names ({len(feature_names)}). Cannot calculate per-variable MSE reliably.")
            per_variable_mse = ['N/A'] * len(feature_names) # Or handle as appropriate


        if f.tell() == 0:
            header_cols = ["model", "pred_len", "seq_len", "label_len", "feature_type", "data"] + feature_names
            f.write(",".join(header_cols) + "\n")
        
        row_values = [
            str(self.args.model),
            str(self.args.pred_len),
            str(self.args.seq_len),
            str(self.args.label_len),
            str(self.args.features),
            str(self.args.data)
        ] + [str(m) for m in per_variable_mse]
        f.write(",".join(row_values) + "\n")
        f.close()

        return
