import os
import os.path as osp
import json
from typing import Any
import torch
import pickle
import logging
import numpy as np
from NMOmodel import neural_manifold_operator
from tqdm import tqdm
from API import *
from utils import *
from timeit import default_timer
from collections import OrderedDict

checkpoint_path = '/data/workspace/yancheng/MM/neural_manifold_operator/NMO_NS_600/NMO_NS_600/checkpoint.pth'

def load_partial_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()

    checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)

def relative_l1_error(true_values, predicted_values):
    error = torch.abs(true_values - predicted_values)
    return torch.mean(error / torch.abs(true_values))


class LpLoss(object):
    def __init__(self, d=5, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d>0 and p>0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    
    def __call__(self, pred_y, batch_y):
        num_examples = pred_y.size()[0]

        h = 1.0 / (pred_y.size()[3] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(pred_y.view(num_examples, -1) - batch_y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        
        return all_norms



class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = neural_manifold_operator(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)
        load_partial_state_dict(self.model, checkpoint_path)
        

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        load_partial_state_dict(self.model, checkpoint_path)
        lamda = 0.5
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y, x_rec = self.model(batch_x)

                #loss = self.criterion(pred_y, batch_y)
                loss_fn = LpLoss(d=5, p=2, size_average=True, reduction=True)
                loss_pre = loss_fn(pred_y, batch_y)
                loss_rec = loss_fn(x_rec, batch_x)
                loss = lamda * loss_pre + (1-lamda) * loss_rec
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
           

            train_loss = np.average(train_loss)
            # 训练SWE时候启动
            values = 1
            train_loss = train_loss * values
            

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.8f} Vali Loss: {2:.8f}\n".format(
                    epoch + 1, train_loss, vali_loss * values))
                recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        values = 1
        self.model.eval()
        train_l2_full = 0
        lamda = 0.5
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)        
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y, x_rec = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            #loss = self.criterion(pred_y, batch_y)
            loss_fn = LpLoss(d=5, p=2, size_average=True, reduction=True)
            loss_pre = loss_fn(pred_y, batch_y)
            loss_rec = loss_fn(x_rec, batch_x)
            loss = lamda * loss_pre + (1-lamda) * loss_rec
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())
            train_l2_full += loss.item()
        #print("train mse:", train_l2_full / length)
        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        # mse, mae, ssim, psnr, rel_err = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        # print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        
        l2_error = torch.nn.MSELoss()(torch.tensor(preds), torch.tensor(trues)).item()
        relative_l2_error = l2_error / torch.nn.MSELoss()(torch.tensor(trues), torch.zeros_like(torch.tensor(trues))).item()

        l1_error = torch.nn.L1Loss()(torch.tensor(preds), torch.tensor(trues)).item()
        rel_l1_err = relative_l1_error(torch.tensor(trues), torch.tensor(preds)).item()
        # 计算RMSE
        rmse = torch.sqrt(torch.mean((torch.tensor(preds) - torch.tensor(trues)) ** 2))
        rmse = rmse.item()
        print_log('RMSE: {:.8f}, L2 error: {:.8f}, Relative L2 error: {:.8f},'.format(rmse, 
                                                                                      l2_error * values, 
                                                                                      relative_l2_error * values))
       
        self.model.train()
        return total_loss

    def test(self, args):
        values = 1
        self.model.eval()
        loss_fn = LpLoss(d=5, p=2, size_average=True, reduction=True)
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            pred_y, x_rec = self.model(batch_x.to(self.device))


            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        l2_error = torch.nn.MSELoss()(torch.tensor(preds), torch.tensor(trues)).item()
        relative_l2_error = l2_error / torch.nn.MSELoss()(torch.tensor(trues), torch.zeros_like(torch.tensor(trues))).item()


        # 计算RMSE
        rmse = torch.sqrt(torch.mean((torch.tensor(preds) - torch.tensor(trues)) ** 2))
        rmse = rmse.item()
        
        

        print_log('RMSE: {:.8f}, L2 error: {:.8f}, Relative L2 error: {:.8f},'.format(rmse, 
                                                                                      l2_error * values, 
                                                                                      relative_l2_error * values))


        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return rmse