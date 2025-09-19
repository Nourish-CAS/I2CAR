
import time
from utils.utils import *
from model.I2CAR import I2CAR
from data_factory.data_loader import get_loader_segment
# from metrics.metrics import *
import warnings
import numpy as np
from contrast_loss import inter_time_contrastive_loss,inter_variable_contrastive_loss
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from fvcore.nn import FlopCountAnalysis, parameter_count



from arguement import *
warnings.filterwarnings('ignore')

# def my_kl_loss(p, q):
#     res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
#     return torch.mean(torch.sum(res, dim=-1), dim=1)

def my_kl_loss(p, q):
    epsilon = 1e-8  # 更小的epsilon，确保数值安全性
    p = torch.clamp(p, min=epsilon)  # 保证p不会有接近0的数值
    q = torch.clamp(q, min=epsilon)  # 保证q不会有接近0的数值

    # 计算KL散度
    res = p * (torch.log(p) - torch.log(q))

    # 对结果求和，并取平均
    return torch.mean(torch.sum(res, dim=-1), dim=1)



def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)



        self.build_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())




        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()


    def build_model(self):
        self.model = I2CAR(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads,
                                d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size,
                                channel=self.input_c)
      
        print(torch.__version__)
        print(torch.cuda.is_available())
        print(torch.version.cuda)
        if torch.cuda.is_available():
            print("build_model_torch.cuda.is_available:",torch.cuda.is_available())
            self.model.cuda()
            #add
            # self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior ,_,_= self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print(self.device)
        print("Is CUDA available: ", torch.cuda.is_available())
        print("Number of GPUs: ", torch.cuda.device_count())
        # print("Current GPU: ", torch.cuda.current_device())
        # print("GPU Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                # print("***************  i  ***************: ",i)
                # print("***************input_data.size()***************: ", input_data.size())
                # print("***************labels.size()***************: ", labels.size())
                # print("***************enumerate(self.train_loader)***************: ", enumerate(self.train_loader).__sizeof__())
                neg_np, info, mask = make_negative_multivariate(
                    input_data,  # [B,T,D]
                    fs=1.0,  # 采样率 (如无特别定义就保持 1.0)
                    max_dim_ratio=0.5,  # 每个 batch 内不超过一半维度
                    noise_scale=0.2,  # jitter 噪声强度
                    scaling_range=(0.5, 1.5),
                    smooth_scaling=False,
                    seed=None,
                    return_mask=True
                )



                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                neg_np = neg_np.float().to(self.device)
                #print(f"==============input shape================{input.shape}")
                series, prior,time_consis,var_consis= self.model(input)
                series_neg, prior_neg,time_consis_neg,var_consis_neg = self.model(neg_np)


                series_loss = 0.0
                prior_loss = 0.0
                inter_time_con = 0.0
                inter_var_con = 0.0

                # print(f"len(prior)  ：{len(prior)}")

                #We are organizing the code and encapsulating the loss function into an independent module, which is used in the ablation experiments and still maintains strong performance.
                for u in range(len(prior)):

                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                    inter_time_con += inter_time_contrastive_loss(time_consis[u],time_consis_neg[u])
                    inter_var_con += inter_variable_contrastive_loss(var_consis[u],var_consis_neg[u])

                inter_time_con = inter_time_con / len(prior)
                inter_var_con = inter_var_con / len(prior)
                loss1 = (inter_time_con + inter_var_con)/2

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                loss2 = prior_loss - series_loss

                # alpha=0.6
                # # loss = alpha*loss1 + (1-alpha)*loss2
                loss = loss1 +  loss2
                # print(f"loss 1 is {loss1}")
                # print(f"loss 2 is {loss2}")
                # loss = loss1



                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    print(f"loss1: {loss1}")

                # print(f"Loss at step: {loss.item()}")
                if torch.isnan(loss):
                    print("NaN detected in loss function!")
                    break

                loss.backward()
                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Apply gradient clipping


                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)


    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Params: total={total_params / 1e6:.3f}M, trainable={trainable_params / 1e6:.3f}M")

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior,_,_ = self.model(input)
            # FLOPs
            #flops = FlopCountAnalysis(self.model, input)
            #print("FLOPs (fvcore):", f"{flops.total() / 1e9:.3f} GFLOPs")  # 已按FLOPs口径给出

            # Params（再算一遍做对照）
            #params_dict = parameter_count(self.model)
            #print("Params (fvcore):", f"{sum(params_dict.values()) / 1e6:.3f} M")


            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            # cri = metric.detach()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior,_,_ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            # cri = metric.detach()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior,_,_ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            # cri = metric.detach()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        print(pred)
        gt = test_labels.astype(int)
        print(gt)



        from eval_metrics import f1_prec_recall_PA,f1_prec_recall_K,aff_metrics,trad_metrics
        f1_pa, precision_pa, recall_pa = f1_prec_recall_PA(pred, gt)

        f1_k, precision_k, recall_k = f1_prec_recall_K(pred, gt)

        precision_aff,recall_aff = aff_metrics(pred,gt)
        # f1_aff = 2 * (precision_aff * recall_aff) / (precision_aff + recall_aff)
        f1_aff = 0

        acc_trad, precision_trad, recall_trad, f1_trad = trad_metrics(pred, gt)


        print("Precision_pa : {:0.4f}, Recall_pa : {:0.4f}, F1_pa : {:0.4f}".format(precision_pa,recall_pa,f1_pa))
        print("Precision_k : {:0.4f}, Recall_k : {:0.4f}, F1_k : {:0.4f}".format(precision_k, recall_k, f1_k))
        print("Precision_aff : {:0.4f}, Recall_aff : {:0.4f}, F1_aff : {:0.4f}".format(precision_aff, recall_aff, f1_aff))
        print("acc_trad : {:0.4f},Precision_trad : {:0.4f}, Recall_trad : {:0.4f}, F1_trad : {:0.4f}".format(acc_trad, precision_trad, recall_trad, f1_trad))


      

        # return accuracy, precision, recall, f_score
        return f1_pa, precision_pa, recall_pa, f1_k, precision_k, recall_k, precision_aff,recall_aff, f1_aff, acc_trad, precision_trad, recall_trad, f1_trad



