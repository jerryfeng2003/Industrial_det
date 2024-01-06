from pprint import pprint
from configs.config import cfg
import pandas as pd
import torch
import sys
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from configs.config import cfg_from_file, project_root
from datasets.dnd_dataset import DND
from models.my_model import MyModel
from utils.pytorch_misc import *
import yaml
from timm.data.mixup import Mixup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        pass




# torch.cuda.synchronize()

def train_epoch(model, train_loader, cfg, optimizer, epoch_num):
    model.train()

    # init
    loss_epoch = AverageMeter()
    top1_epoch = AverageMeter()

    accumulate_step = 0
    optimizer.zero_grad()

    timer_epoch = Timer()
    timer_batch_avg = Timer()

    timer_epoch.tic()
    timer_batch_avg.tic()
    for batch_num, batch in enumerate(train_loader):

        img = batch['img'].to(device)
        labels = batch['label_idxs'].to(device)
        bsz = labels.shape[0]

        # mixup_fn = Mixup(
        #     mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        #     prob=0.1, switch_prob=0.5, mode='batch',
        #     label_smoothing=0.1, num_classes=cfg.num_classes)
        # _, targets = mixup_fn(img, labels)

        scores = model.forward(img)
        loss = F.cross_entropy(scores, labels)
        loss_epoch.update(loss, labels.shape[0])
        # loss = F.cross_entropy(scores, targets)
        # loss_epoch.update(loss, targets.shape[0])

        top1_batch = top_k_accuracy(scores, labels, topk=(1,))[0]
        top1_epoch.update(top1_batch, bsz)

        # gradient accumulation
        accumulate_step += 1
        loss /= cfg.acc_bsz
        loss.backward()
        if accumulate_step == cfg.acc_bsz:
            optimizer.step()
            optimizer.zero_grad()
            accumulate_step = 0

        batch_time_avg = timer_batch_avg.toc(average=True)
        if batch_num % cfg.p_interval == 0:
            print(
                "epo:{:2d}-{:4d}/{:4d}, exp {:4.1f}m/epo, avg {:4.3f}s/b, loss_batch: {:.3f} Acc (train-batch): {:5.2f}% Acc (train-epoch): {:5.2f}%"
                .format(epoch_num, batch_num, len(train_loader) - 1, len(train_loader) * batch_time_avg / 60,
                        batch_time_avg,
                        loss.item(), top1_batch.item() * 100.0, top1_epoch.avg.item() * 100.0), flush=True)
        timer_batch_avg.tic()

    print("Epoch {}: elapsed_time: {:.2f}m Acc (train): {:.2f}% {} loss_epo_avg: {:.3f}"
          .format(epoch_num, (timer_epoch.toc(average=False)) / 60, top1_epoch.avg.item() * 100.0,
                  [top1_epoch.sum.item(), top1_epoch.count], loss_epoch.avg.item()), flush=True)


def val_epoch(model, data_loader, cfg, is_final=False):
    model.eval()

    # init
    loss_epoch = AverageMeter()
    top1_epoch = AverageMeter()
    timer_epoch = Timer()
    timer_epoch.tic()

    results = []
    predict = []
    for batch_num, batch in enumerate(data_loader):

        img = batch['img'].to(device)
        bsz = img.shape[0]

        with torch.no_grad():
            scores = model.forward(img)
            preds = scores.argmax(1)

            if data_loader.dataset.split == 'val':
                # if not is_final:
                labels = batch['label_idxs'].to(device)
                loss = F.cross_entropy(scores, labels)
                loss_epoch.update(loss, bsz)

                top1_batch = top_k_accuracy(scores, labels, topk=(1,))[0]
                top1_epoch.update(top1_batch, bsz)

            results.append(batch['img_name'][0][14:-3] + 'CSV')
            predict.append(chr(ord('A') + int(preds)))

    if data_loader.dataset.split == 'val':
        # if not is_final:
        print("Val: elapsed_time: {:.2f}m Acc (val): {:.2f}% {} loss_epo_avg: {:.3f}"
              .format((timer_epoch.toc(average=False)) / 60, top1_epoch.avg.item() * 100.0,
                      [top1_epoch.sum.item(), top1_epoch.count], loss_epoch.avg.item()), flush=True)

    if is_final:
        pred_path = os.path.join(res_path, 'predictions.csv')
        df = pd.DataFrame(columns=['defectType', 'fileName'])
        df['fileName'] = results
        df['defectType'] = predict
        df.to_csv(pred_path, index=False)
        print("Predictions saved at {}".format(pred_path))

    return top1_epoch.avg * 100.0


def train_model(model, train_loader, val_loader, test_loader, start_epoch, cfg):
    optimizer, scheduler = get_optim(model, cfg)

    best_acc = 0.0
    best_path = ''
    for epoch in range(start_epoch + 1, start_epoch + 1 + cfg.TRAIN.MAX_EPOCH):
        print("=================Epo {}: Training=================".format(epoch))
        train_epoch(model, train_loader, cfg, optimizer, epoch)
        # separate train and val set
        if val_loader is not None:
            print("=================Epo {}: Validating=================".format(epoch))
            cur_acc = val_epoch(model, val_loader, cfg, is_final=False)

            # save the best model
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_path = save_best_model(epoch, model, optimizer, suffix=cfg.model_name)
            scheduler.step(cur_acc)
            # print("Current lr: ", scheduler.get_last_lr())

        # w/o val set
        else:
            best_path = save_best_model(epoch, model, optimizer, suffix=cfg.model_name)  # actually saving current model
            scheduler.step()
            # print("Current lr: ", scheduler.get_last_lr())


if __name__ == '__main__':
    # init
    pprint(cfg)
    set_seed(cfg.SEED)
    res_path = os.path.join(project_root, 'experiments', time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
    os.makedirs(res_path, exist_ok=True)
    sys.stdout = Logger(res_path + '/experiment.log', sys.stdout)

    # dataset & dataloader
    if not cfg.is_test:
        train_dataset = DND(cfg, split=cfg.train_set)
        train_loader = DataLoader(train_dataset, batch_size=cfg.bsz, shuffle=True, num_workers=cfg.NWORK,
                                  drop_last=False)
        val_dataset = DND(cfg, split='val')
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.NWORK, drop_last=False)
        # val_loader = None
    test_dataset = DND(cfg, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.NWORK, drop_last=False)

    # model
    cfg.num_classes = len(test_dataset.class_to_ind)
    model = MyModel(cfg)
    # print(print_para(model), flush=True)

    # load model
    model = torch.nn.DataParallel(model)  # , device_ids=[0, 1]
    model, start_epoch = load_model(cfg, model)
    model.to(device)

    if not cfg.is_test:
        train_model(model, train_loader, val_loader, test_loader, start_epoch, cfg)
    else:
        print("=================Start testing!=================", flush=True)
        _ = val_epoch(model, test_loader, cfg, is_final=True)
