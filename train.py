"""
Training script for 3D YOLO.
Example cmd line call: python train.py --data example.yaml --adam
"""

# standard library imports
import argparse
from copy import deepcopy
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

# set path for local imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO3D root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 2D YOLO imports
from utils.general import colorstr, labels_to_class_weights, increment_path, print_args, \
    check_yaml, check_file, get_latest_run, one_cycle, print_mutation, strip_optimizer, check_suffix
from utils.callbacks import Callbacks
from utils.torch_utils import select_device, de_parallel, EarlyStopping, ModelEMA, torch_distributed_zero_first, intersect_dicts
from utils.metrics import fitness
from utils.plots import plot_evolve

# 3D YOLO imports
from models3D.model import Model, attempt_load
from utils3D.datasets import nifti_dataloader, normalize_CT, normalize_MR
from utils3D.lossandmetrics import ComputeLossVF
from utils3D.anchors import nifti_check_anchors
from utils3D.general import check_dataset

import val


# Configuration
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# testing parameters, remove after dev
default_size = 350 # edge length for testing, below 350 the model can't process the data
default_epochs = 200
default_batch = 8


def train(hyp, opt, device, callbacks):
    # parsing the arguments
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, norm = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.norm
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load model hyps dict
    
    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None
   
    # Config
    plots = False
    cuda = device.type != 'cpu'
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    
    # Model
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights,map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=1, nc=nc, anchors=hyp.get('anchors')).to(device) # create model
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    else:        
        model = Model(cfg = cfg, ch=1, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    # loads from models folder
    with open(data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)  # model dict

    nc = int(data_dict['nc'])  # number of classes

    train_path, val_path = data_dict['train'], data_dict['val']
    
    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    
    # Image size
    imgsz = opt.imgsz
    stride = model.stride
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm3d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)   
    
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2
    
    # Scheduler
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None
    
    # Resume
    start_epoch, best_fitness = 0, 0.0
    
    # Trainloader
    train_loader, train_dataset = nifti_dataloader(train_path,
                                                   imgsz=imgsz,
                                                   batch_size=batch_size,
                                                   hyp=hyp,
                                                   single_cls=single_cls,
                                                   stride=stride,
                                                   rank=LOCAL_RANK,
                                                   workers=workers,
                                                   augment=True)
    mlc = int(np.concatenate(train_dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    
    # Process 0 不采用分布式计算
    if RANK in [-1, 0]:
        val_loader = nifti_dataloader(val_path,
                                      imgsz=imgsz,
                                      batch_size=batch_size,
                                      stride=stride,
                                      single_cls=single_cls,
                                      workers=workers)[0]

        if not resume:
            # Anchors
            if not opt.noautoanchor:
                # print(f"**** in the train and if not opt.noautoanchor:, the hyp['anchor_t'] is {hyp['anchor_t']}")
                nifti_check_anchors(train_dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')
        
    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        
    # Model parameters
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps), default is 3
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / default_size) ** 3 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    # t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLossVF(model)  # init Varifocal loss class

    for epoch in range(epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        
        # train loop
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            # Normalization
            if norm.lower() == 'ct':
                imgs = normalize_CT(imgs.to(device, non_blocking=True).float())  # int to float32, -1024-1024 to 0.0-1.0
            elif norm.lower() == 'mr':
                imgs = normalize_MR(imgs.to(device, non_blocking=True).float())  # int to float32, mean 0 std dev
            else:
                raise NotImplementedError("You'll need to write your own normalization algorithm here.")

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                        
            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                ########## xjl debug ###########
                print(f'@@@ 1 @@@ image shape is {imgs.shape}')
                print(f'### 2 ### pred len and the pred[0] shape is {len(pred), pred[0].shape}')
                print(f'$$$ 3 $$$ target len is and the target[0] is {len(targets), targets[0]}')
                loss, loss_items = compute_loss(pred, targets.to(device).float())  # loss scaled by batch_size
                print(f'&&& 4 &&& the loss is {loss}, the loss_item is {loss_items}')
                del pred

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            
            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
        
            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, False)
            del imgs, targets
            # end batch ------------------------------------------------------------------------------------------------
            
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        
        # validation loop
        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, _, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss,
                                           norm=norm)
                
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
            
            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict()}
                
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
                
            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    # a final validation loop to compare the model with the final trained weights to the model with the best score from above
    if RANK in [-1, 0]:
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            iou_thres=0.60,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss,
                                            norm=norm)  # val best model with plots

        callbacks.run('on_train_end', last, best, plots, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    # parses the options in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models3D/yolo3Ds.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/example.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=default_epochs)
    parser.add_argument('--batch-size', type=int, default=default_batch, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=default_size, help='train, val image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--norm', type=str, default='CT', help='normalization type, options: CT, MR, Other')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        
    # Resume
    if opt.resume and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    
    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            dist.destroy_process_group()
    
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'max_zoom': (0, 1.0, 2.0), # maximum zoom factor
                'min_zoom': (0, 0.4, 1.0), # minimum zoom factor
                'prob_zoom': (0, 0.3, 0.7), # probability of zoom augmentation
                'prob_cutout': (0, 0.3, 0.7), # probability of cutout augmentation
        }
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists
    
        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate
   
            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits
                
            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)

            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)
            
        # Plot results
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')
       
        
def run(**kwargs):
    # Usage: import train; train.run(data='example.yaml', imgsz=350, weights='yolo3Ds.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
   