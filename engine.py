# Copyright (c) V-DETR authors. All Rights Reserved.
import torch
import datetime
import logging
import math
import time
import sys
from MinkowskiEngine.utils import summary, batched_coordinates
import numpy as np
import MinkowskiEngine as ME
import MinkowskiSparseTensor 
from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
    batch_dict_to_cuda,
)
from utils.box_util import (flip_axis_to_camera_tensor, get_3d_box_batch_tensor)


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        if args.lr_scheduler == 'cosine':
            # Cosine Learning Rate Schedule
            curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
                1 + math.cos(math.pi * curr_epoch_normalized)
            )
        else:
            step_1, step_2 = args.step_epoch.split('_')
            step_1, step_2 = int(step_1), int(step_2)
            if curr_epoch_normalized < (step_1 / args.max_epoch):
                curr_lr = args.base_lr
            elif curr_epoch_normalized < (step_2 / args.max_epoch):
                curr_lr = args.base_lr / 10
            else:
                curr_lr = args.base_lr / 100
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
):
    ap_calculator = None

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device
    
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        batch_data_label = batch_dict_to_cuda(batch_data_label,local_rank=net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        if args.use_superpoint:
            inputs["superpoint_per_point"] = batch_data_label["superpoint_labels"]
        outputs = model(inputs)
        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)
        
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            eta_seconds = (max_iters - curr_iter) * (time.time() - curr_time)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; ETA {eta_str}"
            )
        
        curr_iter += 1
        barrier()

    return ap_calculator, curr_iter, curr_lr, loss_avg.avg, loss_dict_reduced


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        no_nms=args.test_no_nms,
        args=args
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        BBoxs = []
        VColors =[]
        ClassLabels = []
        FaceVIDs = []
        count = 0
        #scanname = batch_data_label["scan_name"]
        #print([len(v) for v in batch_data_label["point_clouds"]])
        #print(batch_data_label["point_clouds"])
        batch_data_label = batch_dict_to_cuda(batch_data_label,local_rank=net_device)
            
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        #print("Input Batch Index:",batch_idx)
        #print("Input data Label:",batch_data_label)
        
        #summary(model)
        torch.cuda.synchronize()
        start = time.time()

        outputs = model(inputs)
        #summary(model,inputs)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start

        print(elapsed_time, 'sec.')
        #print("Output:",outputs)
        #
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)
            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"
        else:
            loss_dict_reduced = None

        if args.cls_loss.split('_')[0] == "focalloss":
            outputs["outputs"]["sem_cls_prob"] = outputs["outputs"]["sem_cls_prob"].sigmoid()

        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        if args.axis_align_test:
            outputs["outputs"]["box_corners"] = outputs["outputs"]["box_corners_axis_align"]

        #print(batch_data_label)
        for i in range(len(outputs["outputs"]["box_corners"][0])):
            #print("Box Corner:",outputs["outputs"]["box_corners"][0][i])
            #print("Semantic Class Prob:",outputs["outputs"]["sem_cls_prob"][0][i])
            sem_cls_probs = outputs["outputs"]["sem_cls_prob"][0][i].detach().cpu().numpy()
            box_corners = outputs["outputs"]["box_corners"][0][i].detach().cpu().numpy()
            
            pred_sem_cls = np.argmax(sem_cls_probs, -1)
            pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal
            #print("Conf:",pred_sem_cls_prob)
            if(pred_sem_cls == 0):
                continue
            if(pred_sem_cls_prob < 0.1):
                continue
            for k in range(len(box_corners)):
                BBoxs.append(box_corners[k])
                R = pred_sem_cls%3
                G = (pred_sem_cls/3)%3
                B = (pred_sem_cls/9)%3
                VColors.append([R*100,G*100,B*100])
            ClassLabels.append(pred_sem_cls)
            FaceVIDs.append([1+(count*8),2+(count*8),3+(count*8),4+(count*8)])
            FaceVIDs.append([5+(count*8),6+(count*8),7+(count*8),8+(count*8)])
            FaceVIDs.append([1+(count*8),2+(count*8),6+(count*8),5+(count*8)])
            FaceVIDs.append([3+(count*8),4+(count*8),8+(count*8),7+(count*8)])
            FaceVIDs.append([1+(count*8),4+(count*8),8+(count*8),5+(count*8)])
            FaceVIDs.append([2+(count*8),3+(count*8),7+(count*8),6+(count*8)])
            
            count += 1
            #print("Predicted Label:",pred_sem_cls)

        ap_calculator.step_meter(outputs, batch_data_label)
        if is_primary() and curr_iter % args.log_every == 0:
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]"
            )
        curr_iter += 1
        barrier()
        #print("BBoxs:",BBoxs)
        #saveOBJ(args.out_obj_dir+str(batch_idx)+".obj",BBoxs,FaceVIDs,VColors)


    return ap_calculator, loss_avg.avg, loss_dict_reduced


def saveOBJ(filePath, vertices, faceVertIDs,vertexColors):
    f_out = open(filePath, 'w')

    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# Vertices: %s\n" %(len(vertices)))
    f_out.write("# Faces: %s\n" %(len( faceVertIDs)))
    f_out.write("#\n")
    f_out.write("####\n")


    for vi, v in enumerate( vertices ):
        vStr = "v %s %s %s"  %(v[0], v[1], v[2])
        if len( vertexColors) > 0:
            color = vertexColors[vi]
            vStr += " %s %s %s" %(color[0], color[1], color[2])
        vStr += "\n"
        f_out.write(vStr)
    f_out.write("# %s vertices\n\n"  %(len(vertices)))

    for fi, fvID in enumerate( faceVertIDs ):
        fStr = "f"
        for fvi, fvIDi in enumerate( fvID ):
            fStr += " %s" %(fvIDi)
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n"  %( len( faceVertIDs)) )

    f_out.write("# End of File\n")
    f_out.close()
