import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.Experiment import get_experiment
from lib.utils.dist_utils import is_distributed, all_reduce_mean

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1
EPOCHS = 200


def get_optimizer(model, learning_rate, weight_decay):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    parameters_decay, parameters_no_decay = model.separate_parameters()
    print("separate_parameters:", len(parameters_decay), len(parameters_no_decay))
    optim_groups = [
        {"params": parameters_decay, "weight_decay": weight_decay},
        {"params": parameters_no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def train(rank: int, args):
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
        )
        print(f"Distributed init: {rank}/{args.world_size}")

    print(f"is_distributed: {is_distributed()}")

    torch.cuda.set_device(rank)
    if is_distributed():
        dist.barrier()

    torch.manual_seed(0)

    experiment = get_experiment(args.experiment_patches)

    model = experiment.model
    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if 0 == rank:
        print(f"total_params: {total_params}, trainable_params: {trainable_params}")

    train_loader, validation_loader = experiment.train_loader, experiment.val_loader
    train_loss, validation_loss = experiment.train_loss, experiment.val_loss

    optimizer = get_optimizer(model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                                       steps_per_epoch=len(train_loader), epochs=EPOCHS)
    iters_per_epoch = len(train_loader)
    total_iters = iters_per_epoch * EPOCHS

    running_loss = None
    running_accuracy = None
    start_time = time.time()
    for epoch in range(EPOCHS):

        model.train()
        if is_distributed():
            train_loader.sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(rank)
            labels = labels.to(rank)

            images, labels = train_loss.prepare_batch(images, labels)

            optimizer.zero_grad()

            outputs = model(images)
            loss = train_loss(outputs, labels)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            predictions = torch.max(outputs, 1)[1]
            accuracy = torch.eq(predictions, labels).float().sum() / images.shape[0]

            running_loss = loss.detach() if running_loss is None else 0.99 * running_loss + 0.01 * loss.detach()
            running_accuracy = accuracy.detach() if running_accuracy is None else 0.99 * running_accuracy + 0.01 * accuracy.detach()

            if 0 == (1 + i) % 100 or 1 + i == iters_per_epoch:
                running_loss = all_reduce_mean(running_loss)
                running_accuracy = all_reduce_mean(running_accuracy)

                time_elapsed = time.time() - start_time
                iters = epoch * iters_per_epoch + i + 1
                remaining_iters = total_iters - iters
                eta_secs = remaining_iters / iters * time_elapsed
                eta_str = str(datetime.timedelta(seconds=eta_secs))

                if 0 == rank:
                    lr = lr_scheduler.get_last_lr()[0]
                    print(f"epoch: {epoch:4d}/{EPOCHS:4d}, iter: {1 + i: 6d}, running_loss: {running_loss.item():.4f}, "
                          f"running_accuracy: {running_accuracy.item():.4f}, "
                          f" lr: {lr:.6f}, "
                          f" eta: {eta_str}",
                          flush=True)

        model.eval()
        avg_loss = 0.0
        avg_accuracy = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(validation_loader):
                images = images.to(rank)
                labels = labels.to(rank)
                outputs = model(images)
                loss = validation_loss(outputs, labels)

                predictions = torch.max(outputs, 1)[1]
                accuracy = torch.eq(predictions, labels).float().sum() / images.shape[0]

                avg_loss += loss.detach()
                avg_accuracy += accuracy.detach()

        avg_loss = all_reduce_mean(avg_loss) / len(validation_loader)
        avg_accuracy = all_reduce_mean(avg_accuracy) / len(validation_loader)

        if 0 == rank:
            print(f"Validation: average loss: {avg_loss:.4f}, accuracy: {avg_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('experiment_patches', nargs='+', type=str)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print("world_size: ", world_size)
    args.world_size = world_size
    if world_size > 1:
        print("spawn: ", world_size)
        mp.spawn(train,
                 args=(args,),
                 nprocs=world_size,
                 join=True)
    else:
        train(0, args)


if __name__ == '__main__':
    main()
