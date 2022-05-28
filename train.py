import copy
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torchvision
import torch.multiprocessing as mp
from tqdm import tqdm
import hydra
from log_utils import Logger, Visualizer
from utils import collate_fn, change_requires_grad
import os
from time import time
import traceback


def train(proc_id, cfg):
    print(proc_id)
    parallel = len(cfg.devices) > 1
    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend="nccl", init_method="env://", world_size=len(cfg.devices), rank=proc_id)
    device = torch.device(proc_id)
    dataset = hydra.utils.instantiate(cfg.dataset, _recursive_=False)
    print(cfg.dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader, collate_fn=collate_fn)
    model = hydra.utils.instantiate(cfg.model).to(device)
    if cfg.resume:
        state_dict = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    loss_fn = hydra.utils.instantiate(cfg.loss).to(device)
    logger = Logger(proc_id, device=device)

    if cfg.mode == "pretrain_segmentation":
        optim = torch.optim.Adam(model.parameters(), **cfg.optimizer)

        for step in range(cfg.total_steps):
            data = next(dataloader_iter)
            _, seg = data
            seg = seg.to(device)
            seg_rec, q_loss = model(seg)
            loss = loss_fn(q_loss, seg, seg_rec)

            if step % cfg.log_period == 0:
                logger.log(loss, q_loss, seg, seg_rec, step)
                torch.save(model.state_dict(), "checkpoint.pt")

            loss.backward()
            if step % cfg.accumulate_grad == 0:
                optim.step()
                optim.zero_grad()

            if step == cfg.total_steps:
                torch.save(model.state_dict(), "final.pt")
                return

    elif cfg.mode == "pretrain_image":
        vq_optim = torch.optim.Adam(model.parameters(), **cfg.optimizer.vq)
        disc_optim = torch.optim.Adam(loss_fn.discriminator.parameters(), **cfg.optimizer.disc)

        pbar = tqdm(enumerate(dataloader), total=cfg.total_steps) if proc_id==0 else enumerate(dataloader)
        try:
            t = time()
            for step, data in pbar:
                t = time()
                img, _, bbox_objects_all, bbox_faces_all, _ = data
                img = img.to(device)
                bbox_objects = copy.deepcopy(bbox_objects_all)
                bbox_faces = copy.deepcopy(bbox_faces_all)
                t = time()
                img_rec, q_loss = model(img)
                t = time()

                change_requires_grad(model, False)
                d_loss = loss_fn(optimizer_idx=1, global_step=step, images=img, reconstructions=img_rec)
                d_loss.backward()
                change_requires_grad(model, True)
                t = time()

                change_requires_grad(loss_fn.discriminator, False)
                loss, (nll_loss, object_loss, face_loss) = loss_fn(optimizer_idx=0, global_step=step, images=img,
                                                                   reconstructions=img_rec,
                                                                   codebook_loss=q_loss, bbox_obj=bbox_objects,
                                                                   bbox_face=bbox_faces,
                                                                   last_layer=model.module.decoder.model[-1])
                t = time()
                loss.backward()
                change_requires_grad(loss_fn.discriminator, True)
                t = time()
                if step%cfg.accumulate_grad==0:
                    disc_optim.step()
                    disc_optim.zero_grad()
                    vq_optim.step()
                    vq_optim.zero_grad()

                ### LOGGING PART
                if step % cfg.log_period == 0:
                    logger.log(loss=loss, q_loss=q_loss, img=img, img_rec=img_rec, d_loss=d_loss, nll_loss=nll_loss, object_loss=object_loss, face_loss=face_loss, step=step)
                    torch.save(model.module.state_dict(), "checkpoint.pt")

                if step == cfg.total_steps:
                    torch.save(model.module.state_dict(), "final.pt")
                    return
        except Exception as e:
            print('Caught exception in worker thread (x = %d):' % proc_id)

            # This prints the type, value, and stack trace of the
            # current exception being handled.
            with open("error.log", "a") as f:
                traceback.print_exc(file=f)
            raise e


def visualize(cfg):
    device = torch.device(cfg.devices[0])
    model = torch.nn.DataParallel(hydra.utils.instantiate(cfg.model)).to(device)
    checkpoint = hydra.utils.to_absolute_path(cfg.checkpoint)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.module
    model.eval()
    img = torch.rand(1, 159, 256, 256).to(device)
    dataset = hydra.utils.instantiate(cfg.dataset)
    visualizer = Visualizer(device=device)
    print("Processing...")
    for i, data in enumerate(dataset):
        if i == 40:
            break
        print("Processing image ", i)
        img, seg = data
        img = img.to(device).unsqueeze(0)
        seg = seg.to(device).unsqueeze(0)
        seg_rec, _ = model(seg)
        visualizer(i, image=img, seg=seg, seg_rec=seg_rec, )

    print(model(dataset[0][1].unsqueeze(0).to(device))[0].shape)


def preprocess_dataset(cfg):
    # dataset = hydra.utils.instantiate(cfg.dataset,)
    dataset = cfg.dataset
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)
    preprocessor(dataset)


@hydra.main(config_path="conf", config_name="img_config", version_base="1.2")
def launch(cfg):
    if "pretrain" in cfg.mode:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in cfg.devices])
        cfg.checkpoint = hydra.utils.to_absolute_path(cfg.checkpoint)
        if len(cfg.devices) == 1:
            train(0, cfg)
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "33751"
            p = mp.spawn(train, nprocs=len(cfg.devices), args=(cfg,))
    elif "show" in cfg.mode:
        visualize(cfg)
    elif "preprocess_dataset" in cfg.mode:
        preprocess_dataset(cfg)


if __name__ == "__main__":
    launch()
