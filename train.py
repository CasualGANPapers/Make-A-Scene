import copy
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from tqdm import tqdm
import hydra
from log_utils import Logger, Visualizer
from utils import collate_fn
import os


def train(proc_id, cfg):
    print(proc_id)
    parallel = len(cfg.devices) > 1
    if parallel:
        torch.cuda.set_device(proc_id)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend="nccl", init_method="env://", world_size=len(cfg.devices), rank=proc_id)
    device = torch.device(proc_id)
    dataset = hydra.utils.instantiate(cfg.dataset, _recursive_=False)
    dataloader = DataLoader(dataset, **cfg.dataloader, collate_fn=collate_fn)
    model = hydra.utils.instantiate(cfg.model).to(device)
    if cfg.resume:
        state_dict = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    if parallel:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    loss_fn = hydra.utils.instantiate(cfg.loss).to(device)
    logger = Logger(proc_id, device=device)

    dataloader_iter = iter(dataloader)

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

        pbar = tqdm(range(cfg.total_steps))
        data = next(dataloader_iter)  # try to overfit single image
        img, _, bbox_objects_all, bbox_faces_all, _ = data
        img = img.to(device)
        for step in pbar:
            # data = next(dataloader_iter)
            # img, _, bbox_objects, bbox_faces, _ = data
            # img = img.to(device)
            bbox_objects = copy.deepcopy(bbox_objects_all)
            bbox_faces = copy.deepcopy(bbox_faces_all)
            img_rec, q_loss = model(img)

            d_loss = loss_fn(optimizer_idx=1, global_step=step, images=img, reconstructions=img_rec)
            disc_optim.zero_grad()
            d_loss.backward()
            disc_optim.step()

            loss, (nll_loss, object_loss, face_loss) = loss_fn(optimizer_idx=0, global_step=step, images=img,
                                                               reconstructions=img_rec,
                                                               codebook_loss=q_loss, bbox_obj=bbox_objects,
                                                               bbox_face=bbox_faces,
                                                               last_layer=model.decoder.model[-1])
            vq_optim.zero_grad()
            loss.backward()
            vq_optim.step()

            if step % 100 == 0:
                plt.imshow(img_rec[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.show()

            if step % cfg.log_period == 0:
                logger.log(loss=loss, q_loss=q_loss, img=img, img_rec=img_rec, d_loss=d_loss, step=step)
                torch.save(model.state_dict(), "checkpoint.pt")
            pbar.set_postfix(loss=loss.item(), q_loss=q_loss.item(), d_loss=d_loss.item(),
                             nll_loss=nll_loss.item(), object_loss=object_loss.item(), face_loss=face_loss.item())

            # if step % cfg.accumulate_grad == 0:
            #     vq_optim.step()
            #     vq_optim.zero_grad()
            #     disc_optim.step()
            #     disc_optim.zero_grad()

            if step == cfg.total_steps:
                torch.save(model.state_dict(), "final.pt")
                return


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


@hydra.main(config_path="conf", config_name="img_config")
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
