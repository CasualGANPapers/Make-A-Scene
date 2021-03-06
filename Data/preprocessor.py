import os
import torch
import torch.multiprocessing as mp
from .preprocessors import Detectron2
from .preprocessors import HumanParts
from .preprocessors import HumanFace
from tqdm import tqdm
import numpy as np


class BasePreprocessor:
    proc_types = {"panoptic": Detectron2, "human": HumanParts, "face": HumanFace}

    def __init__(
        self,
        preprocessed_folder,
        proc_per_gpu=None,
        proc_per_cpu=None,
        devices=None,
        machine_idx=0,
        machines_total=1,
    ):
        self.idx = machine_idx
        self.machines_total = machines_total
        self.devices = list(devices)
        self.proc_per_gpu = proc_per_gpu
        self.proc_per_cpu = proc_per_cpu
        self.preprocessed_folder = preprocessed_folder
        self.preprocessed_path = os.path.join(
            preprocessed_folder,
            "segmentations",
            f"%s_%s.npz",
        )
        self.log_path = os.path.join(
            preprocessed_folder,
            f"%s_%s.log",
        )

    def __call__(self, dataset):
        self.dataset = dataset
        assert torch.cuda.is_available(), "GPU required for preprocessing"
        procs = []
        mp.set_start_method("spawn")
        for proc_type in self.proc_per_gpu:
            devices = self.devices * self.proc_per_gpu[proc_type]
            n_cpus = self.proc_per_cpu[proc_type]
            proc_per_machine = len(devices) + n_cpus
            proc_total = proc_per_machine * self.machines_total
            # GPUs
            for proc_id, dev_id in enumerate(devices):
                p = mp.Process(
                    target=self.preprocess_single_process,
                    args=(
                        proc_type,
                        self.idx * proc_per_machine + proc_id,
                        dev_id,
                        proc_total,
                    ),
                )
                p.start()
                procs.append(p)
            # CPUs
            for proc_id in range(n_cpus):
                p = mp.Process(
                    target=self.preprocess_single_process,
                    args=(
                        proc_type,
                        self.idx * proc_per_machine + len(devices) + proc_id,
                        "cpu",
                        proc_total,
                    ),
                )
                p.start()
                procs.append(p)
        for proc in procs:
            proc.join()

    def preprocess_single_process(self, proc_type, proc_id, dev_id, proc_total):
        correct_names = []
        if dev_id != "cpu":
            torch.cuda.set_device(  # https://github.com/pytorch/pytorch/issues/21819#issuecomment-553310128
                dev_id
            )
            device = f"cuda:{dev_id}"
        else:
            device = "cpu"
        processor = self.proc_types[proc_type](device=device)
        log_path = self.log_path % (proc_id, proc_type)
        self.check_path(log_path)
        with open(log_path, "w") as logfile:
            for idx in tqdm(range(len(self.dataset)), file=logfile):
                if idx % proc_total != proc_id:
                    continue
                image, image_name = self.dataset.get_image(idx)
                image_name = os.path.splitext(image_name)[0]
                data = processor(image)
                save_path = self.preprocessed_path % (image_name, proc_type)
                self.check_path(save_path)
                np.savez(save_path, **data)

    def check_path(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass
