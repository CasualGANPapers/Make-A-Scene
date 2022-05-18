import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from .preprocessors import Detectron2
from .preprocessors import HumanParts
from .preprocessors import HumanFace
from tqdm import tqdm
import numpy as np
import hydra
from webdataset import WebDataset, TarWriter


class WebPreprocessor:
    proc_types = {"panoptic": Detectron2, "human": HumanParts, "face": HumanFace}

    def __init__(
        self,
        preprocessed_folder,
        proc_per_gpu=None,
        proc_per_cpu=None,
        devices=None,
        machine_idx=0,
        machines_total=1,
        batch_size=5,
        num_workers=2
    ):
        self.idx = machine_idx
        self.machines_total = machines_total
        self.devices = list(devices)
        self.proc_per_gpu = proc_per_gpu
        self.proc_per_cpu = proc_per_cpu
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessed_folder = preprocessed_folder
        self.preprocessed_path = os.path.join(
            preprocessed_folder,
            "untars",
            f"%s/%s_%s.npz",
        )
        self.repacked_path = os.path.join(
            preprocessed_folder,
            "tars",
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
        ready_queue = mp.Queue()
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
                        ready_queue,
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
                        ready_queue,
                    ),
                )
                p.start()
                procs.append(p)

        self.repacker_process(ready_queue, proc_per_machine)
        for proc in procs:
            proc.join()

    def preprocess_single_process(self, proc_type, proc_id, dev_id, proc_total, ready_queue):
        os.environ["RANK"] = str(proc_id)
        os.environ["WORLD_SIZE"] = str(proc_total)
        dataset = hydra.utils.instantiate(self.dataset, ready_queue=ready_queue)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
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
            x = 0
            for batch in tqdm(dataloader, file=logfile):
                imgnames, tarnames, images = batch
                batched_data = processor(images)
                #print(proc_id, proc_total, imgnames, tarnames)
                #print(batched_data["box_things"])
                for i in range(len(imgnames)):
                    tarname, imgname = tarnames[i], imgnames[i]
                    save_path = self.preprocessed_path % (tarname, imgname, proc_type)
                    self.check_path(save_path)
                    data = {key: batched_data[key][i] for key in batched_data}
                    np.savez(save_path, **data)
        ready_queue.put("%s/und/done/und" % proc_id)

    def repacker_process(self, ready_queue, proc_per_machine):
        proc_done = 0
        info = {str(i): {} for i in  range(proc_per_machine)}
        while proc_done < proc_per_machine:
            command = ready_queue.get()
            proc_id, worker, state, tarname = command.split("/")
            info[tarname] = 0 if tarname not in info else info[tarname]
            if state == "done":
                proc_done += 1
                for worker in info[proc_id]:
                    tarname = info[proc_id][worker]
                    info[tarname] += 1
                    if info[tarname] == 3:
                        self.repack_single_tar(tarname)
            elif state == "started":
                info[proc_id][worker] = tarname 
            elif state == "processed":
                info[tarname] += 1
                if info[tarname] == 3:
                    self.repack_single_tar(tarname)


        print("Processed all data!")

    def repack_single_tar(self, tarname):
        old_data = WebDataset(os.path.join(self.dataset.root, tarname))
        new_path = os.path.join(self.repacked_path, tarname)
        self.check_path(new_path)
        new_data = TarWriter(new_path)
        for sample in old_data:
            new_sample = {}
            imgname = new_sample["__key__"] = sample["__key__"]
            new_sample["jpg"] = sample["jpg"]
            new_sample["txt"] = sample["txt"]

            data_face = np.load(self.preprocessed_path % (tarname, imgname, "face"), allow_pickle=True)
            data_human = np.load(self.preprocessed_path % (tarname, imgname, "human"))
            data_panoptic = np.load(self.preprocessed_path % (tarname, imgname, "panoptic"))

            data_merged = {}
            data_merged["seg_panoptic"] = data_panoptic["seg_panoptic"]
            data_merged["edge_panoptic"] = data_panoptic["edges"]
            data_merged["box_things"] = data_panoptic["box_things"]
            data_merged["seg_human"] = data_human["seg_human"]
            data_merged["edge_human"] = data_human["edges"]
            data_merged["seg_face"] = data_face["seg_face"]
            data_merged["box_face"] = data_face["box_face"]
            new_sample["npz"] = data_merged

            new_data.write(new_sample)
        new_data.close()



    def check_path(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass
