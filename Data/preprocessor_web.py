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
from webdataset.handlers import warn_and_continue
from time import time, sleep
import json
import shutil
import fsspec
import queue


class WebPreprocessor:
    proc_types = {"panoptic": Detectron2, "human": HumanParts, "face": HumanFace}

    def __init__(
        self,
        preprocessed_folder,
        output_folder,
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
        self.output_folder = output_folder
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
            f"%s\\%s_%s.log",
        )

    def __call__(self, dataset):
        self.dataset = dataset
        assert torch.cuda.is_available(), "GPU required for preprocessing"
        procs = []
        mp.set_start_method("spawn")
        ready_queue = mp.Queue()
        proc_type_locks = {proc_type: mp.Value("b", 0) for proc_type in self.proc_types}
        proc_per_machine = 0
        for value in self.proc_per_gpu:
            proc_per_machine += len(value)
        for proc_type in self.proc_per_gpu:
            devices = len(self.proc_per_gpu[proc_type])
            n_cpus = self.proc_per_cpu[proc_type]
            proc_total_type = devices * self.machines_total
            # GPUs
            for proc_id, dev_id in enumerate(self.proc_per_gpu[proc_type]):
                p = mp.Process(
                    target=self.preprocess_single_process,
                    args=(
                        proc_type,
                        self.idx * devices + proc_id,
                        dev_id,
                        proc_total_type,
                        ready_queue,
                        proc_type_locks[proc_type]
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

        self.repacker_process(ready_queue, proc_per_machine, proc_type_locks)
        for proc in procs:
            proc.join()

    def preprocess_single_process(self, proc_type, proc_id, dev_id, proc_total, ready_queue, proc_type_lock):
        os.environ["RANK"] = str(proc_id)
        os.environ["TYPE"] = proc_type
        os.environ["WORLD_SIZE"] = str(proc_total)
        ready_queue.put("%s/%s/Init/%s" %(proc_id, proc_total, proc_type))
        dataset = hydra.utils.instantiate(self.dataset, ready_queue=ready_queue)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        correct_names = []
        torch.set_num_threads(15)
        if dev_id != "cpu":
            torch.cuda.set_device(  # https://github.com/pytorch/pytorch/issues/21819#issuecomment-553310128
                dev_id
            )
            device = f"cuda:{dev_id}"
        else:
            device = "cpu"
        processor = self.proc_types[proc_type](device=device)
        log_path = self.log_path % (proc_id, proc_total, proc_type)
        self.check_path(log_path)
        with open(log_path, "w") as logfile:
            x = 0
            t = time()
            times = []
            for batch in tqdm(dataloader, file=logfile):
                if proc_type_lock.value:
                    print("Waiting slower preprocessors ", proc_type)
                    while proc_type_lock.value:
                        sleep(0.1)
                    print("Waiting released", proc_type)
                times.append(str(time()-t))
                t = time()
                imgnames, tarnames, images = batch
                times.append(str(time()-t))
                t = time()
                batched_data = processor(images*255.)
                times.append(str(time()-t))
                t = time()
                #print(proc_id, proc_total, imgnames, tarnames)
                #print(batched_data["box_things"])
                for i in range(len(imgnames)):
                    tarname, imgname = tarnames[i], imgnames[i]
                    save_path = self.preprocessed_path % (tarname, imgname, proc_type)
                    self.check_path(save_path)
                    data = {key: batched_data[key][i] for key in batched_data}
                    np.savez(save_path, **data)
                times.append(str(time()-t))
                t = time()
                #ready_queue.put("%s/%s/info/%s" % (proc_id, proc_type, ",".join(times)))
                times = []
        ready_queue.put("%s/%s/done/und" % (str(proc_id) +"-"+ proc_type, proc_type))

    def repacker_process(self, ready_queue, proc_per_machine, proc_type_locks):
        proc_done = 0
        procs = []
        max_repackings = 20
        repackings_queue = queue.Queue()
        repacking_done = mp.Queue()
        info = {"repackings": 0}
        progress = {"panoptic": 0, "human": 0, "face": 0}
        while proc_done < proc_per_machine:
            command = ready_queue.get()
            print("Got", command)
            proc_id, worker, state, tarname = command.split("/")
            if state == "done":
                proc_done += 1
                for worker in info[proc_id]:
                    tarname = info[proc_id][worker]
                    info[tarname] = 0 if tarname not in info else info[tarname]
                    info[tarname] += 1
                    if info[tarname] == 3:
                        repackings_queue.put(tarname)
                        #self.repack_single_tar(tarname)
            elif state == "started":
                if proc_id not in info:
                    info[proc_id] = {}
                info[proc_id][worker] = tarname 
            elif state == "processed":
                info[tarname] = 0 if tarname not in info else info[tarname]
                info[tarname] += 1
                proc_type = proc_id.split("-")[1]
                progress[proc_type] += 1
                if info[tarname] == 3:
                    repackings_queue.put(tarname)
                    #self.repack_single_tar(tarname)
                if progress[proc_type] - np.min(list(progress.values())) > 30:
                    proc_type_locks[proc_type].value = 1
                if progress[proc_type] == np.min(list(progress.values())):
                    for t in proc_type_locks:
                        proc_type_locks[t].value = 0
                

            elif state== "info":
                continue

            while repacking_done.qsize() > 0:
                msg = repacking_done.get()
                with open("info.log", "a") as f:
                    f.write(msg+"\n")
                info["repackings"] -= 1

            # Repack original and processed data to new tar
            while info["repackings"] < max_repackings and repackings_queue.qsize() > 0:
                tarname = repackings_queue.get()
                print("Started repacking", tarname)
                p = mp.Process(
                    target=self.repack_single_tar,
                    args=(
                        tarname,
                        repacking_done,
                    ),
                )
                p.start()
                procs.append(p)
                info["repackings"] += 1

            
            with open("info.state", "w") as f:
                line = json.dumps(info, indent=3, sort_keys=True)
                f.write(str(line))
            with open("info.log", "a") as f:
                f.write(command+"\n")



        for p in procs:
            p.join()
        print("Processed all data!")

    def repack_single_tar(self, tarname, repacking_done):
        if os.path.isdir(self.dataset.root):
            root = self.datatset.root
        else:
            root = os.path.dirname(self.dataset.root)
        old_data = WebDataset(os.path.join(root, tarname), handler=warn_and_continue)
        output_folder = "s3://s-mas/" + self.output_folder
        fs, output_path = fsspec.core.url_to_fs(output_folder)
        tar_fd = fs.open(f"{output_path}/{tarname.split(' ')[0]}", "wb")
        new_data = TarWriter(tar_fd)
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
        print("Finished repacking", tarname)
        shutil.rmtree(os.path.join(self.preprocessed_folder, "untars", tarname))
        new_data.close()
        repacking_done.put("Finished repacking "+ tarname)



    def check_path(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass
