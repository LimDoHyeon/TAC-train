import os
import csv
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

import torchaudio
from itertools import chain
from FaSNet import FaSNet_origin, FaSNet_TAC
from utility.sdr import batch_SDR_torch


###############################################################################
# 1. Settings: random seed, DDP initialization
###############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    # GPU 설정은 main_worker에서 device_ids로 처리


###############################################################################
# 2. Exploring Data Path & Defining Dataset
###############################################################################
def collect_sample_paths(root_dir, mic_list):
    """
    root_dir 아래에 mic 폴더 (예: '2mic', '4mic', '6mic')를 검색하여,
    해당 폴더 안의 모든 sampleX 하위 폴더 경로를 리스트업.
    """
    all_sample_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath)  # 예: '2mic' 또는 'sample1'
        mic_base = os.path.dirname(dirpath)
        mic_folder = os.path.basename(mic_base)  # 예: '2mic' 또는 '6mic'
        if "mic" in mic_folder:
            try:
                k = int(mic_folder.replace("mic", ""))
            except:
                continue
            if k in mic_list:
                if dirpath not in all_sample_paths:
                    all_sample_paths.append(dirpath)
    return all_sample_paths


class MultiMicAudioDataset(Dataset):
    """
    Zero-padding 방식을 사용하여 (k, length) 형태의 입력을 (K_max, length)로 확장.
    K_max는 ad-hoc 환경에서 사용 가능한 최대 마이크 수 (예: 6).
    타겟(spk1, spk2)은 ref mic1만 사용하며, 실제 유효 채널 수 k는 num_mic_info로 반환.
    """

    def __init__(self, sample_paths, sr=16000, max_mic=6):
        super().__init__()
        self.sample_paths = sample_paths
        self.sr = sr
        self.max_mic = max_mic

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]
        # 상위 폴더 이름에서 마이크 수 파싱 (예: "2mic" -> k=2)
        mic_folder = os.path.basename(os.path.dirname(sample_dir))
        k = int(mic_folder.replace("mic", ""))

        # 1) Mixture 로드: (k, length)
        mixture_list = []
        for mic_idx in range(1, k + 1):
            mix_path = os.path.join(sample_dir, f"mixture_mic{mic_idx}.wav")
            wave, sr_ = torchaudio.load(mix_path)  # (1, length)
            if sr_ != self.sr:
                raise ValueError(f"Sample rate mismatch: got {sr_}, expected {self.sr}")
            mixture_list.append(wave[0])
        mixture = torch.stack(mixture_list, dim=0)  # (k, length)

        # 2) Zero-padding to (max_mic, length)
        padded_mix = torch.zeros(self.max_mic, mixture.size(1), dtype=mixture.dtype, device=mixture.device)
        padded_mix[:k, :] = mixture

        # 3) Target source: ref mic1만 사용 (spk1, spk2) → (2, length)
        tgt_list = []
        for spk_idx in [1, 2]:
            spk_path = os.path.join(sample_dir, f"spk{spk_idx}_mic1.wav")
            spk_wave, sr_ = torchaudio.load(spk_path)
            if sr_ != self.sr:
                raise ValueError(f"Sample rate mismatch: got {sr_}, expected {self.sr}")
            tgt_list.append(spk_wave[0])
        tgt = torch.stack(tgt_list, dim=0)  # (2, length)

        return padded_mix, tgt, k


###############################################################################
# 3. Model, Loss, Evaluation
###############################################################################
def create_model():
    """
    FaSNet 모델 생성.
    여기서는 FaSNet_origin을 사용합니다.
    필요에 따라 FaSNet_TAC로 변경 가능합니다.
    """
    model = FaSNet_TAC(
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        layer=6,
        segment_size=50,
        nspk=2,
        win_len=4,
        context_len=16,
        sr=16000
    )
    return model


def separation_loss(est_sources, target_sources):
    """
    SISNR 기반 Loss: Loss = - mean(SISNR)
    est_sources: (batch, nspk, length)
    target_sources: (batch, nspk, length)
    """
    sdr = batch_SDR_torch(est_sources, target_sources, return_perm=False)
    loss = - torch.mean(sdr)
    return loss


@torch.no_grad()
def evaluate(rank, model, val_loader, epoch):
    model.eval()
    total_loss = 0.0
    if rank == 0:
        val_loader = tqdm(val_loader, desc=f"Validate Epoch {epoch}", leave=False)
    for mix, tgt, k in val_loader:
        mix = mix.cuda(rank, non_blocking=True)
        tgt = tgt.cuda(rank, non_blocking=True)
        num_mic_tensor = k.to(mix.device)
        est_sources = model(mix, num_mic_tensor)
        loss = separation_loss(est_sources, tgt)
        total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


@torch.no_grad()
def test_model(rank, model, test_loader):
    model.eval()
    total_sdr = 0.0
    count = 0
    for mix, tgt, k in test_loader:
        mix = mix.cuda(rank, non_blocking=True)
        tgt = tgt.cuda(rank, non_blocking=True)
        num_mic_tensor = k.to(mix.device)
        est_sources = model(mix, num_mic_tensor)
        sdr_vals = batch_SDR_torch(est_sources, tgt, return_perm=False)
        total_sdr += sdr_vals.sum().item()
        count += sdr_vals.size(0)
    avg_sdr = total_sdr / count if count > 0 else 0.0
    return avg_sdr


def train_one_epoch(rank, model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    if rank == 0:
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for mix, tgt, k in train_loader:
        mix = mix.cuda(rank, non_blocking=True)
        tgt = tgt.cuda(rank, non_blocking=True)
        num_mic_tensor = k.to(mix.device)
        optimizer.zero_grad()
        est_sources = model(mix, num_mic_tensor)
        loss = separation_loss(est_sources, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if rank == 0:
            train_loader.set_postfix({"loss": f"{loss.item():.3f}"})
    avg_loss = total_loss / len(train_loader)
    return avg_loss


###############################################################################
# 4. Training Loop with Checkpoint Resume (DDP)
###############################################################################
def main_worker(rank, world_size, args):
    ddp_setup(rank, world_size)
    set_seed(args.seed)

    # 1) Collect sample paths for Train/Val/Test
    mic_list = args.mic_list  # e.g. [2,4,6]
    train_paths = collect_sample_paths(args.train_dir, mic_list)
    val_paths = collect_sample_paths(args.val_dir, mic_list)
    test_paths = collect_sample_paths(args.test_dir, mic_list)

    # 2) Dataset / DataLoader
    train_dataset = MultiMicAudioDataset(train_paths, sr=16000)
    val_dataset = MultiMicAudioDataset(val_paths, sr=16000)
    test_dataset = MultiMicAudioDataset(test_paths, sr=16000)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.num_workers, pin_memory=True
    )

    # 3) Model, Optimizer, and Checkpoint Resume
    model = create_model().cuda(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        if rank == 0:
            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    else:
        best_loss = float("inf")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        log_csv = open(os.path.join(args.save_dir, "training_log.csv"), "w", newline="")
        csv_writer = csv.writer(log_csv)
        csv_writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(rank, model, train_loader, optimizer, epoch)

        if epoch % 10 == 0:
            val_loss = evaluate(rank, model, val_loader, epoch)
            if rank == 0:
                print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                csv_writer.writerow([epoch, train_loss, val_loss])
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.module.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_loss": best_loss
                }
                torch.save(checkpoint, checkpoint_path)
        else:
            if rank == 0:
                print(f"[Epoch {epoch}] train_loss={train_loss:.4f}")

    if rank == 0:
        log_csv.close()
        print(f"Training finished. Best val loss: {best_loss:.4f}")

        if os.path.exists(os.path.join(args.save_dir, "best_model.pth")):
            print("Loading best_model.pth for final test...")
            model.module.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pth"),
                                                    map_location=lambda storage, loc: storage))
        test_sampler.set_epoch(999)
        test_sdr = test_model(rank, model, test_loader)
        print(f"[TEST] SISNR: {test_sdr:.4f} dB")

    dist.destroy_process_group()


###############################################################################
# 5. Main function (mp.spawn with DDP)
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2, help="사용할 GPU 수")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_FaSNet")
    parser.add_argument("--train_dir", type=str,
                        default="/mnt/raid0/ldh/workspace/TAC/data/audio/output/MC_Libri_adhoc/train")
    parser.add_argument("--val_dir", type=str,
                        default="/mnt/raid0/ldh/workspace/TAC/data/audio/output/MC_Libri_adhoc/validation")
    parser.add_argument("--test_dir", type=str,
                        default="/mnt/raid0/ldh/workspace/TAC/data/audio/output/MC_Libri_adhoc/test")
    parser.add_argument("--mic_list", nargs="+", type=int, default=[2, 4, 6],
                        help="Ex) 2 4 6 for ad-hoc channels")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "3333"

    mp.spawn(
        main_worker,
        args=(args.world_size, args),
        nprocs=args.world_size
    )


if __name__ == "__main__":
    main()
