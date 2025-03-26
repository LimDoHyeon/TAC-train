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
from torch.optim.lr_scheduler import StepLR

import torchaudio
from itertools import chain

from iFaSNet import iFaSNet
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
    torch.cuda.set_device(rank)


###############################################################################
# 2. Exploring Data Path & Defining Dataset
###############################################################################
def collect_sample_paths(root_dir, mic_list):
    """
    root_dir 아래에 mic 폴더(예: '2mic', '4mic', '6mic')를 검색하고,
    해당 폴더 안의 모든 sampleX 하위 폴더 경로를 리스트업한다.

    mic_list: [2,4,6]처럼, 사용할 마이크 개수들을 명시한 리스트
    예) root_dir = "TAC/audio/output/MC_Libri_adhoc/train"
    """
    all_sample_paths = []
    # 예) root_dir/train -> 2mic, 4mic, 6mic 등 디렉토리 탐색
    # os.walk(root_dir) -> (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath)  # e.g. '2mic' or 'sample1'

        # 마이크 폴더인지 확인 (e.g. '2mic', '4mic', '6mic')
        # mic_list = [2,4,6] -> 폴더명에서 숫자만 파싱
        for m in mic_list:
            if base == f"{m}mic":
                # 해당 디렉토리는 m개의 마이크 폴더
                # 이제 이 안의 하위 디렉토리(sample1, sample2, ...)가 실제 샘플 폴더
                # 하지만 os.walk는 재귀적으로 더 내려가므로, 별도 처리 없이 그냥 진행하면
                # 아래 루프에서 'sampleN' 디렉토리도 잡힌다.
                pass
        # 'sampleN' 폴더라면, 상위 폴더가 'Xmic' 형태인지 확인
        # -> dirpath: ".../train/2mic/sample1"
        # 상위 폴더 이름 mic_base: ".../train/2mic"
        mic_base = os.path.dirname(dirpath)
        mic_folder = os.path.basename(mic_base)  # e.g. '2mic' or '6mic'

        # 폴더 이름에서 mic 수 파싱
        if "mic" in mic_folder:
            try:
                k = int(mic_folder.replace("mic", ""))  # e.g. "2" -> 2
            except:
                continue

            # k가 mic_list 내에 있으면 이 폴더/sampleN을 샘플 경로로 인정
            if k in mic_list:
                # 실제로 이 dirpath가 sample 폴더인지 확인 (파일이 mixture_mic1.wav 등 있는지)
                # os.walk는 파일명도 주지만, 일단 폴더 판단만 먼저
                # 별도 로직 없이 dirpath 자체를 담되, 실제 Dataset에서 유효성 판단할 수 있음
                if dirpath not in all_sample_paths:
                    all_sample_paths.append(dirpath)

    return all_sample_paths


class MultiMicAudioDataset(Dataset):
    """
    Zero-padding 방식을 통해 (k, length) -> (K_max, length) 형태로 확장.
    K_max = 6 (ad-hoc: 2, 4, 6mic 중 가장 큰 값).
    타겟(spk1, spk2)은 기존대로 (2, length) 만 쓰고,
    실제 사용 채널 수 'k'는 num_mic_info로 반환.
    """

    def __init__(self, sample_paths, sr=16000, max_mic=6):
        super().__init__()
        self.sample_paths = sample_paths
        self.sr = sr
        self.max_mic = max_mic  # e.g. 6

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]
        # Parsing the number of mics(k) in upper folders' name(ex: "2mic" -> k=2)
        mic_folder = os.path.basename(os.path.dirname(sample_dir))  # e.g. '2mic', '4mic', '6mic'
        k = int(mic_folder.replace("mic", ""))  # num of mics

        # ---------------------------
        # 1) Loading Mixture
        # ---------------------------
        # mixture는 shape (k, length)
        mixture_list = []
        for mic_idx in range(1, k + 1):
            mix_path = os.path.join(sample_dir, f"mixture_mic{mic_idx}.wav")
            wave, sr_ = torchaudio.load(mix_path)  # (1, length)
            if sr_ != self.sr:
                raise ValueError(f"Sample rate mismatch: got {sr_}, expected {self.sr}")
            mixture_list.append(wave[0])  # shape: (length,)

        mixture = torch.stack(mixture_list, dim=0)  # (k, length)

        # ---------------------------
        # 2) Zero-padding (max_mic, length)
        # ---------------------------
        # ex: if k=2, extend (2, length) into (6, length)
        # Otherwise(ex: 3~6) will be 0
        # -> DataLoader의 default_collate로 (batch, 6, length) 스택 가능
        padded_mix = torch.zeros(self.max_mic, mixture.size(1))  # (6, length)
        padded_mix[:k, :] = mixture  # 앞쪽 k개 채널만 실제 음성

        # ---------------------------
        # 3) Target source (spk1, spk2)
        # ---------------------------
        # ref mic1 only for prediction label -> shape: (2, length)
        tgt_list = []
        for spk_idx in [1, 2]:
            spk_path = os.path.join(sample_dir, f"spk{spk_idx}_mic1.wav")
            spk_wave, sr_ = torchaudio.load(spk_path)
            if sr_ != self.sr:
                raise ValueError(f"Sample rate mismatch: got {sr_}, expected {self.sr}")
            tgt_list.append(spk_wave[0])  # (length,)
        tgt = torch.stack(tgt_list, dim=0)  # (2, length)

        return padded_mix, tgt, k  # (6, length), (2, length), num of valid channels


###############################################################################
# 3. Movdel, Loss, Evaluation
###############################################################################
def create_model():
    """
    Create iFaSNet model
    """
    model = iFaSNet(
        enc_dim=64,
        feature_dim=64,
        hidden_dim=128,
        layer=6,
        segment_size=24,
        nspk=2,
        win_len=16,
        context_len=16,
        sr=16000
    )
    return model


def separation_loss(est_sources, target_sources):
    """
    est_sources: (batch, nspk, length)
    target_sources: (batch, nspk, length)
    -> Maximize SI-SNR usring batch_SDR_torch => Minimize Loss
    """
    sdr = batch_SDR_torch(est_sources, target_sources, return_perm=False)  # (batch,)
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

        # forward
        est_sources = model(mix, num_mic_tensor)
        loss = separation_loss(est_sources, tgt)
        total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


@torch.no_grad()
def test_model(rank, model, test_loader):
    """
    최종 에폭이 끝난 후, best 모델 로드한 뒤 test set SISNR 평가
    """
    model.eval()
    total_sdr = 0.0
    count = 0
    for mix, tgt, k in test_loader:
        mix = mix.cuda(rank, non_blocking=True)
        tgt = tgt.cuda(rank, non_blocking=True)
        num_mic_tensor = k.to(mix.device)

        est_sources = model(mix, num_mic_tensor)
        # SISNR 계산
        sdr_vals = batch_SDR_torch(est_sources, tgt, return_perm=False)  # (batch,)
        total_sdr += sdr_vals.sum().item()
        count += sdr_vals.size(0)

    avg_sdr = total_sdr / count if count > 0 else 0.0
    return avg_sdr


###############################################################################
# 4. Training Loop (DDP)
###############################################################################
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
        est_sources = model(mix, num_mic_tensor)  # (batch, nspk, length)
        loss = separation_loss(est_sources, tgt)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping (norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        if rank == 0:
            train_loader.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def main_worker(rank, world_size, args):
    ddp_setup(rank, world_size)
    set_seed(args.seed)

    # ------------------------------------------------------------------------
    # 1) Collecting sample paths for Train/Val/Test
    #    mic_list ex: [2,4,6]
    # ------------------------------------------------------------------------
    mic_list = args.mic_list  # shapes like [2,4,6]
    train_paths = collect_sample_paths(args.train_dir, mic_list)
    val_paths = collect_sample_paths(args.val_dir, mic_list)
    test_paths = collect_sample_paths(args.test_dir, mic_list)

    # ------------------------------------------------------------------------
    # 2) Dataset / DataLoader
    # ------------------------------------------------------------------------
    train_dataset = MultiMicAudioDataset(train_paths, sr=16000)
    val_dataset = MultiMicAudioDataset(val_paths, sr=16000)
    test_dataset = MultiMicAudioDataset(test_paths, sr=16000)

    # DDP Sampler
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

    # ------------------------------------------------------------------------
    # 3) Model, Optimizer
    # ------------------------------------------------------------------------
    model = create_model().cuda(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------------
    # 4) Training
    # ------------------------------------------------------------------------
    print('Data loaded completely. Train start.')

    # Decaying learning rate
    scheduler = StepLR(optimizer, step_size=2, gamma=0.98)

    # Early stopping
    no_improve_count = 0
    max_no_improve = 10  # stop condition

    # Load checkpoint
    start_epoch = 1
    best_loss = float("inf")
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        no_improve_count = checkpoint["no_improve_count"]
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}. best_loss={best_loss:.4f}, no_improve_count={no_improve_count}")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        log_csv = open(os.path.join(args.save_dir, "training_log.csv"), "w", newline="")
        csv_writer = csv.writer(log_csv)
        csv_writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(rank, model, train_loader, optimizer, epoch)

        # lr scheduler step
        scheduler.step()

        val_loss = None
        # Validate every 10 epochs
        if epoch % 10 == 0:
            val_loss = evaluate(rank, model, val_loader, epoch)
            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}")

                csv_writer.writerow([epoch, train_loss, val_loss, current_lr])
                log_csv.flush()

                # Best Model
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve_count = 0
                    torch.save(model.module.state_dict(),
                               os.path.join(args.save_dir, "best_model.pth"))
                else:
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        print(f"No improvement for {max_no_improve} checks. Early Stopping.")
                        break

                # Save checkpoint every 10 epochs
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "no_improve_count": no_improve_count
                }
                torch.save(checkpoint, checkpoint_path)

        else:
            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, lr={current_lr:.6f}")

    if rank == 0:
        log_csv.close()
        print(f"Training finished. Best val loss: {best_loss:.4f}")

        # Test with best model
        best_model_path = os.path.join(args.save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            print("Loading best_model.pth for final test...")
            model.module.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        test_sampler.set_epoch(999)
        test_sdr = test_model(rank, model, test_loader)
        print(f"[TEST] SISNR: {test_sdr:.4f} dB")

    dist.destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2, help="사용할 GPU 수")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_dir", type=str,
                        default="/mnt/raid0/ldh/workspace/TAC/data/audio/output/MC_Libri_adhoc/train")
    parser.add_argument("--val_dir", type=str,
                        default="/mnt/raid0/ldh/workspace/TAC/data/audio/output/MC_Libri_adhoc/validation")
    parser.add_argument("--test_dir", type=str,
                        default="/mnt/raid0/ldh/workspace/TAC/data/audio/output/MC_Libri_adhoc/test")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # mic_list: list of mics you want to use (ad-hoc)
    # ex) --mic_list 2 4 6
    parser.add_argument("--mic_list", nargs="+", type=int, default=[2, 4, 6],
                        help="Ex) 2 4 6 for ad-hoc channels")

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "3333"

    mp.spawn(
        main_worker,
        args=(args.world_size, args),
        nprocs=args.world_size
    )


if __name__ == "__main__":
    main()
