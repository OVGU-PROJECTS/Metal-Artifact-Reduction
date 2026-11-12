"""CSV + TensorBoard logging for training/trainer.py (main process only).
  csv/train.csv:                 epoch-level summary + loss weights
  logs/loss_breakdown_val.csv:   calibrated, unweighted terms + image metrics
  logs/loss_breakdown_train.csv: same columns, on the train set
"""
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

EPOCH_HEADER = "epoch,train_loss,val_loss,lr,w_rmse,w_hole,w_image,w_edge,w_dc,w_perc\n"
BD_HEADER = (
    "epoch,rmse,hole,dc,img,edge,perc,ssim_aapm,"
    "ssim_trace,mae_near_metal,streak_edge_err\n"
)

@dataclass
class Logs:
    csv_path: object
    bd_val_csv: object
    bd_train_csv: object
    tb_writer: SummaryWriter

def init_logs(csv_dir, logs_dir, tb_dir, initial_epoch):
    """Open the CSVs (headers on a fresh run) and the TensorBoard writer."""
    csv_path = csv_dir / "train.csv"
    bd_val_csv = logs_dir / "loss_breakdown_val.csv"
    bd_train_csv = logs_dir / "loss_breakdown_train.csv"
    csv_files = [(csv_path, EPOCH_HEADER), (bd_val_csv, BD_HEADER), (bd_train_csv, BD_HEADER)]
    if initial_epoch > 0:
        for path, header in csv_files:
            if path.exists():
                with path.open() as f:
                    existing_header = f.readline().rstrip("\r\n")
                if existing_header != header.rstrip("\n"):
                    raise ValueError(
                        f"Cannot append to {path}: metric columns differ. "
                        "Preserve the historical log and resume into a new run directory."
                    )
    for path, header in csv_files:
        if initial_epoch == 0 or not path.exists():
            path.write_text(header)
    return Logs(csv_path, bd_val_csv, bd_train_csv, SummaryWriter(log_dir=str(tb_dir / "main")))


def write_epoch_row(csv_path, epoch, train_loss, val_loss, lr, weights):
    """append one epoch summary row. `epoch` is 0-based"""
    with open(csv_path, "a") as f:
        f.write(
            f"{epoch + 1},{train_loss},{val_loss},{lr},"
            f"{weights['rmse']},{weights['hole']},{weights['image']},"
            f"{weights['edge']},{weights['dc']},{weights['perc']}\n"
        )

def write_breakdown(bd, csv_path, tb_writer, tb_prefix, epoch):
    """append a per-term breakdown row and mirror every term to TensorBoard."""
    with open(csv_path, "a") as f:
        f.write(
            f"{epoch + 1},{bd['rmse']},{bd['hole']},{bd['dc']},"
            f"{bd['img']},{bd['edge']},{bd['perc']},"
            f"{bd['ssim_aapm']},{bd['ssim_trace']},{bd['mae_near_metal']},"
            f"{bd['streak_edge_err']}\n"
        )
    for term, v in bd.items():
        tb_writer.add_scalar(f"TermBreakdown/{tb_prefix}_{term}", v, epoch + 1)


def write_tb_epoch(tb_writer, epoch, train_loss, val_loss, lr, weights):
    """mirror the epoch summary (losses, LR, per-term weights) to TensorBoard"""
    tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
    tb_writer.add_scalar("Loss/val", val_loss, epoch + 1)
    tb_writer.add_scalar("LR", lr, epoch + 1)
    for term, w in weights.items():
        tb_writer.add_scalar(f"LossWeight/{term}", w, epoch + 1)
        
