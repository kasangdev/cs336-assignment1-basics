import argparse
from contextlib import nullcontext
import numpy as np
import torch
from pathlib import Path
import time

from cs336_basics.transformer import TRANSFORMER_VARIANTS, TransformerLM
from cs336_basics.optimizers import AdamW
from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy
from cs336_basics.schedules import get_lr_cosine_schedule
from cs336_basics.nn_utils import gradient_clipping
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.experiment import save_config, append_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--valid-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=320)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--variant", type=str, choices=TRANSFORMER_VARIANTS, default="baseline")
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)

    return parser.parse_args()


def evaluate_loss(model: torch.nn.Module,
                  dataset: np.ndarray,
                  batch_size: int,
                  context_length: int,
                  device: str,
                  eval_batches: int,
                  use_bfloat16: bool = False) -> float:
    if use_bfloat16 and torch.device(device).type != "cuda":
        raise ValueError("bfloat16 evaluation requires a CUDA device.")
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
    )
    was_training = model.training
    model.eval()
    total_loss = 0.0
    try:
        with torch.no_grad():
            for batch_idx in range(eval_batches):
                inputs, targets = get_batch(dataset, batch_size, context_length, device)
                with autocast_context:
                    logits = model(inputs)
                loss_logits = logits.float() if use_bfloat16 else logits
                loss = cross_entropy(loss_logits, targets)
                total_loss += loss.item()
            mean_loss = total_loss / eval_batches
    finally:
        model.train(was_training)
    return mean_loss


def main():
    args = parse_args()
    if args.checkpoint_interval is not None:
        checkpoint_interval = args.checkpoint_interval
    else:
        checkpoint_interval = args.eval_interval
    device = torch.device(args.device)
    is_cuda = (device.type == "cuda")
    use_bfloat16 = (args.precision == "bfloat16")
    if use_bfloat16 and (not is_cuda or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 precision requires a CUDA GPU with BF16 support.")
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = np.load(args.train_path, mmap_mode="r")
    valid_data = np.load(args.valid_path, mmap_mode="r")
    model = TransformerLM(args.vocab_size,
                          args.context_length,
                          args.d_model,
                          args.num_layers,
                          args.num_heads,
                          args.d_ff, args.rope_theta,
                          device=args.device,
                          dtype=torch.float32,
                          variant=args.variant)
    betas = (args.beta1, args.beta2)
    optimizer = AdamW(model.parameters(),
                      lr=args.max_learning_rate,
                      betas=betas,
                      eps=args.eps,
                      weight_decay=args.weight_decay)
    model.train()
    cosine_cycle_iters = args.num_iterations - 1
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    config_path = checkpoint_dir / "config.json"
    metrics_path = checkpoint_dir / "metrics.jsonl"
    config = vars(args)
    save_config(config, config_path)
    start_iteration = 0
    if args.resume_path is not None:
        start_iteration = load_checkpoint(args.resume_path, model, optimizer)
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    for iteration in range(start_iteration, args.num_iterations):
        lr = get_lr_cosine_schedule(iteration, args.max_learning_rate, args.min_learning_rate, args.warmup_iters,
                                    cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)
        with autocast_context:
            logits = model(inputs)
        loss_logits = logits.float() if use_bfloat16 else logits
        loss = cross_entropy(loss_logits, targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        optimizer.step()
        completed = iteration + 1
        if completed % args.log_interval == 0:
            loss_value = loss.item()
            elapsed = time.perf_counter() - start_time
            record = {"split": "train", "iteration": completed, "elapsed_seconds": elapsed, "loss": loss_value,
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            append_metrics(record, metrics_path)
            print(f"迭代步数: {completed}, 损失: {loss_value:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.2e}")
        if completed % args.eval_interval == 0 or completed == args.num_iterations:
            valid_loss = evaluate_loss(model, valid_data, args.batch_size, args.context_length, args.device,
                                       args.eval_batches, use_bfloat16=use_bfloat16)
            elapsed = time.perf_counter() - start_time
            record = {"split": "validation", "iteration": completed, "elapsed_seconds": elapsed, "loss": valid_loss,
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            append_metrics(record, metrics_path)

            print(f"验证步数: {completed}, 验证损失: {valid_loss:.4f}")
        if completed % checkpoint_interval == 0 or completed == args.num_iterations:
            checkpoint_path = checkpoint_dir / f"checkpoint_{completed}.pt"
            save_checkpoint(model, optimizer, completed, checkpoint_path)

    if is_cuda:
        torch.cuda.synchronize(device)
        device_name = torch.cuda.get_device_name(device)
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        device_name = "CPU"
        peak_allocated_mib = 0.0
        peak_reserved_mib = 0.0

    total_elapsed_seconds = time.perf_counter() - start_time
    processed_tokens = (args.num_iterations - start_iteration) * args.batch_size * args.context_length
    tokens_per_second = processed_tokens / total_elapsed_seconds
    print(f"设备名称: {device_name}")
    print(f"总耗时: {total_elapsed_seconds:.2f} 秒")
    print(f"处理 token 数: {processed_tokens}")
    print(f"吞吐量: {tokens_per_second:.2f} tokens/s")
    print(f"峰值显存 (分配): {peak_allocated_mib:.2f} MiB")
    print(f"峰值显存 (保留): {peak_reserved_mib:.2f} MiB")
    performance_path = checkpoint_dir / "performance.json"
    performance = {"device_name": device_name,
                   "batch_size": args.batch_size,
                   "context_length": args.context_length,
                   "total_elapsed_seconds": total_elapsed_seconds,
                   "processed_tokens": processed_tokens,
                   "tokens_per_second": tokens_per_second,
                   "peak_allocated_mib": peak_allocated_mib,
                   "peak_reserved_mib": peak_reserved_mib}
    save_config(performance, performance_path)


if __name__ == "__main__":
    main()
