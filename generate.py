"""generate.py — 从训练好的检查点加载 TransformerLM 并生成文本。"""

import argparse

import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generation import generate_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained TransformerLM checkpoint."
    )

    # === 必传：检查点与分词器文件 ===
    parser.add_argument(
        "--checkpoint-path", type=str, required=True,
        help="Path to the model checkpoint (.pt) saved by train.py",
    )
    parser.add_argument(
        "--vocab-path", type=str, required=True,
        help="Path to the vocab file, which stores a dict[int, bytes]. Its token_id_by_bytes attribute is the inverse index.",
    )
    parser.add_argument(
        "--merges-path", type=str, required=True,
        help="Path to the BPE merges file",
    )

    # === 生成控制参数 ===
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    # === 模型架构参数（必须与训练时保持一致） ===
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=320)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- 动作 2：初始化 ---
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    completed = checkpoint["iteration"]

    model.to(device)
    model.eval()

    # --- 动作 3：加载 Tokenizer ---
    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    # --- 动作 4：找 EOS 的 ID ---
    eos_token_id = tokenizer.token_id_by_bytes[b"<|endoftext|>"]

    # --- 动作 5：文本转 ID ---
    prompt_ids = tokenizer.encode(args.prompt)

    # --- 动作 6：调用生成 ---
    generated_ids = generate_tokens(
        model,
        prompt_ids,
        args.max_new_tokens,
        args.context_length,  # 默认为 64，与讲义一致
        eos_token_id,
        args.temperature,
        args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    # --- 动作 7：解码并打印 ---
    generated_text = tokenizer.decode(generated_ids)
    new_tokens = len(generated_ids) - len(prompt_ids)

    print(f"加载步数: {completed}")
    print(f"新增 token 数: {new_tokens}")
    print(f"生成的文本: {generated_text}")


if __name__ == "__main__":
    main()
