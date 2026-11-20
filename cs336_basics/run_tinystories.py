import pickle

import psutil

from cs336_basics.train_bpe import train_bpe

if __name__ == '__main__':
    token_bytes_by_id, merges = train_bpe(
        r"E:\Codex\CS336\assignment1-basics\data\owt_train.txt\owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )

    main_memory = psutil.Process().memory_info()

    print(f"主进程峰值工作集: {main_memory.peak_wset / 1073741824} GiB")

    with open(r"E:\Codex\CS336\assignment1-basics\owt_vocab_32000.pkl", "xb") as f:
        pickle.dump(token_bytes_by_id, f)

    with open(r"E:\Codex\CS336\assignment1-basics\owt_merges_32000.pkl", "xb") as f:
        pickle.dump(merges, f)
