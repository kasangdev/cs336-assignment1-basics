import pickle

with open(r"E:\Codex\CS336\assignment1-basics\owt_merges_32000.pkl", "rb") as f:
    merges = pickle.load(f)
with open(r"E:\Codex\CS336\assignment1-basics\owt_vocab_32000.pkl", "rb") as f:
    vocab = pickle.load(f)

print("词表长度： "    f"{len(vocab)}")
print("合并词长度： " f"{len(merges)}")
longest_token = max(vocab.values(), key=len)
decoded_token = longest_token.decode("utf-8", errors="replace")
print("最大词(原始bytes)： ", longest_token)
print("最大词(解码后)： ", decoded_token)
print("字节长度： ", len(longest_token))
print("字符长度： ", len(decoded_token))
