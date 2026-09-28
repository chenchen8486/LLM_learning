import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
enc2 = tiktoken.get_encoding("o200k_base")

def show(name, text, e=enc):
    ids = e.encode(text)
    pieces = [e.decode([i]) for i in ids]
    print(f"[{name}] {text!r}")
    print(f"  tokens({len(ids)}): {pieces}")
    print(f"  ids: {ids}")

show("英文", "Tokenization is fun.")
show("中文", "分词是大语言模型的第一道工序。")
show("中文2", "我爱大语言模型")
show("数字整", "537")
show("数字长", "123456789")
show("数字长2", "1234567890")
show("中英对照-EN", "I love large language models.")
show("中英对照-ZH", "我爱大语言模型。")
show("emoji", "🚀 发射！")
show("空格", "hello  world")
show("罕见词", "SolidGoldMagikarp")
show("o200k中文", "分词是大语言模型的第一道工序。", enc2)
print("cl100k vocab:", enc.n_vocab, " o200k vocab:", enc2.n_vocab)
