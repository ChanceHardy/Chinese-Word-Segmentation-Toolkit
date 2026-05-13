import torch
import torch.nn as nn
import pickle
import os

# 严格对齐参数
EMBEDDING_DIM, CNN_FILTERS = 128, 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tag2id = {"B": 0, "M": 1, "E": 2, "S": 3, "<START>": 4, "<STOP>": 5}
id2tag = {v: k for k, v in tag2id.items()}

class IDCNN(nn.Module):
    def __init__(self, input_size, filters, dilations):
        super(IDCNN, self).__init__()
        self.linear_in = nn.Linear(input_size, filters)
        self.layers = nn.ModuleList([nn.Conv1d(filters, filters, kernel_size=3, padding=d, dilation=d) for d in dilations])
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.linear_in(x).unsqueeze(0).transpose(1, 2)
        for conv in self.layers: x = self.relu(conv(x))
        return x.squeeze(0).transpose(0, 1)

class IDCNN_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, filters):
        super(IDCNN_CRF, self).__init__()
        self.tag2id, self.tagset_size = tag2id, len(tag2id)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.idcnn = IDCNN(embedding_dim, filters, dilations=[1, 1, 2, 4])
        self.hidden2tag = nn.Linear(filters, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

    def _viterbi_decode(self, feats):
        backpointers = []
        vvars = torch.full((1, self.tagset_size), -10000.).to(device); vvars[0][self.tag2id["<START>"]] = 0
        for feat in feats:
            next_tag_var = vvars.expand(self.tagset_size, self.tagset_size) + self.transitions
            best_tag_ids = torch.argmax(next_tag_var, dim=1)
            vvars = (next_tag_var[range(self.tagset_size), best_tag_ids] + feat).view(1, -1)
            backpointers.append(best_tag_ids.tolist())
        terminal_var = vvars + self.transitions[self.tag2id["<STOP>"]]
        best_tag_id = torch.argmax(terminal_var).item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]; best_path.append(best_tag_id)
        return terminal_var[0][best_tag_id], best_path[::-1][1:]

    def forward(self, sentence):
        feats = self.hidden2tag(self.idcnn(self.word_embeds(sentence)))
        return self._viterbi_decode(feats)

if __name__ == "__main__":
    if not (os.path.exists("vocab_idcnn.pkl") and os.path.exists("idcnn_crf.pth")):
        print("错误：未找到配套文件！")
        exit()
    with open("vocab_idcnn.pkl", 'rb') as f: w2i = pickle.load(f)
    model = IDCNN_CRF(len(w2i), tag2id, EMBEDDING_DIM, CNN_FILTERS).to(device)
    model.load_state_dict(torch.load("idcnn_crf.pth", map_location=device))
    model.eval()
    print("--- 模型加载成功 ---")

    while True:
        text = input("\n请输入测试句子 (q退出): ").strip()
        if text.lower() == 'q': break
        if not text: continue
        with torch.no_grad():
            char_in = torch.tensor([w2i.get(x, 0) for x in list(text)], dtype=torch.long).to(device)
            _, tags = model(char_in)
            res, word = [], ""
            for char, tid in zip(list(text), tags):
                tag = id2tag[tid]
                if tag == 'S': res.append(char)
                elif tag == 'B': word = char
                elif tag == 'M': word += char
                elif tag == 'E': word += char; res.append(word); word = ""
            if word: res.append(word)
            print("结果: " + " / ".join(res))