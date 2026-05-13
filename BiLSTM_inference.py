import torch
import torch.nn as nn
import pickle
import os

# 严格对齐配置
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tag2id = {"B": 0, "M": 1, "E": 2, "S": 3, "<START>": 4, "<STOP>": 5}
id2tag = {v: k for k, v in tag2id.items()}

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.tag2id, self.tagset_size = tag2id, len(tag2id)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, dropout=0.3)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
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
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embeds)
        feats = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return self._viterbi_decode(feats)

if __name__ == "__main__":
    if not (os.path.exists("vocab_bilstm.pkl") and os.path.exists("bilstm_crf.pth")):
        print("错误：未找到配套文件！")
        exit()

    with open("vocab_bilstm.pkl", 'rb') as f: w2i = pickle.load(f)
    model = BiLSTM_CRF(len(w2i), tag2id, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load("bilstm_crf.pth", map_location=device))
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