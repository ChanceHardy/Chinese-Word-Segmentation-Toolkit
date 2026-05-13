import torch
import torch.nn as nn
import pickle
import os
import time

# ==========================================
# 1. 个人信息配置 (请核对)
# ==========================================
STUDENT_ID = "2351577"
NAME = "张宸浩"
INPUT_FILE = "raw.txt"
OUTPUT_FILE = f"{STUDENT_ID}_{NAME}_pred.txt"

# ==========================================
# 2. 模型配置 (必须与训练时完全一致)
# ==========================================
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
        return None, best_path[::-1][1:]

    def forward(self, sentence):
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embeds)
        feats = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return self._viterbi_decode(feats)

# ==========================================
# 3. 推理与格式化逻辑
# ==========================================
def process_line(text, model, w2i):
    if not text.strip():
        return ""
    
    # 标点符号列表（确保标点单独切分）
    punctuations = set("，。！？；：、“”（）《》—…,.!?;:\"\'()<>")
    
    with torch.no_grad():
        char_in = torch.tensor([w2i.get(x, 0) for x in list(text)], dtype=torch.long).to(device)
        _, tags = model(char_in)
        
        res = []
        word = ""
        for i, (char, tid) in enumerate(zip(list(text), tags)):
            tag = id2tag[tid]
            
            # 强制逻辑：如果是标点符号，必须单独成词
            if char in punctuations:
                if word:
                    res.append(word)
                    word = ""
                res.append(char)
                continue
            
            if tag == 'S':
                if word: res.append(word); word = ""
                res.append(char)
            elif tag == 'B':
                if word: res.append(word)
                word = char
            elif tag == 'M':
                word += char
            elif tag == 'E':
                word += char
                res.append(word)
                word = ""
        if word: res.append(word)
        
        # 按照规范使用 "/" 分隔，标点后也要有 "/"
        # 结果末尾根据规范也需要有 "/"
        return "/".join(res) + "/"

if __name__ == "__main__":
    if not os.path.exists("vocab_bilstm.pkl") or not os.path.exists("bilstm_crf.pth"):
        print("错误：缺失模型文件或词表文件！")
        exit()

    # 加载资源
    with open("vocab_bilstm.pkl", 'rb') as f: w2i = pickle.load(f)
    model = BiLSTM_CRF(len(w2i), tag2id, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load("bilstm_crf.pth", map_location=device))
    model.eval()

    print(f"开始处理 {INPUT_FILE}...")
    
    start_time = time.time()
    output_lines = []
    
    # 逐行读取并分词
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            # 保持原句内容，仅去掉末尾换行符
            clean_line = line.rstrip('\n')
            segmented = process_line(clean_line, model, w2i)
            output_lines.append(segmented)
    
    end_time = time.time()
    runtime = end_time - start_time

    # 写入结果文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + "\n")
        # 规范要求：最后一行记录运行时间
        f.write(f"# runtime_seconds: {runtime:.3f}")

    print(f"成功！结果已保存至: {OUTPUT_FILE}")
    print(f"总运行时间: {runtime:.3f} 秒")