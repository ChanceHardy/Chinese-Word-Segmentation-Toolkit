import torch
import torch.nn as nn
import torch.optim as optim
import os, pickle, time, random

# ==========================================
# 1. 指标计算函数 (与 BiLSTM 版本 100% 一致)
# ==========================================
def get_metrics(true_tags, pred_tags):
    tp, fp, fn = 0, 0, 0
    def get_intervals(tags):
        intervals = set()
        start = -1
        for i, tag_id in enumerate(tags):
            tag = id2tag[tag_id]
            if tag == 'S': intervals.add((i, i))
            elif tag == 'B': start = i
            elif tag == 'E' and start != -1:
                intervals.add((start, i))
                start = -1
        return intervals
    for gold, pred in zip(true_tags, pred_tags):
        gold_inter = get_intervals(gold)
        pred_inter = get_intervals(pred)
        tp += len(gold_inter & pred_inter)
        fp += len(pred_inter - gold_inter)
        fn += len(gold_inter - pred_inter)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

# ==========================================
# 2. 核心模型：IDCNN-CRF (心脏移植：换成膨胀卷积)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TAG, STOP_TAG = "<START>", "<STOP>"
tag2id = {"B": 0, "M": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}
id2tag = {v: k for k, v in tag2id.items()}

# 模型超参数 (Filters=256 对应 BiLSTM 的 Hidden=256)
EMBEDDING_DIM = 128
CNN_FILTERS = 256
EPOCHS = 4
MODEL_PATH = "idcnn_crf.pth"
VOCAB_PATH = "vocab_idcnn.pkl"

class IDCNN(nn.Module):
    def __init__(self, input_size, filters, dilations):
        super(IDCNN, self).__init__()
        self.linear_in = nn.Linear(input_size, filters)
        self.layers = nn.ModuleList([
            nn.Conv1d(filters, filters, kernel_size=3, padding=d, dilation=d) 
            for d in dilations
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [seq_len, embed_dim] -> [1, filters, seq_len]
        x = self.linear_in(x).unsqueeze(0).transpose(1, 2)
        for conv in self.layers:
            x = self.relu(conv(x))
            x = self.dropout(x)
        # -> [seq_len, filters]
        return x.squeeze(0).transpose(0, 1)

class IDCNN_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, filters):
        super(IDCNN_CRF, self).__init__()
        self.tag2id, self.tagset_size = tag2id, len(tag2id)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 使用 4 层膨胀卷积，膨胀率分别为 1, 1, 2, 4
        self.idcnn = IDCNN(embedding_dim, filters, dilations=[1, 1, 2, 4])
        self.hidden2tag = nn.Linear(filters, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag2id[START_TAG], :] = -10000
        self.transitions.data[:, tag2id[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        alphas[0][self.tag2id[START_TAG]] = 0.
        for feat in feats:
            alphas_t = alphas.expand(self.tagset_size, self.tagset_size) + \
                       self.transitions + feat.view(self.tagset_size, 1).expand(self.tagset_size, self.tagset_size)
            alphas = torch.logsumexp(alphas_t, dim=1).view(1, -1)
        return torch.logsumexp(alphas + self.transitions[self.tag2id[STOP_TAG]], dim=1)

    def _viterbi_decode(self, feats):
        backpointers = []
        vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        vvars[0][self.tag2id[START_TAG]] = 0
        for feat in feats:
            next_tag_var = vvars.expand(self.tagset_size, self.tagset_size) + self.transitions
            best_tag_ids = torch.argmax(next_tag_var, dim=1)
            vvars = (next_tag_var[range(self.tagset_size), best_tag_ids] + feat).view(1, -1)
            backpointers.append(best_tag_ids.tolist())
        terminal_var = vvars + self.transitions[self.tag2id[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        return terminal_var[0][best_tag_id], best_path[::-1][1:]

    def neg_log_likelihood(self, sentence, tags):
        feats = self.hidden2tag(self.idcnn(self.word_embeds(sentence)))
        forward_score = self._forward_alg(feats)
        tags_start = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long).to(device), tags])
        gold_score = torch.zeros(1).to(device)
        for i, feat in enumerate(feats):
            gold_score += self.transitions[tags_start[i+1], tags_start[i]] + feat[tags_start[i+1]]
        gold_score += self.transitions[self.tag2id[STOP_TAG], tags_start[-1]]
        return forward_score - gold_score

    def forward(self, sentence):
        feats = self.hidden2tag(self.idcnn(self.word_embeds(sentence)))
        return self._viterbi_decode(feats)

# ==========================================
# 3. 数据处理逻辑 (完全对齐)
# ==========================================
def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip().split()
            if not w: continue
            c, t = [], []
            for word in w:
                if len(word)==1: c.append(word); t.append('S')
                else: c.extend(list(word)); t.extend(['B']+['M']*(len(word)-2)+['E'])
            data.append((c, t))
    return data

if __name__ == "__main__":
    print(f"模式: IDCNN-CRF | 设备: {device}")
    all_data = load_data("pku_training.utf8")
    random.seed(42); random.shuffle(all_data)
    split = int(len(all_data) * 0.9)
    train_data, test_data = all_data[:split], all_data[split:]

    w2i = {"<UNK>": 0}
    for c, _ in train_data:
        for char in c:
            if char not in w2i: w2i[char] = len(w2i)
    with open(VOCAB_PATH, "wb") as f: pickle.dump(w2i, f)
    print(f"词表已保存，大小为: {len(w2i)}")
    
    model = IDCNN_CRF(len(w2i), tag2id, EMBEDDING_DIM, CNN_FILTERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"训练集: {len(train_data)} | 测试集: {len(test_data)}")
    for epoch in range(EPOCHS):
        model.train(); total_loss, start = 0, time.time()
        for i, (c, t) in enumerate(train_data):
            model.zero_grad()
            sentence_in = torch.tensor([w2i.get(x, 0) for x in c], dtype=torch.long).to(device)
            targets = torch.tensor([tag2id[x] for x in t], dtype=torch.long).to(device)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward(); optimizer.step(); total_loss += loss.item()
            if (i+1) % 2000 == 0:
                print(f"Epoch {epoch+1}, {i+1}/{len(train_data)}, Loss: {total_loss/(i+1):.4f}")
        
        model.eval(); true_tags, pred_tags = [], []
        with torch.no_grad():
            for c, t in test_data:
                in_t = torch.tensor([w2i.get(x, 0) for x in c], dtype=torch.long).to(device)
                _, tags = model(in_t); true_tags.append([tag2id[x] for x in t]); pred_tags.append(tags)
        
        p, r, f1 = get_metrics(true_tags, pred_tags)
        print(f"\n>>> Epoch {epoch+1} 评估: P={p:.4f}, R={r:.4f}, F1={f1:.4f} (耗时: {time.time()-start:.1f}s)\n")
    torch.save(model.state_dict(), MODEL_PATH)