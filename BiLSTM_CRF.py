import torch
import torch.nn as nn
import torch.optim as optim
import os, pickle, time, random

# ==========================================
# 1. 指标计算函数 (完全符合课件要求)
# ==========================================
def get_metrics(true_tags, pred_tags):
    """ 计算精确度、召回率、F1 """
    tp, fp, fn = 0, 0, 0
    # 将标签序列转为区间段 (B...E)
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
# 2. 核心模型：向量化 BiLSTM-CRF (2层+Dropout)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TAG, STOP_TAG = "<START>", "<STOP>"
tag2id = {"B": 0, "M": 1, "E": 2, "S": 3, START_TAG: 4, STOP_TAG: 5}
id2tag = {v: k for k, v in tag2id.items()}

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.tag2id, self.tagset_size = tag2id, len(tag2id)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 增加到2层，加入Dropout增强鲁棒性
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, 
                            bidirectional=True, dropout=0.3)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
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
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embeds)
        feats = self.hidden2tag(lstm_out.view(len(sentence), -1))
        forward_score = self._forward_alg(feats)
        tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype=torch.long).to(device), tags])
        gold_score = torch.zeros(1).to(device)
        for i, feat in enumerate(feats):
            gold_score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        gold_score += self.transitions[self.tag2id[STOP_TAG], tags[-1]]
        return forward_score - gold_score

    def forward(self, sentence):
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embeds)
        feats = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return self._viterbi_decode(feats)

# ==========================================
# 3. 数据与主流程
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
    print(f"模式: BiLSTM-CRF | 设备: {device}")
    
    # --- 修改处：不再读取 pku_test.utf8，而是划分 training 文件 ---
    all_data = load_data("pku_training.utf8")
    random.seed(42) # 固定随机种子，确保每次运行划分一致
    random.shuffle(all_data)
    
    split = int(len(all_data) * 0.9)
    train_data = all_data[:split]
    test_data = all_data[split:] # 这 10% 是带空格的，可以算出真实的 F1
    # ---------------------------------------------------------

    w2i = {"<UNK>": 0}
    for c, _ in train_data:
        for char in c:
            if char not in w2i: w2i[char] = len(w2i)

    with open("vocab_bilstm.pkl", "wb") as f:
        pickle.dump(w2i, f)
    print(f"词表已保存，大小为: {len(w2i)}")
    
    model = BiLSTM_CRF(len(w2i), tag2id, 128, 256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"训练集: {len(train_data)} | 测试集: {len(test_data)}")
    for epoch in range(5):
        model.train()
        total_loss, start = 0, time.time()
        for i, (c, t) in enumerate(train_data):
            model.zero_grad()
            sentence_in = torch.tensor([w2i.get(x, 0) for x in c], dtype=torch.long).to(device)
            targets = torch.tensor([tag2id[x] for x in t], dtype=torch.long).to(device)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward(); optimizer.step(); total_loss += loss.item()
            if (i+1) % 2000 == 0:
                print(f"Epoch {epoch+1}, {i+1}/{len(train_data)}, Loss: {total_loss/(i+1):.4f}")
        
        # 每一轮结束进行评估
        model.eval()
        true_tags, pred_tags = [], []
        with torch.no_grad():
            for c, t in test_data:
                in_t = torch.tensor([w2i.get(x, 0) for x in c], dtype=torch.long).to(device)
                _, tags = model(in_t)
                true_tags.append([tag2id[x] for x in t])
                pred_tags.append(tags)
        
        p, r, f1 = get_metrics(true_tags, pred_tags)
        print(f"\n>>> Epoch {epoch+1} 评估: P={p:.4f}, R={r:.4f}, F1={f1:.4f} (耗时: {time.time()-start:.1f}s)\n")

    torch.save(model.state_dict(), "bilstm_crf.pth")