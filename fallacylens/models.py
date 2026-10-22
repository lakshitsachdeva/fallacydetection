"""Model definitions: sentence-CNN, attention BiLSTM, and a pretrained Transformer."""
import torch
import torch.nn as nn


class FallacyCNN(nn.Module):
    """Kim-style sentence CNN (identical to the Phase 1 baseline)."""

    def __init__(self, vocab_size, n_classes, embed_dim=100, n_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, n_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x):
        e = self.embedding(x).transpose(1, 2)
        feats = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


class FallacyBiLSTM(nn.Module):
    """Bidirectional LSTM with additive attention pooling over time steps."""

    def __init__(self, vocab_size, n_classes, embed_dim=100, hidden=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)
        self.last_attention = None

    def forward(self, x):
        mask = x != 0
        h, _ = self.lstm(self.dropout(self.embedding(x)))
        scores = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)
        self.last_attention = alpha.detach()
        z = (alpha.unsqueeze(-1) * h).sum(dim=1)
        return self.fc(self.dropout(z))


def build_transformer(name, n_classes, label_names=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kwargs = {}
    if label_names:
        kwargs = {"id2label": dict(enumerate(label_names)),
                  "label2id": {l: i for i, l in enumerate(label_names)}}
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_classes, **kwargs)
    return tok, model
