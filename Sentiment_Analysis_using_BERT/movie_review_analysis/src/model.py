import config
import transformers
import torch.nn as nn

class BERTBasedUncased(nn.Module):
    def __init__(self):
        super(BERTBasedUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)