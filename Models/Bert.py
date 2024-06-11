from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.2, classes=13):
        super(BertClassifier, self).__init__()
        self.lin1 = nn.Linear(3000, 512)
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(768, classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        input_id = self.lin1(input_id.float())
        input_id = self.relu(input_id).long()
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.lin2(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer