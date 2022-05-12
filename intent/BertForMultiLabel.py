import torch
from transformers import ElectraModel,AutoModel

class BertForMultiLabel(torch.nn.Module):
    def __init__(self,num_labels):
        super(BertForMultiLabel, self).__init__()
        self.model = ElectraModel.from_pretrained('NlpHUST/electra-large-vn')
        self.num_labels = num_labels
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        pooled_output = self.model(input_ids, attention_mask, token_type_ids)
        pooled_output = pooled_output[0][0,0,:]
        pooler_output = self.pre_classifier(pooled_output)
        pooler_output = torch.nn.ReLU()(pooler_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits.unsqueeze(dim=0)
