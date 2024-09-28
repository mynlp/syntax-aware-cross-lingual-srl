from prettytable import PrettyTable
import torch.nn as nn
from models.gtn import GTN


class GTNClassifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gtn = GTN(opt)
        self.classifier = nn.Linear(opt.out_dim, opt.label_size)

    def forward(self, **kwargs):
        outputs, pooled_out = self.gtn(**kwargs)
        logits = self.classifier(outputs)

        return logits, pooled_out

    def count_parameters(self):
        total = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                if 'pretrained_word_emb' in name:
                    if self.opt.fine_tuned_we:
                        total.append(p.numel())
                else:
                    total.append(p.numel())
        return sum(total)

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ['Layer Name', 'Output Shape', 'Param #', 'Train?']
        table.align['Layer Name'] = 'l'
        table.align['Output Shape'] = 'r'
        table.align['Param #'] = 'r'
        table.align['Trainable'] = 'c'
        for name, parameters in self.named_parameters():
            is_trainable = 'Yes' if parameters.requires_grad else 'No'
            table.add_row([name, str(list(parameters.shape)), parameters.numel(), is_trainable])
        return table
