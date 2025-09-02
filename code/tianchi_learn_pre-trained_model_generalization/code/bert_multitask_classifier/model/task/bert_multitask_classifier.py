import torch
import torch.nn as nn

from transformers import BertConfig, BertPreTrainedModel, BertModel

from dataclasses import dataclass


from ...common import logger
from ..classifier import MultiTaskClassifier


class BertMultitaskClassifier(BertPreTrainedModel):
    def __init__(self, config: BertConfig, task_args: list, model_path: str):
        super().__init__(config)
        self._bert = BertModel.from_pretrained(model_path, config = config)
        self._multi_task_classifier = MultiTaskClassifier(task_args)
        self._dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                task_name: str,
                input_ids: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        outputs = self._bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
    
        mask = input_ids == 0
        hidden_states = self._dropout(outputs.last_hidden_state)
        logits = self._multi_task_classifier(task_name, hidden_states, mask)
        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() 
            loss = loss_fct(logits.view(-1, self._multi_task_classifier.get_num_classes(task_name)), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

def main():
    model_path = 'hfl/chinese-roberta-wwm-ext-large'
    config = BertConfig.from_pretrained(model_path, output_hidden_states = True)
    task_num_classes = {'0': 3, '1': 7, '2': 15}

    from ..classifier import ClassifierType, ClassifierTaskArgs
    task_args = [
        ClassifierTaskArgs(task_name="OCNLI",
                           classifier_type = ClassifierType.ATTENTION,
                           hidden_size=128,
                           num_classes=3),
        ClassifierTaskArgs(task_name="OCEMOTION",
                           classifier_type = ClassifierType.ATTENTION,
                           hidden_size=128,
                           num_classes=7),
        ClassifierTaskArgs(task_name="TNEWS",
                           classifier_type = ClassifierType.ATTENTION,
                           hidden_size=128,
                           num_classes=15),                     
    ]

    model = BertMultitaskClassifier(config, task_args, model_path)
    
    import json
    inputs = torch.load('../user_data/encoding_data/OCEMOTION/train.pt')
    with open('../user_data/encoding_data/OCEMOTION/label2id.json', 'r', encoding='utf-8') as f:
        label2id = json.load(f)  # 从文件对象加载 JSON 数据
    logger.info(f"inputs keys: {inputs.keys()}\n, \
                label2id: {label2id}\n, \
                inputs['input_ids']: {inputs['input_ids'][:3]}\n, \
                inputs['labels']: {inputs['labels'][:3]}")
    print(type(inputs['input_ids']))
    index = 0
    output = model(
        task_name = "OCEMOTION",
        input_ids = torch.tensor(inputs['input_ids'][index]),
        token_type_ids = torch.tensor(inputs['token_type_ids'][index]),
        attention_mask = torch.tensor(inputs['attention_mask'][index]),
        labels = torch.tensor(inputs['labels'][index])
    )
    logger.info(f"Output: {output}")

if __name__ == '__main__':
    main()
