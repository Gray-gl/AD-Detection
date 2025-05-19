import json
from datasets import load_dataset

CBPath = r'/home/public/gl/MultiDetection/PromptADDetection/openPromptLearning/CB/CB/'
raw_dataset = load_dataset('text', data_files={"train": CBPath + "train.jsonl", "validation": CBPath + "val.jsonl", "test": CBPath + "test.jsonl"})
print(raw_dataset['train'][0])

from openprompt.data_utils import InputExample
label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
dataset = {}
for split in ['train', 'validation', 'test']:
    dataset[split] = []
    for data in raw_dataset[split]:
        data = json.loads(data['text'])
        if 'label' not in data:  # 检查标签是否存在
            continue  # 如果标签不存在，则跳过当前样本
        input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(label_map[data['label']]), guid=data['idx'])
        dataset[split].append(input_example)
print(dataset['train'][0])

# You can load the plm related things provided by openprompt simply by calling:
# 加载相关的nlp模型，获取plm、分词器、模型配置文件还有classWrapper
from openprompt.plms import load_plm
t5_path = r'/home/public/gl/MultiDetection/PromptADDetection/openPromptLearning/t5-base'
plm, tokenizer, model_config, WrapperClass = load_plm("t5", t5_path)


# 除了可以使用手动模板，还可以使用混合模板。在混合模板中，可以使用soft来表示一个可调节的模板
from openprompt.prompts import MixedTemplate

# 手动模板
# template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
# mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
# 混合模板1
mytemplate1 = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft": "Question:"} {"placeholder":"text_b"}? Is it correct? {"mask"}.')
# 混合模板2
mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_b"} {"soft"} {"mask"}.')

# 这里是具体的mixtemplate输出，可以看到，多了一个soft_token_ids，表示这个位置是可调节的。
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
wrapped_example1 = mytemplate1.wrap_one_example(dataset['train'][0])
# [[  {'text': 'It was a complex language. Not written down but handed down. One might say it was peeled down.', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 1},
#     {'text': '', 'soft_token_ids': 1, 'loss_ids': 0, 'shortenable_ids': 0},
#     {'text': '', 'soft_token_ids': 2, 'loss_ids': 0, 'shortenable_ids': 0},
#     {'text': '', 'soft_token_ids': 3, 'loss_ids': 0, 'shortenable_ids': 0},
#     {'text': ' the language was peeled down', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 1},
#     {'text': '', 'soft_token_ids': 4, 'loss_ids': 0, 'shortenable_ids': 0},
#     {'text': '<mask>', 'soft_token_ids': 0, 'loss_ids': 1, 'shortenable_ids': 0},
#     {'text': '.', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 0}],
#   {'guid': 0, 'label': 0}]
print(wrapped_example)
# [[  {'text': 'It was a complex language. Not written down but handed down. One might say it was peeled down.', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 1},
#     {'text': '▁Question', 'soft_token_ids': 1, 'loss_ids': 0, 'shortenable_ids': 0},
#     {'text': ':', 'soft_token_ids': 2, 'loss_ids': 0, 'shortenable_ids': 0},
#     {'text': ' the language was peeled down', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 1},
#     {'text': '? Is it correct?', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 0},
#     {'text': '<mask>', 'soft_token_ids': 0, 'loss_ids': 1, 'shortenable_ids': 0},
#     {'text': '.', 'soft_token_ids': 0, 'loss_ids': 0, 'shortenable_ids': 0}],
#   {'guid': 0, 'label': 0}]
print(wrapped_example1)


# 构建对应分瓷器的包装器
wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
# next(iter(train_dataloader))

# ## Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability.
# Let's have a look at the verbalizer details:

from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=3,
                        label_words=[["yes"], ["no"], ["maybe"]])

print(myverbalizer.label_words_ids)
logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm
myverbalizer.process_logits(logits)


from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# ## below is standard training


from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters
# 对于prompt和模型参数使用不同优化方式，这个是指模板中的soft参数，这个是可以调节的
optimizer_grouped_parameters2 = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)

for epoch in range(10):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        # print(tot_loss/(step+1))
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

# ## evaluate

# %%
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")


allpreds = []
alllabels = []
for step, inputs in enumerate(validation_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print(acc)