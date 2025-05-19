# In this scripts, you will learn
# 1. 将huggingface中的数据集工具和openprompt组合能够一块训练在不同的数据集上。
# 2. 如何使用模版语言初始化一模板
# 3. 如何使用模板包裹输入样例并将其转换为模型输入
# 4. 隐藏分词器的细节，并提供一个简单的分词器
# 5. 如何创建一个使用一个或者多个标签词语的分词器
# 5. 如何像床痛的预训练语言模型一样，训练一个prompt
import json

# load dataset
from datasets import load_dataset

CBPath = r'/home/public/gl/MultiDetection/PromptADDetection/openPromptLearning/CB/CB/'
raw_dataset = load_dataset('text', data_files={"train": CBPath + "train.jsonl", "validation": CBPath + "val.jsonl", "test": CBPath + "test.jsonl"})
print(raw_dataset['train'][0])
# from datasets import load_from_disk
# raw_dataset = load_from_disk("/home/hushengding/huggingface_datasets/saved_to_disk/super_glue.cb")
# Note that if you are running this scripts inside a GPU cluster, there are chances are you are not able to connect to huggingface website directly.
# In this case, we recommend you to run `raw_dataset = load_dataset(...)` on some machine that have internet connections.
# Then use `raw_dataset.save_to_disk(path)` method to save to local path.
# Thirdly upload the saved content into the machiine in cluster.
# Then use `load_from_disk` method to load the dataset.

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

# 创建模板
# 可以通过yaml配置文件创建一个模板，也可以直接使用手动传入参数创建
from openprompt.prompts import ManualTemplate
template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

# To better understand how does the template wrap the example, we visualize one instance.
# 为了更好的理解模板是如何包装样例的，我们实例化了一个例子
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

# 现在，我们将样例包装，准备传入分词器，产生语言模型的输入
# 你能够使用分词器将你自己的输入进行保障，但是我们推荐使用plm对应的包装分词器，这是针对当前模型输入量身定制的。
# 一般来说，如果你是用load_plm，会自动返回对应的wrapper，不然你要自己创建一个对应的类
# 当我们使用天t5进行分类的时候，我们只需要传入<pad> <extra_id_0> <eos>到解码器
# 模型只需要输出<pad> <extra_id_0> <eos>三个标记.
# 自主设置对应的wrapper
wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
# 使用加载模型对应的wrapper
from openprompt.plms import T5TokenizerWrapper
wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

# You can see what a tokenized example looks like by
tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
print(tokenized_example)
print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))


# 现在将整个数据集转变成对应的输出格式，单纯循环即可。
model_inputs = {}
for split in ['train', 'validation', 'test']:
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
        model_inputs[split].append(tokenized_example)
print('------------------------------------------------------')
print('-1、数据处理完毕-----------------------------------------')
print('------------------------------------------------------')

# We provide a `PromptDataLoader` class to help you do all the above matters and wrap them into an `torch.DataLoader` style iterator.
from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
    batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
next(iter(train_dataloader))
print('------------------------------------------------------')
print('-2、Dataloader加载完毕----------------------------------')
print('------------------------------------------------------')

# 定义语言转换器
# 对于分类问题，需要定义语言转换器，将词汇表中的数据转换成最终的类别概率，下面是具体实现细节
from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer, num_classes=3,
                        label_words=[["yes"], ["no"], ["maybe"]])

print('类别标签在词汇表中的数字表示：',myverbalizer.label_words_ids)
logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and
print(myverbalizer.process_logits(logits)) # 创建每一个类别标签的概率分布
print('------------------------------------------------------')
print('-3、Verbalizer加载完毕----------------------------------')
print('------------------------------------------------------')

# 虽然已经手动组合了plm、模板还有verbalizer，现在提出一个管道模型，获取batched数据，并且产生相应的概率大小。
from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

# Now the training is standard
from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters()
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters()
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
print('------------------------------------------------------')
print('-4、模型训练参数设置成功----------------------------------')
print('------------------------------------------------------')
# 可以看到，这里的具体流程和常规的深度学习一模一样，计算损失函数，然后进行反向传播
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
        optimizer.step()
        optimizer.zero_grad()
        if step %100 ==1:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

print('------------------------------------------------------')
print('-5、模型训练完毕----------------------------------')
print('------------------------------------------------------')

# Evaluate
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
print('------------------------------------------------------')
print('-6、模型测试完毕----------------------------------')
print('------------------------------------------------------')
