# copied and modified from  NtaylorOX/Public_Clinical_Prompt
from typing import Dict

import matplotlib
from pip._vendor import chardet
from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
import pandas as pd
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, ManualTemplate, SoftVerbalizer, PTRTemplate, PTRVerbalizer
from openprompt.prompts import SoftTemplate, MixedTemplate
from openprompt import PromptForClassification
# from openprompt.utils.logging import logger
from loguru import logger
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json
import itertools
from collections import Counter
import os
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm
from prompt_ad_utils import read_input_text_len_control, read_input_no_len_control
from openprompt.utils.reproduciblity import set_seed


def loading_data_asexample(data_save_dir, sample_size, classes, model, mode='train', validation_dict=None,args=None):
    """
    PS: Load data from the saved csv files.    加载处理过的数据，并且统一长度，并生成对应的InputExample
    data_save_dir: 数据保存目录的路径。
    sample_size: 样本大小，用于确定要加载的数据量。
    classes: 类别，可能用于后续处理但在此函数中未直接使用。
    model: 用于处理数据的模型，可能影响数据的预处理方式。
    mode: 指示当前是训练模式、测试模式还是交叉验证模式。
    validation_dict: 在交叉验证模式下，包含训练和测试发言人标识的字典。
    """

    # 根据加载的模式确定数据集路径
    if 'cv' in mode:
        data_file = os.path.join(data_save_dir, 'train_chas_A')  # .format(trans_type, manual_type)
    else:
        data_file = os.path.join(data_save_dir, '{:s}'.format(mode))

    # 这里表示数据是否已经处理过
    if args.data_not_saved:
        data_file += '.csv'
    else:
        data_file += '_cut.csv'
    print('处理数据',data_file)

    # 读取数据
    # Detect encoding
    with open(data_file, 'rb') as file:
        result = chardet.detect(file.read())

    # Read file with detected encoding
    raw_df = pd.read_csv(data_file, encoding=result['encoding'])
    # 如果是交叉验证模式，需要单独读取训练集和测试集
    if "cv" in mode:
        if validation_dict == None:
            raise ValueError("Cross validation mode requires validation_dict input")
        if args.data_not_saved:
            raise ValueError("Data proprocessing (when data_not_saved == True) is only supported by test mode")
        train_speaker = validation_dict['train_speaker']
        validation_speaker = validation_dict['test_speaker']
        if mode == "train_cv":
            load_data_df = raw_df[raw_df["id"].apply(lambda x: True if x in train_speaker else False)]
        elif mode == "test_cv":
            load_data_df = raw_df[raw_df["id"].apply(lambda x: True if x in validation_speaker else False)]

    else:
        load_data_df = raw_df

    # 控制数据的长度，只需要运行一次，数据是否经过预处理
    if args.data_not_saved:
        cut_data_save_path = os.path.join(data_save_dir, '{:s}_chas_A_cut.csv'.format(mode))
        org_data = read_input_no_len_control(load_data_df, sample_size=sample_size, max_len=512, model=model,
                                             save_trans=cut_data_save_path)
    else:
        org_data = read_input_no_len_control(load_data_df, sample_size=sample_size, max_len=512, model=model,
                                             save_trans=None)

    # 从数据中提取出来的数据，以及数据的类别分布
    data_list = []
    label_list = []
    if sample_size == -1:
        # 遍历所有的行，为每一行创建一个InputExample对象，同时添加对象和标签
        for index, data in org_data.iterrows():
            text_a_without_postclitic = data['joined_all_par_trans'].replace("POSTCLITIC", "")
            if args.use_pic_type:
                if data['id'].startswith('adrs'):
                    pic_type = "Cookie Theft"
                else:
                    pic_type = "Many Animals"
                input_example = InputExample(text_b = pic_type,text_a=text_a_without_postclitic, label=data['ad'], guid=data["id"])
            else:
                input_example = InputExample(text_a=text_a_without_postclitic, label=data['ad'], guid=data["id"])
            data_list.append(input_example)
            label_list.append(data['ad'])
        return data_list, Counter(label_list)



# 绘制混合矩阵
def plot_confusion_matrix(cm, class_names,save_path):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    credit: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('normal')

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix: ADReSS")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() * 0.90

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    figure.savefig(f'{save_path}/test_mtx.png')
    matplotlib.pyplot.close()
    return figure



'''
Script to run different setups of prompt learning.
'''

# 创建参数解析器
parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--plm_eval_mode", action="store_true",
                    help="whether to turn off the dropout in the freezed model. Set to true to turn off.")

# 模型微调参数
parser.add_argument("--tune_plm", action="store_true",default = True)
parser.add_argument("--part_tuning", action="store_true",default=True)
parser.add_argument("--freeze_verbalizer_plm", action="store_true")

# 模型路径参数
parser.add_argument("--model", type=str, default='roberta',
                    help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name", type=str,
                    # default='roberta'
                    # default='roberta_control_gpt'
                    # default='roberta_content_reduced_mixed'
                    # default='roberta_all_mixed'
                    # default='roberta_raw_mixed'
                    # default='roberta_mlm_raw_normal'
                    # default='roberta_mlm_pure_normal'
                    default='roberta_mlm_raw_lexical_deficits'
                    # default='roberta_mlm_raw_reduced_content'
                    # default='roberta_mlm'
                    )
parser.add_argument("--project_root", type=str, default='/home/public/gl/MultiDetection/PromptADDetection/',
                    help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--logs_root", type=str, default='/home/public/gl/MultiDetection/PromptADDetection/model_log/',
                    help="The dir in which project results are stored in, i.e. the absolute path of OpenPrompt")
parser.add_argument("--off_line_model_dir", type=str,
                    # default = '/home/public/gl/MultiDetection/PromptADDetection/plm',
                    default = '/home/public/gl/MultiDetection/PromptADDetection/plm',
                    help="The dir in which pre-trained model are stored in")

# 模板调整参数
parser.add_argument("--scripts_path", type=str, default="template")
parser.add_argument("--template_id", type=int, default=6)
parser.add_argument("--template_type", type=str, default="manual")
parser.add_argument("--use_pic_type", type=bool, default=False)   # 是否使用图片类型特征作为模板
parser.add_argument("--verbalizer_type", type=str, default="manual")
parser.add_argument("--manual_type", type=str, default="A")

# 数据集参数
parser.add_argument("--data_dir", type=str,
                    default= '/home/public/gl/MultiDetection/PromptADDetection/data/adress')
parser.add_argument("--data_not_saved", action="store_true",default = True)  # 数据是否经过剪切，统一长度在512个单词内

# 训练参数
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_steps", default=5000, type=int)
parser.add_argument("--plm_lr", type=float, default=1e-5)
parser.add_argument("--plm_warmup_steps", type=float, default=5)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=5)
parser.add_argument("--init_from_vocab", action="store_true")
parser.add_argument("--eval_every_steps", type=int, default=100)
parser.add_argument("--soft_token_num", type=int, default=5)
parser.add_argument("--optimizer", type=str, default="adamw")
parser.add_argument("--gradient_accum_steps", type=int, default=1)
parser.add_argument("--dev_run",action="store_true")
parser.add_argument("--gpu_num", type=int, default=0)
parser.add_argument("--balance_data", action="store_true") # whether to downsample data to majority class
parser.add_argument("--ce_class_weights",
                    action="store_true")  # whether to apply class weights to cross entropy loss fn
parser.add_argument("--sampler_weights", action="store_true")  # apply weights to weighted data sampler
parser.add_argument("--training_size", type=str, default="full")  # or fewshot or zero
parser.add_argument("--no_ckpt", type=bool, default=False)
parser.add_argument("--last_ckpt", action="store_true",default=True)  # 是否能保存最有个ckpt文件
parser.add_argument("--crossvalidation", action="store_true",default = False)
parser.add_argument("--val_file_dir", type=str, default=r'/home/public/gl/MultiDetection/PromptADDetection/latest_tmp_dir/ten_fold_1.json')
parser.add_argument("--val_fold_idx", type=int, default=999)
parser.add_argument("--tensorboard", action="store_true")
parser.add_argument("--tunable_layernum", type=int, default=0)
parser.add_argument("--pause", action="store_true")
parser.add_argument("--pause_threshold", type=str)
parser.add_argument("--transcription", type=str, default='chas')
parser.add_argument("--asr_format", type=int, default=3)
parser.add_argument("--dataset_name", type=str, default="ADReSS")
parser.add_argument("--model_parallelize",action="store_true", default=True)
parser.add_argument("--zero_shot", action="store_true")
parser.add_argument("--few_shot_n", type=int, default=100)
parser.add_argument("--no_training", action="store_true",default=False)
parser.add_argument("--run_evaluation", action="store_true",default=True)

parser.add_argument(
    '--sensitivity',
    default=False,
    type=bool,
    help='Run sensitivity trials - investigating the influence of classifier hidden dimension on performance in frozen plm setting.'
)
# 解析参数
args = parser.parse_args()
logger.info(f" arguments provided were: {args}")
print('-----------------------------------------------------')
print('1、参数解析完毕')
print('-----------------------------------------------------')


#################################################################
# 参数初始化：设置随机种子，设置时间戳，设置模型路径、设置交叉验证参数、设置cuda设备
#################################################################
# 1、随机种子、时间戳、模型路径、路径设置
set_seed(args.seed)  # 设置随机种子
time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))   # 设置时间戳，指定checkpoint文件和日志文件的名称
raw_time_now = time_now.split('--')[0]

# 2、校验交叉验证并设置交叉验证参数
if args.crossvalidation:
    if args.val_file_dir == None:
        raise ValueError("Need to specify val_file_dir")
    assert args.val_file_dir.split(".")[-1] == 'json'
    run_idx = int(args.val_file_dir.split("fold_")[-1].split('.')[0])
    assert run_idx in range(1, 11)
    version = f"version_{args.seed}_val"
else:
    version = f"version_{args.seed}"

# 3、根据是否微调模型，设置模型的保存路径、日志路径、参数保存路径
args.logs_root = args.logs_root.rstrip("/") + "/"
args.project_root = args.project_root.rstrip("/") + "/"
if args.tune_plm == True:
    logger.warning("Unfreezing the plm - will be updated during training")
    freeze_plm = False
    # 设置checkpoint、logs和params保存路径
    if args.sensitivity:  # 敏感性分析实验
        logger.warning(f"performing sensitivity analysis experiment!")
        logs_dir = f"{args.logs_root}sensitivity/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_lr{args.plm_lr}/{version}"
        ckpt_dir = f"{logs_dir}/checkpoints/"
    elif args.part_tuning:  # 局部微调实验
        logger.warning(f"part parameters tuning run!")
        logs_dir = f"{args.logs_root}parttuning/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_lr{args.plm_lr}_layernum{args.tunable_layernum}/{version}"
        ckpt_dir = f"{logs_dir}/checkpoints/"
    else:   #
        if args.manual_type == "A":   # 根据模板的版本号进行设置
            logs_dir = f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        else:
            logs_dir = f"{args.logs_root}{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_{args.few_shot_n}_{args.manual_type}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
else:
    logger.warning("Freezing the plm")
    freeze_plm = True
    # we have to account for the slight issue with softverbalizer being wrongly implemented by openprompt
    # here we have an extra agument which will correctly freeze the PLM parts of softverbalizer if set to true
    if args.freeze_verbalizer_plm and args.verbalizer_type == "soft":
        logger.warning("also will be explicitly freezing plm parts of the soft verbalizer")
        if args.sensitivity:
            logger.warning(f"performing sensitivity analysis experiment!")
            logs_dir = f"{args.logs_root}sensitivity/frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_frozenverb{args.verbalizer_type}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        else:
            logs_dir = f"{args.logs_root}frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_frozenverb{args.verbalizer_type}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
    else:  # set checkpoint, logs and params save_dirs
        if args.sensitivity:
            logger.warning(f"performing sensitivity analysis experiment!")
            logs_dir = f"{args.logs_root}sensitivity/frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
        else:
            logs_dir = f"{args.logs_root}frozen_plm/{args.model_name}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}_{args.training_size}_{args.few_shot_n}/{version}"
            ckpt_dir = f"{logs_dir}/checkpoints/"
# 校验相关路径是否存在，并进行创建，同时设置tensorboard日志
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if args.tensorboard:
    writer = SummaryWriter(logs_dir)

# 4、训练参数保存并记录到tensorboard日志中
if not os.path.exists(os.path.join(ckpt_dir, 'hparams.txt')):
    with open(os.path.join(ckpt_dir, 'hparams.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
save_metrics = {"random/metric": 0}
if args.tensorboard:
    writer.add_hparams(args.__dict__, save_metrics)
print(f"hparams dict: {args.__dict__}")

# 5、设定使用的cuda设备、类别标签、训练标志
use_cuda = True
if use_cuda:
    cuda_device = torch.device(f'cuda:{args.gpu_num}')
else:
    cuda_device = torch.device('cpu')
torch.cuda.set_device(cuda_device)
class_labels = [
    "healthy",
    "dementia",
]
do_training = not args.no_training


# 6、设置序列长度，并且根据是否微调模型，设置批处理大小和累计补偿步数
max_seq_l = 512 # this should be specified according to the running GPU's capacity
# 根据是否对预训练模型进行微调，配置批处理大小和累计补偿步数
if args.tune_plm:  # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
    batchsize_t = args.batch_size
    gradient_accumulation_steps = args.gradient_accum_steps
    model_parallelize = True  # if multiple gpus are available, one can use model_parallelize
else:
    batchsize_t = 4
    gradient_accumulation_steps = 1
    model_parallelize = False

print('-----------------------------------------------------')
print('2、训练初始化完毕')
print('-----------------------------------------------------')


################################################################
# 数据预处理：加载数据集、初始化数据集、统一数据长度
################################################################
logger.warning(f"Using the following dataset: {args.dataset_name} ")

# 1、设置采样器参数，获取类别权重和采样权重，处理数据不平衡问题
sampler = None
sampler_weights = args.sampler_weights

# 2、设置数据集的类别名和数据集路径
dataset = {}   #
data_dir = args.data_dir
SAMPLE_SIZE = -1

# 3、分情况处理不同的数据
if args.crossvalidation:
    with open(args.val_file_dir, 'r') as json_read:
        cv_fold_list = json.load(json_read)
    validation_dict = cv_fold_list[args.val_fold_idx]
    dataset['train'], train_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels,
                                                                   args.model_name, mode='train_cv',
                                                                   validation_dict=validation_dict,args = args)
    dataset['validation'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels,
                                                      args.model_name, mode='test_cv',
                                                      validation_dict=validation_dict,args = args)
    dataset['test'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels,
                                                args.model_name, mode='test_cv',
                                                validation_dict=validation_dict,args = args)
else:

    # 加载手录的数据
    dataset['train'], train_classes_count = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels,
                                                                   args.model_name, mode='train',args = args)
    dataset['validation'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name,
                                                      mode='test',args = args)
    dataset['test'], _ = loading_data_asexample(data_dir, SAMPLE_SIZE, class_labels, args.model_name,
                                                    mode='test',args = args)
# 如果是小样本训练，就在本地产生数据集
if args.training_size == "fewshot":
    logger.warning(f"Will be performing few shot learning.")
    # 如果想要在测试和训练阶段都使用小样本学习，需要创建小样本采样器
    support_sampler = FewShotSampler(num_examples_per_label=args.few_shot_n, also_sample_dev=False)

    # 使用小样本采样器分别创建训练集，测试集和验证集并不需要使用小样本采集器重新进行采集
    dataset['train'] = support_sampler(dataset['train'], seed=args.seed)


# 4、根据训练集中各类别的数量计算类别权重，以用于训练过程中的损失函数，缓解类别不平衡的问题
ce_class_weights = args.ce_class_weights
if args.ce_class_weights:
    task_class_weights = [train_classes_count[0] / train_classes_count[i] for i in range(len(class_labels))]


print('-----------------------------------------------------')
print('2、数据加载完毕')
print('-----------------------------------------------------')


################################################################
# 训练流程搭建：加载预训练plm模型、加载模板、加载分词器
################################################################
# 设置模型路径====》需要根据变动进行增加
model_dict = {
    'bert-base-uncased': os.path.join(args.off_line_model_dir, 'bert-base-uncased'),
    'roberta-base': os.path.join(args.off_line_model_dir, 'roberta-base'),
    'roberta': os.path.join(args.off_line_model_dir, 'roberta'),
    'roberta_mlm': os.path.join(args.off_line_model_dir, 'roberta_mlm'),
    'roberta_mlm_raw_normal': os.path.join(args.off_line_model_dir, 'roberta_mlm_raw_normal'),
    'roberta_mlm_pure_normal': os.path.join(args.off_line_model_dir, 'roberta_mlm_pure_normal'),
    'roberta_mlm_raw_lexical_deficits': os.path.join(args.off_line_model_dir, 'roberta_mlm_raw_lexical_deficits'),
    'roberta_mlm_raw_reduced_content': os.path.join(args.off_line_model_dir, 'roberta_mlm_raw_reduced_content'),
    # 'roberta_control_gpt': os.path.join(args.off_line_model_dir, 'roberta_control_gpt'),
    # 'roberta_mlm_label': os.path.join(args.off_line_model_dir, 'roberta_mlm_label'),
    # 'roberta_content_reduced_mixed': os.path.join(args.off_line_model_dir, 'roberta_content_reduced_mixed'),
    # 'roberta_all_mixed': os.path.join(args.off_line_model_dir, 'roberta_all_mixed'),
    # 'roberta_raw_mixed': os.path.join(args.off_line_model_dir, 'roberta_raw_mixed'),

    # 重新训练的数据，使用raw_normal,原始数据和正常数据的混合
    'roberta_mlm_raw_normal': os.path.join(args.off_line_model_dir, 'roberta_mlm_raw_normal'),
              }
# 1、加载预训练plm模型
print(args.model_name)
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, model_dict[args.model_name])

# 2、加载模板和分词器====》需要根据变动进行增加
scriptsbase = f"{args.project_root}{args.scripts_path}"
scriptformat = "txt"
my_template = None
if args.template_type == "manual":
    # 手动模板
    print(f"manual template selected, with id :{args.template_id}")
    my_template = ManualTemplate(tokenizer=tokenizer).from_file(
                                                          f"{scriptsbase}/manual_template.txt",
                                                               choice=args.template_id
                                                              )

elif args.template_type == "soft":
    # 加载软模板，模板内容可调整训练
    print(f"soft template selected, with id :{args.template_id}")

    my_template = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num,
                               initialize_from_vocab=args.init_from_vocab).from_file(
                                                                                    f"{scriptsbase}/soft_template.txt",
                                                                                    choice=args.template_id
                                                                                    )

elif args.template_type == "ptr":
    print(f"ptr template selected, with id :{args.template_id}")
    my_template = ManualTemplate(tokenizer=tokenizer).from_file(
                                                                f"{scriptsbase}/ptr_template.txt",
                                                                choice=args.template_id
                                                                )
elif args.template_type == "mixed":
    # 加载混合模板，手工模板和软模板的混合，一部分可以自己训练，还有一部分不能训练
    print(f"mixed template selected, with id :{args.template_id}")
    my_template = MixedTemplate(model=plm, tokenizer=tokenizer).from_file(f"{scriptsbase}/mixed_template.txt",
                                                                          choice=args.template_id)

# 3、加载分词器
my_verbalizer = None
if args.verbalizer_type == "manual":
    # 手工分词器
    print(f"manual verbalizer selected!")
    my_verbalizer = ManualVerbalizer(
        classes=class_labels,
        label_words={
            "dementia": ["dementia"],  #
            "healthy": ["healthy"],
        },
        tokenizer=tokenizer,
    )

elif args.verbalizer_type == "soft":
    # 软分词器，需要在训练的时候选定参数
    print(f"soft verbalizer selected!")
    my_verbalizer = SoftVerbalizer(
        classes=class_labels,
        label_words={
            "dementia": ["dementia"],
            "healthy": ["healthy"],
        },
        tokenizer=tokenizer,
        model=plm,
        num_classes=len(class_labels)
    )

    # we noticed a bug where soft verbalizer was technically not freezing alongside the PLM -
    # meaning it had considerably greater number of trainable parameters
    # so if we want to properly freeze the verbalizer plm components as described here: https://github.com/thunlp/OpenPrompt/blob/4ba7cb380e7b42c19d566e9836dce7efdb2cc235/openprompt/prompts/soft_verbalizer.py#L82
    # we now need to actively set grouped_parameters_1 to requires_grad = False
    if args.freeze_verbalizer_plm and freeze_plm:
        logger.warning(f"We have a soft verbalizer and want to freeze it alongside the PLM!")
        # now set the grouped_parameters_1 require grad to False
        for param in my_verbalizer.group_parameters_1:
            param.requires_grad = False

wrapped_example = my_template.wrap_one_example(dataset['train'][0])
print(wrapped_example)
print('-----------------------------------------------------')
print('3、template、tokenizer加载完毕！')
print('-----------------------------------------------------')


################################################################
# 模型加载：加载模型、加载dataloader
################################################################
# 1、组装分类模型
print(f"tune_plm value: {args.tune_plm}")
prompt_model = PromptForClassification(plm=plm, template=my_template, verbalizer=my_verbalizer,
                                       freeze_plm=freeze_plm,plm_eval_mode=args.plm_eval_mode)
if use_cuda:  # 参数传递给cuda
    prompt_model = prompt_model.to(cuda_device)
# TODO：需要进一步实现模型并行化训练
# if args.model_parallelize:  # 模型并行化训练
#     prompt_model.parallelize()

# 2、加载dataloader
if not args.no_training:
    if args.template_type == 'soft':
        max_seq_l -= args.soft_token_num
    # if we have a sampler .e.g weightedrandomsampler. Do not shuffle
    if "WeightedRandom" in type(sampler).__name__:
        logger.warning("Sampler is WeightedRandom - will not be shuffling training data!")
        shuffle = False
    else:
        shuffle = True
    logger.warning(f"Do training is True - creating train and validation dataloders!")
    train_data_loader = PromptDataLoader(
        dataset=dataset['train'],
        tokenizer=tokenizer,
        template=my_template,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_l,
        decoder_max_length=3,
        batch_size=args.batch_size,
        truncate_method="tail",
        shuffle=True,
    )
    validation_data_loader = PromptDataLoader(
        dataset=dataset['validation'],
        tokenizer=tokenizer,
        template=my_template,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=max_seq_l,
        decoder_max_length=3,
        batch_size=args.batch_size,
        truncate_method="tail",
        shuffle=True,
    )

# zero-shot test
test_data_loader = PromptDataLoader(
    dataset=dataset['test'],
    tokenizer=tokenizer,
    template=my_template,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=max_seq_l,
    decoder_max_length=3,
    batch_size=args.batch_size,
    truncate_method="tail",
    shuffle=True,
)

print('-----------------------------------------------------')
print('4、模型加载完毕、dataloader创建完毕!')
print('-----------------------------------------------------')


################################################################
# 确定模型训练细节：指定损失函数、
################################################################
from transformers import AdamW, get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5

# 1、指定模型训练的损失函数
if ce_class_weights:
    # 这里是根据类别分布设置权重，更加合理
    logger.warning("we have some task specific class weights - passing to CE loss")
    task_class_weights = torch.tensor(task_class_weights, dtype=torch.float).to(cuda_device)
    loss_func = torch.nn.CrossEntropyLoss(weight=task_class_weights, reduction='mean')
else:
    loss_func = torch.nn.CrossEntropyLoss()

# 2、设置模型训练的最大步数（参数更新的最多次数）
tot_step = args.max_steps

# 3、指定模型训练的参数更新内容、学习策略、优化器
if args.tune_plm:
    # 正常情况下，如果我们使用soft template话，会将模型的参数完全冻结，但是这里为了逻辑完善，增加一个功能
    logger.warning("We will be tuning the PLM!")
    no_decay = ['bias',
                'LayerNorm.weight']

    # 判定是否为局部微调和全局微调
    if args.part_tuning:
        # 部分微调
        optimizer_grouped_parameters_plm = [
            {'params': [p for n, p in prompt_model.plm.named_parameters() if
                        (not any(nd in n for nd in no_decay))],
             'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.plm.named_parameters() if
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
    else:
        # 全局微调
        tunable_layers = []
        # 如果是全局微调，这里仅仅修改编码器
        for layer_num in range(12 - args.tunable_layernum, 12):
            tunable_layers.append("bert.encoder.layer.{}".format(layer_num))
        tunable_layers.append('cls.predictions')
        optimizer_grouped_parameters_plm = [
            # 不在衰减列表的参数保存
            {'params': [p for n, p in prompt_model.plm.named_parameters() if
                        (not any(nd in n for nd in no_decay)) and
                        (any(fl in n for fl in tunable_layers))],
             'weight_decay': 0.01},
            # 在参数列表的保存
            {'params': [p for n, p in prompt_model.plm.named_parameters() if
                        any(nd in n for nd in no_decay) and
                        (any(fl in n for fl in tunable_layers))],
             'weight_decay': 0.0}
        ]

    # 指定模型的优化器
    optimizer_plm = AdamW(optimizer_grouped_parameters_plm,
                          lr=args.plm_lr,
                          no_deprecation_warning=True)
    # 指定学习策略
    scheduler_plm = get_linear_schedule_with_warmup(
        optimizer_plm,
        num_warmup_steps=args.plm_warmup_steps,
        num_training_steps=tot_step
    )
else:
    # 不微调模型的参数
    logger.warning("We will not be tunning the plm - i.e. the PLM layers are frozen during training")
    optimizer_plm = None
    scheduler_plm = None

# 3、使用soft template，需要单独指定参数修改策略、参数优化器
if args.template_type == "soft" or args.template_type == "mixed":
    logger.warning(f"{args.template_type} template used - will be fine tuning the prompt embeddings!")
    # note that you have to remove the raw_embedding manually from the optimization
    optimizer_grouped_parameters_template = [{'params': [p for name, p in prompt_model.template.named_parameters()
                                                         if 'raw_embedding' not in name]}]
    if args.optimizer.lower() == "adafactor":
        optimizer_template = Adafactor(optimizer_grouped_parameters_template,
                                       lr=args.prompt_lr,
                                       relative_step=False,
                                       scale_parameter=False,
                                       warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        scheduler_template = get_constant_schedule_with_warmup(optimizer_template,
                                                               num_warmup_steps=args.warmup_step_prompt)  # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
        optimizer_template = AdamW(optimizer_grouped_parameters_template, lr=args.prompt_lr)  # usually lr = 0.5
        scheduler_template = get_linear_schedule_with_warmup(
            optimizer_template,
            num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step)  # usually num_warmup_steps is 500

elif args.template_type == "manual" or args.template_type == "ptr":
    optimizer_template = None
    scheduler_template = None

# 4、为soft verbalizer指定参数修改策略、参数优化器
if args.verbalizer_type == "soft":
    logger.warning("Soft verbalizer used - will be fine tuning the verbalizer/answer embeddings!")
    optimizer_grouped_parameters_verb = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr": args.plm_lr},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr": args.plm_lr},
    ]
    optimizer_verb = AdamW(optimizer_grouped_parameters_verb)
    scheduler_verb = get_linear_schedule_with_warmup(
        optimizer_verb,
        num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step)  # usually num_warmup_steps is 500
elif args.verbalizer_type == "manual" or args.verbalizer_type == "ptr":
    optimizer_verb = None
    scheduler_verb = None
print('-----------------------------------------------------')
print('5、训练参数设置完毕！！')
print('-----------------------------------------------------')



################################################################
# 确定模型训练过程
################################################################
def train(prompt_model, train_data_loader, num_epochs, mode="train", ckpt_dir=ckpt_dir):
    logger.warning(f"cuda current device inside training is: {torch.cuda.current_device()}")
    # 设置模型的状态为训练
    prompt_model.train()

    # 设置一些计数器
    actual_step = 0
    glb_step = 0

    # 设置验证集的一些指标进行监测
    best_val_acc = 0
    best_val_f1 = 0
    best_val_prec = 0
    best_val_recall = 0

    # this will be set to true when max steps are reached
    leave_training = False

    # 对于每一个训练周期数，遍历数据加载器中的每一个批次的数据
    for epoch in tqdm(range(num_epochs)):
        print(f"On epoch: {epoch}")
        tot_loss = 0
        epoch_loss = 0
        # 遍历每一个批次的数据
        for step, inputs in enumerate(train_data_loader):

            if use_cuda:
                inputs = inputs.to(cuda_device)
            logits = prompt_model(inputs)
            labels = inputs['label']

            loss = loss_func(logits, labels)

            # 标准化损失，处理梯度累计步数
            loss = loss / gradient_accumulation_steps

            # 反向传播损失，计算梯度
            loss.backward()
            tot_loss += loss.item()
            actual_step += 1

            # 满梯度累积步数之后，记录实际损失
            if actual_step % gradient_accumulation_steps == 0:
                # 记录参数
                aveloss = tot_loss / (step + 1)
                if args.tensorboard:
                    writer.add_scalar("train/batch_loss", aveloss, glb_step)

                # 梯度剪裁，防止模型梯度爆炸
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                glb_step += 1

                # 反向传播损失，并更新优化器和调度器，同时将梯度置为零
                if optimizer_plm is not None:
                    optimizer_plm.step()
                    optimizer_plm.zero_grad()
                if scheduler_plm is not None:
                    scheduler_plm.step()
                # template
                if optimizer_template is not None:
                    optimizer_template.step()
                    optimizer_template.zero_grad()
                if scheduler_template is not None:
                    scheduler_template.step()
                # verbalizer
                if optimizer_verb is not None:
                    optimizer_verb.step()
                    optimizer_verb.zero_grad()
                if scheduler_verb is not None:
                    scheduler_verb.step()

                # 检查是都超过最大迭代步数
                if glb_step > args.max_steps:
                    logger.warning("max steps reached - stopping training!")
                    leave_training = True
                    break

        # 获取每一个周期的损失，并将结果写入到tensorboard的日志中
        epoch_loss = tot_loss / len(train_data_loader)
        print("Epoch {}, loss: {}".format(epoch, epoch_loss), flush=True)
        if args.tensorboard:
            writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # 运行验证集去获取一些参数
        (val_loss, val_acc, val_prec_weighted, val_prec_macro,
         val_recall_weighted, val_recall_macro, val_f1_weighted,
         val_f1_macro, val_auc_weighted, val_auc_macro) = evaluate(
            prompt_model, validation_data_loader, epoch=epoch)
        if args.tensorboard:
            writer.add_scalar("valid/loss", val_loss, epoch)
            writer.add_scalar("valid/balanced_accuracy", val_acc, epoch)
            writer.add_scalar("valid/precision_weighted", val_prec_weighted, epoch)
            writer.add_scalar("valid/precision_macro", val_prec_macro, epoch)
            writer.add_scalar("valid/recall_weighted", val_recall_weighted, epoch)
            writer.add_scalar("valid/recall_macro", val_recall_macro, epoch)
            writer.add_scalar("valid/f1_weighted", val_f1_weighted, epoch)
            writer.add_scalar("valid/f1_macro", val_f1_macro, epoch)

            # TODO add binary classification metrics e.g. roc/auc
            writer.add_scalar("valid/auc_weighted", val_auc_weighted, epoch)
            writer.add_scalar("valid/auc_macro", val_auc_macro, epoch)

            # # add cm to tensorboard
            # writer.add_figure("valid/Confusion_Matrix", cm_figure, epoch)

        # 如果验证集精确度有改良，就保存模型的检查点
        if val_acc >= best_val_acc:
            # only save ckpts if no_ckpt is False - we do not always want to save - especially when developing code
            if not args.no_ckpt:
                logger.warning(f"Accuracy improved! Saving checkpoint at :{ckpt_dir}!")
                if not args.crossvalidation:
                    torch.save(prompt_model.state_dict(), os.path.join(ckpt_dir, "best-checkpoint.ckpt"))
                else:
                    torch.save(prompt_model.state_dict(),
                               os.path.join(ckpt_dir,
                                            "best-checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)
                                            )
                               )
            best_val_acc = val_acc

        # 如果达到了最大训练步数，提前结束训练
        if glb_step > args.max_steps:
            leave_training = True
            break

        if leave_training:
            logger.warning("Leaving training as max steps have been met!")
            break

def evaluate(prompt_model, dataloader, mode="validation", class_labels=class_labels, epoch=None):
    prompt_model.eval()
    tot_loss = 0
    allpreds = []
    alllabels = []
    # record logits from the the model
    alllogits = []
    # store probabilties i.e. softmax applied to logits
    allscores = []

    allids = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.to(cuda_device)
            logits = prompt_model(inputs)
            labels = inputs['label']

            loss = loss_func(logits, labels)
            tot_loss += loss.item()

            # 将所有的标签加入到列表中
            alllabels.extend(labels.cpu().tolist())

            # 将所有的id加入到列表中
            allids.extend(inputs['guid'])
            alllogits.extend(logits.cpu().tolist())
            # use softmax to normalize, as the sum of probs should be 1
            # if binary classification we just want the positive class probabilities
            if len(class_labels) > 2:
                allscores.extend(torch.nn.functional.softmax(logits, dim=-1).cpu().tolist())
            else:
                allscores.extend(torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().tolist())

            # add predicted labels
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    val_loss = tot_loss / len(dataloader)
    # get sklearn based metrics
    acc = balanced_accuracy_score(alllabels, allpreds)
    f1_weighted = f1_score(alllabels, allpreds, average='weighted')
    f1_macro = f1_score(alllabels, allpreds, average='macro')
    prec_weighted = precision_score(alllabels, allpreds, average='weighted',
                                    zero_division=1.0)
    prec_macro = precision_score(alllabels, allpreds, average='macro',
                                 zero_division=1.0)
    recall_weighted = recall_score(alllabels, allpreds, average='weighted')
    recall_macro = recall_score(alllabels, allpreds, average='macro')

    # roc_auc 计算roc
    roc_auc_weighted = roc_auc_score(alllabels, allscores, average="weighted")
    roc_auc_macro = roc_auc_score(alllabels, allscores, average="macro")

    # 获取并绘制融合矩阵
    cm = confusion_matrix(alllabels, allpreds)
    if not args.crossvalidation:
        plot_confusion_matrix(cm, class_labels,ckpt_dir)

    # 如果正在做最后的测试阶段，就将数据预测结果进行保存，绘制的图片进行保存
    if mode == 'test':
        assert epoch != None
        epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch))

        # 输出预测结果，并将结果进行保存
        logger.warning(f"mode was: {mode} so will be saving evaluation results to file as well as tensorboard!")
        print(classification_report(alllabels, allpreds, target_names=class_labels))
        test_report = classification_report(alllabels, allpreds, target_names=class_labels, output_dict=True)
        test_report_df = pd.DataFrame(test_report).transpose()
        if args.crossvalidation:
            test_report_name = "test_class_report_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            test_results_name = "test_results_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            figure_name = "test_cm_cv{}_fold{}.png".format(run_idx, args.val_fold_idx)
        else:
            if args.transcription in ['cnntdnn', 'fusionshujie', "sys14_26.4", "sys18_25.9"]:  # "cnntdnn":
                test_report_name = "test_{}_{}_class_report.csv".format(args.transcription, args.asr_format)
                test_results_name = "test_{}_{}_results.csv".format(args.transcription, args.asr_format)
                figure_name = "test_{}_{}_cm.png".format(args.transcription, args.asr_format)
            elif args.transcription == "chas":
                test_report_name = "test_class_report.csv"
                test_results_name = "test_results.csv"
                figure_name = "test_cm.png"
        test_report_df.to_csv(os.path.join(epoch_dir, test_report_name), index=False)

        # 保存所有的预测概率、标签、预测标签、logits
        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        pd.DataFrame(results_dict).to_csv(os.path.join(epoch_dir, test_results_name), index=False)

    if mode == 'last':
        # 该模式是针对少样本学习使用的
        logger.warning(f"mode was: {mode} so will be saving evaluation results to file as well as tensorboard!")

        # 将所有预测结果进行保存
        test_report = classification_report(alllabels, allpreds, target_names=class_labels, output_dict=True)
        test_report_df = pd.DataFrame(test_report).transpose()
        if args.crossvalidation:
            test_report_name = "test_class_report_last_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            test_results_name = "test_results_last_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            figure_name = "test_cm_last_cv{}_fold{}.png".format(run_idx, args.val_fold_idx)
        else:
            test_report_name = "test_class_report_last.csv"
            test_results_name = "test_results_last.csv"
            figure_name = "test_cm_last.png"
        test_report_df.to_csv(os.path.join(ckpt_dir, test_report_name), index=False)

        # 保存所有的预测概率值
        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        pd.DataFrame(results_dict).to_csv(os.path.join(ckpt_dir, test_results_name), index=False)

        # 少样本学习不需要微调模型
        assert args.part_tuning == False
    if (mode == 'validation') and (epoch >= 7) and (epoch < 10):
        epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch))
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        logger.warning(f"mode was: {mode} so will be saving evaluation results to file!")

        # 将预测结果保存到文件
        test_report = classification_report(alllabels, allpreds, target_names=class_labels, output_dict=True)
        test_report_df = pd.DataFrame(test_report).transpose()
        if args.crossvalidation:
            test_report_name = "test_class_report_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            test_results_name = "test_results_cv{}_fold{}.csv".format(run_idx, args.val_fold_idx)
            figure_name = "test_cm_cv{}_fold{}.png".format(run_idx, args.val_fold_idx)
        else:
            if args.transcription in ['cnntdnn', 'fusionshujie', "sys14_26.4", "sys18_25.9"]:  # "cnntdnn":
                test_report_name = "test_{}_{}_class_report.csv".format(args.transcription, args.asr_format)
                test_results_name = "test_{}_{}_results.csv".format(args.transcription, args.asr_format)
                figure_name = "test_{}_{}_cm.png".format(args.transcription, args.asr_format)
            elif args.transcription == "chas":
                test_report_name = "test_class_report.csv"
                test_results_name = "test_results.csv"
                figure_name = "test_cm.png"
            else:
                NotImplemented
        logger.warning('save to {}'.format(os.path.join(epoch_dir, test_report_name)))
        test_report_df.to_csv(os.path.join(epoch_dir, test_report_name), index=False)

        # 保存所有的预测概率
        results_dict = {}
        results_dict['id'] = allids
        results_dict['labels'] = alllabels
        results_dict['pred_labels'] = allpreds
        results_dict['logits'] = alllogits
        results_dict['probas'] = allscores
        pd.DataFrame(results_dict).to_csv(os.path.join(epoch_dir, test_results_name), index=False)

        if args.last_ckpt:
            if not args.crossvalidation:
                torch.save(prompt_model.state_dict(), os.path.join(epoch_dir, "checkpoint.ckpt"))
            else:
                torch.save(prompt_model.state_dict(),
                           os.path.join(epoch_dir, "checkpoint_cv{}_fold{}.ckpt".format(run_idx, args.val_fold_idx)))

    return val_loss, acc, prec_weighted, prec_macro, recall_weighted, recall_macro, f1_weighted, f1_macro, roc_auc_weighted, roc_auc_macro


# TODO - add a test function to load the best checkpoint and obtain metrics on all test data. Can do this post training but may be nicer to do after training to avoid having to repeat.

def test_evaluation(prompt_model, ckpt_dir, dataloader, epoch_num=None):
    # 一旦模型已经训练，我们想要加载最好的模型检查带你，并且在测试集上进行测试，同时将预测概率保存到文件中
    # 在代码中需要添加一个测试函数，该函数可以加载已保存的最佳检查点（best checkpoint），并在所有测试数据上计算评估指标。
    # 该注释建议在训练后添加这个测试函数，因为这样做可以避免重复进行训练。
    if args.last_ckpt:
        assert epoch_num != None
        epoch_dir = os.path.join(ckpt_dir, "epoch{}".format(epoch_num))
        if not args.crossvalidation:
            loaded_model = torch.load(os.path.join(epoch_dir, "checkpoint.ckpt"))
        else:
            loaded_model = torch.load(
                os.path.join(epoch_dir,
                             "checkpoint_cv{}_fold{}.ckpt".format(run_idx,
                                                                  args.val_fold_idx)
                             )
            )
    if not args.no_ckpt:
        if not args.crossvalidation:
            loaded_model = torch.load(os.path.join(ckpt_dir, "best-checkpoint.ckpt"))
        else:
            loaded_model = torch.load(
                os.path.join(ckpt_dir, "best-checkpoint_cv{}_fold{}.ckpt".format(run_idx,
                                                                                 args.val_fold_idx)
                             )
            )

    prompt_model.load_state_dict(state_dict=loaded_model)
    print("cuda_device", cuda_device)
    prompt_model.to(cuda_device)

    # then run evaluation on test_dataloader
    if args.last_ckpt:
        (test_loss, test_acc, test_prec_weighted, test_prec_macro,
         test_recall_weighted, test_recall_macro, test_f1_weighted,
         test_f1_macro, test_auc_weighted, test_auc_macro) = evaluate(
            prompt_model,
            mode='test', dataloader=dataloader, epoch=epoch_num)
    else:
        (test_loss, test_acc, test_prec_weighted, test_prec_macro,
         test_recall_weighted, test_recall_macro, test_f1_weighted,
         test_f1_macro, test_auc_weighted, test_auc_macro) = evaluate(
            prompt_model,
            mode='test', dataloader=dataloader)
    if args.tensorboard:
        # write to tensorboard
        writer.add_scalar("test/loss", test_loss, 0)
        writer.add_scalar("test/balanced_accuracy", test_acc, 0)
        writer.add_scalar("test/precision_weighted", test_prec_weighted, 0)
        writer.add_scalar("test/precision_macro", test_prec_macro, 0)
        writer.add_scalar("test/recall_weighted", test_recall_weighted, 0)
        writer.add_scalar("test/recall_macro", test_recall_macro, 0)
        writer.add_scalar("test/f1_weighted", test_f1_weighted, 0)
        writer.add_scalar("test/f1_macro", test_f1_macro, 0)

        # TODO add binary classification metrics e.g. roc/auc
        writer.add_scalar("test/auc_weighted", test_auc_weighted, 0)
        writer.add_scalar("test/auc_macro", test_auc_macro, 0)

        # # add cm to tensorboard
        # writer.add_figure("test/Confusion_Matrix", cm_figure, 0)


# if refactor this has to be run before any training has occured
if args.zero_shot:
    logger.info("Obtaining zero shot performance on test set!")

    (zero_loss, zero_acc, zero_prec_weighted, zero_prec_macro,
     zero_recall_weighted, zero_recall_macro, zero_f1_weighted,
     zero_f1_macro, zero_auc_weighted, zero_auc_macro) = evaluate(
        prompt_model, test_data_loader, mode='last')

    if args.tensorboard:
        writer.add_scalar("zero_shot/loss", zero_loss, 0)
        writer.add_scalar("zero_shot/balanced_accuracy", zero_acc, 0)
        writer.add_scalar("zero_shot/precision_weighted", zero_prec_weighted, 0)
        writer.add_scalar("zero_shot/precision_macro", zero_prec_macro, 0)
        writer.add_scalar("zero_shot/recall_weighted", zero_recall_weighted, 0)
        writer.add_scalar("zero_shot/recall_macro", zero_recall_macro, 0)
        writer.add_scalar("zero_shot/f1_weighted", zero_f1_weighted, 0)
        writer.add_scalar("zero_shot/f1_macro", zero_f1_macro, 0)

        # TODO add binary classification metrics e.g. roc/auc
        writer.add_scalar("zero_shot/auc_weighted", zero_auc_weighted, 0)
        writer.add_scalar("zero_shot/auc_macro", zero_auc_macro, 0)

        # # add cm to tensorboard
        # writer.add_figure("zero_shot/Confusion_Matrix", zero_cm_figure, 0)

# run training

logger.warning(f"do training : {do_training}")
if do_training:
    logger.warning("Beginning full training!")
    train(prompt_model, train_data_loader, args.num_epochs, ckpt_dir)

elif not do_training:
    logger.warning("No training will be performed!")
print('-----------------------------------------------------')
print('7、模型训练完毕！')
print('-----------------------------------------------------')
# run on test set if desired

if args.run_evaluation:
    logger.warning("Running evaluation on test set using best checkpoint!")
    print('seed', args.seed)
    for epoch_num in [7, 8, 9]:
        test_evaluation(prompt_model, ckpt_dir, test_data_loader, epoch_num)

# write the contents to file
if args.tensorboard:
    writer.flush()

