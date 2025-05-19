import os
import re
import pylangacq
import pandas as pd
from pip._vendor import chardet


# 创建一个读取txt的函数，并将txt中的信息转换成对应的csv文件
def gene_csv(txt_path, csv_path,cd_train_meta_data=None):
    # 读取原始文本文件
    with open(txt_path, 'r') as file:
        lines = file.readlines()[1:]

    # 提取每一行中的数据
    data = []
    for line in lines:
        line = line.strip().split(';')
        data.append(line)

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['ID', 'age', 'gender', 'mmse'])

    # 添加transcription列
    df['transcription'] = ''
    df['label'] = 1

    # 保存为CSV文件
    df.to_csv(csv_path, index=False)

def combine_and_label_data(txt_file1, txt_file2,csv_path):
    # 读取第一个txt文件
    with open(txt_file1, 'r') as file:
        lines1 = file.readlines()

    # 读取第二个txt文件
    with open(txt_file2, 'r') as file:
        lines2 = file.readlines()

    # 提取第一个txt文件中的数据，并添加标签为1
    data1 = [{'text': line.strip(), 'label': 1} for line in lines1]

    # 提取第二个txt文件中的数据，并添加标签为0
    data2 = [{'text': line.strip(), 'label': 0} for line in lines2]

    # 合并两个数据列表
    combined_data = data1 + data2

    # 创建DataFrame
    df = pd.DataFrame(combined_data)
    # 添加transcription列
    df['transcription'] = ''

    # 保存为CSV文件
    df.to_csv(csv_path, index=False)
import re

def clean_utterance(utterance):
    # 使用正则表达式移除时间戳和其他非话语文本内容
    cleaned_utterance = re.sub(r".*?", "", utterance)
    return cleaned_utterance.strip()

# 修改后的extract_transcription函数
def extract_transcription(csv_file, cha_dir):
    df = pd.read_csv(csv_file)
    combined_transcriptions = []
    for index, row in df.iterrows():
        participant_id = row['ID'].strip()
        cha_file_path = os.path.join(cha_dir, participant_id + '.cha')
        participant_transcription = []  # 用于存储单个参与者的所有话语文本
        if os.path.exists(cha_file_path):
            chat = pylangacq.read_chat(cha_file_path)
            # 由于不支持clean参数，我们不在这里清洗数据
            par_utterances = chat.utterances(participants="PAR")
            for utterance_obj in par_utterances:
                # 从每个Utterance的tokens属性中提取单词并拼接
                words = [token.word for token in utterance_obj.tokens]
                sentence = " ".join(words)
                participant_transcription.append(sentence)
            print(participant_transcription)
        df.at[index, 'transcription'] = " ".join(participant_transcription)
    # 将对应的csv文件保存
    df.to_csv(csv_file, index=False)
    return df

# 数据切分，实现K折交叉验证
import json
from math import ceil

def create_ten_fold_cross_validation(train_speakers, test_speakers):
    num_folds = 10
    num_train_speakers = len(train_speakers)
    num_test_speakers = len(test_speakers)

    # Calculate number of speakers in each fold for train and test sets
    speakers_per_fold_train = ceil(num_train_speakers / num_folds)
    speakers_per_fold_test = ceil(num_test_speakers / num_folds)

    ten_fold_data = []

    # Split train speakers into folds
    for fold in range(num_folds):
        start_train_idx = fold * speakers_per_fold_train
        end_train_idx = min((fold + 1) * speakers_per_fold_train, num_train_speakers)
        train_fold = train_speakers[start_train_idx:end_train_idx]

        start_test_idx = fold * speakers_per_fold_test
        end_test_idx = min((fold + 1) * speakers_per_fold_test, num_test_speakers)
        test_fold = test_speakers[start_test_idx:end_test_idx]

        ten_fold_data.append({
            "train_speakers": train_fold,
            "test_speakers": test_fold
        })

    # Save the ten-fold data to JSON file
    with open("ten_fold_1.json", "w") as f:
        json.dump(ten_fold_data, f, indent=4)

def gene_adress_m_dataset(sour_path,target_path,csv_path,mode = 'train'):
    # 定义函数用于处理 transcript_path 中的 JSON 文件并拼接 text 字段
    def extract_text(sour_path1,transcript_path1):
        transcript_file = os.path.join(sour_path1, transcript_path1)
        with open(transcript_file, "r") as file:
            data = json.load(file)
            all_text = ""
            for item in data:
                if "text" in item:
                    all_text += item["text"]
            return all_text

    # 读取原始数据
    df = pd.read_csv(csv_path)
    df["joined_all_par_trans"] = df.apply(lambda x: extract_text(sour_path, x["transcript_path"]), axis=1)
    df["id"] = df["audio_path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    selected_columns = df[["id","age", "gender", "ad", "joined_all_par_trans"]]

    # 输出并进行保存
    print(selected_columns.head())
    selected_columns.to_csv(os.path.join(target_path,f"{mode}.csv"), index=False)


def add_computer_description(csv_path,pic_type):
    # 检测文件编码
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read())

    # 使用检测到的编码格式读取文件
    data = pd.read_csv(csv_path, encoding=result['encoding'])

    # 根据pic_type的值添加不同的内容
    for index, row in data.iterrows():
        if pic_type == 'cookie_theft' and row['id'].startswith('adrs'):
            data.at[index, 'pd'] = "The image depicts a scene inside a kitchen. There is a woman standing near the sink, washing dishes. Next to her, on the countertop, there's a cat lying down, possibly sleeping or just resting. In the background, there are two children. One child is standing on a step stool and reaching into a cupboard labeled \"COOKIE JAR,\" while the other child stands beside the step stool, looking up towards the first child, perhaps waiting for a cookie. The scene captures a common household moment, where kids might be trying to sneak some cookies while an adult is occupied with chores."
        else:
            data.at[index, 'pd'] = " I see a ring and a ring. This is a wooden stick, it is made of wood and it is made of wood. I don't know if they call this animal. It's like they've taken it and it's close to this animal. If it's a Greek animal, I don't know what it is. I see a mirror here."
        # 可以添加更多的条件根据需要

    # 将修改后的内容保存到原始文件中
    data.to_csv(csv_path, index=False)
    pass

def merge_csv_text_to_txt(csv_files, output_file,root_path):
    """
    将多个CSV文件中的"text"关键字合并成一个TXT文件，一个样本一行。

    参数：
    - csv_files：包含CSV文件名的列表。
    - output_file：输出的TXT文件名。

    返回值：
    无。
    """
    # 读取每个CSV文件中的"text"列并合并到一个列表中
    text_list = []
    ad_text_list = []

    for file in csv_files:
        # 检测文件编码
        with open(os.path.join(root_path,file), 'rb') as f:
            result = chardet.detect(f.read())
        df = pd.read_csv(os.path.join(root_path,file),encoding=result['encoding'])
        if "joined_all_par_trans" in df.columns:
            text_list.extend(df["joined_all_par_trans"].tolist())
        if "ad" in df.columns:
            ad_text_list.extend(df["ad"].tolist())

    # 将列表中的文本内容写入到一个TXT文件中，每个样本一行
    with open(os.path.join(root_path,output_file), "w", encoding="utf-8") as f:
        for text,ad in zip(text_list,ad_text_list):
            if ad == 0:
                f.write(text + "   "+ 'The diagnosis is healthy.' + "\n")
            else:
                f.write(text + "   "+ 'The diagnosis is dementia.' + "\n")


if __name__ == '__main__':
    # ADReSS2020数据集处理方式
    # # 测试的csv文件
    # test_cha_root_path = r'/home/public/gl/Dataset/ADReSS-2020/test/ADReSS-IS2020-data/test/transcription/'
    # test_meta_data = r'/home/public/gl/Dataset/ADReSS-2020/test/ADReSS-IS2020-data/test/test_results.txt'
    #
    # # 训练的csv文件
    # train_cha_root_path = r'/home/public/gl/Dataset/ADReSS-2020/train/train/transcription/cc/'
    # cd_train_meta_data = r'/home/public/gl/Dataset/ADReSS-2020/train/train/cd_meta_data.txt'
    # cc_train_meta_data = r'/home/public/gl/Dataset/ADReSS-2020/train/train/cc_meta_data.txt'
    #
    # # 生成csv文件的目标保存文件
    # target_save_path = r'/home/public/gl/MultiDetection/PromptADDetection/data/'
    # train_csv_path = target_save_path + 'train_cc.csv'
    # test_csv_path = target_save_path + 'test.csv'
    #
    # # 生成csv文件
    # # combine_and_label_data(cd_train_meta_data,cc_train_meta_data, train_csv_path)
    # gene_csv(cc_train_meta_data,train_csv_path)
    #
    # #拼接对应的csv文件
    # # train_df = extract_transcription(train_csv_path, train_cha_root_path)
    # test = extract_transcription(train_csv_path, train_cha_root_path)
    # # train = extract_transcription(test_csv_path, test_cha_root_path)

    # ADReSS-M数据集处理
    # project_root = r'/home/public/gl/MultiDetection/egemaps_disfluency'
    # train_meta_data = r'/home/public/gl/MultiDetection/egemaps_disfluency/data/processed/en_balanced/data_train.csv'
    # val_meta_data = r'/home/public/gl/MultiDetection/egemaps_disfluency/data/processed/en_balanced/data_val.csv'
    # test_meta_data = r'/home/public/gl/MultiDetection/egemaps_disfluency/data/processed/gr_test/data.csv'
    # gr_sample_meta_data = r'/home/public/gl/MultiDetection/egemaps_disfluency/data/processed/gr_sample/data.csv'
    # target_path = r'/home/public/gl/MultiDetection/PromptADDetection/data/adress_m'
    # gene_adress_m_dataset(project_root, target_path, gr_sample_meta_data, mode='gr_sample')

    # 增加计算机描述
    # train_csv_path = r'/home/public/gl/MultiDetection/PromptADDetection/data/adress_m_compare/train.csv'
    # test_csv_path = r'/home/public/gl/MultiDetection/PromptADDetection/data/adress_m_compare/test.csv'
    # add_computer_description(train_csv_path,'cookie_theft')
    # add_computer_description(test_csv_path,'cookie_theft')

    # 提取pd中的文本内容
    root_path = r'/home/public/gl/MultiDetection/PromptADDetection/data/adress/'
    csv_files = ["train.csv", "test.csv"]
    output_file = "merged_text_label.txt"

    merge_csv_text_to_txt(csv_files, output_file,root_path)