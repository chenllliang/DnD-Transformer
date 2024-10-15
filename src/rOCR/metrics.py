import os
import json
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer

def edit_distance(str1, str2):
    m, n = len(str1), len(str2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  
            else:
                dp[i][j] = min(dp[i - 1][j],    
                               dp[i][j - 1],    
                               dp[i - 1][j - 1] 
                              ) + 1

    return dp[m][n]

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = scorer.score(reference, hypothesis)
    
    rouge1 = scores['rouge1']
    rouge2 = scores['rouge2']
    rougeL = scores['rougeL']
    
    return scores




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--predict_dir", type=str, default=0, help="")
    parser.add_argument("--gt_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")

    args = parser.parse_args()
    

    
    with open(args.predict_dir, 'r') as fp:
        predict_list = json.load(fp)
    
    gt_list = []
    with open(args.gt_dir, 'r') as fp:
        for line in fp.readlines():
            for key, item in json.loads(line).items():
                gt_list.append(item['text'])
    
    if "text" in args.predict_dir:
        with open(args.gt_dir, 'r') as fp:
            for line in fp.readlines():
                for key, item in json.loads(line).items():
                    gt_list.append(item['text'])
    elif "ARXIV" in args.predict_dir:
        for root, dirs, files in os.walk(args.images_dir):
            for file_name in files:
                file_path = os.path.join(args.images_dir, file_name)
                gt_list.append(item['text'])
    else:
        raise ValueError("Neither ARXIV dataset nor text dataset")
    
    rouge_1_precision_list = []
    rouge_1_recall_list = []
    rouge_1_F1_list = []

    rouge_2_precision_list = []
    rouge_2_recall_list = []
    rouge_2_F1_list = []

    rouge_L_precision_list = []
    rouge_L_recall_list = []
    rouge_L_F1_list = []
    
    distances = []

    for index in tqdm(range(len(predict_list))):

        # 将参考文本拼接为一个字符串
        reference_text = gt_list[index]
        hypothesis_text = predict_list[index]

        # bleu_score = calculate_bleu(reference_sentences, candidate_sentence)
        scores = calculate_rouge(reference_text, hypothesis_text)

        rouge_1_precision_list.append(scores['rouge1'].precision)
        rouge_1_recall_list.append(scores['rouge1'].recall)
        rouge_1_F1_list.append(scores['rouge1'].fmeasure)

        rouge_2_precision_list.append(scores['rouge2'].precision)
        rouge_2_recall_list.append(scores['rouge2'].recall)
        rouge_2_F1_list.append(scores['rouge2'].fmeasure)
        
        rouge_L_precision_list.append(scores['rougeL'].precision)
        rouge_L_recall_list.append(scores['rougeL'].recall)
        rouge_L_F1_list.append(scores['rougeL'].fmeasure)
        
    
    print(f"ROUGE 1 Precision: {sum(rouge_1_precision_list)/len(rouge_1_precision_list)}")
    print(f"ROUGE 1 Recall: {sum(rouge_1_recall_list)/len(rouge_1_recall_list)}")
    print(f"ROUGE 1 F1: {sum(rouge_1_F1_list)/len(rouge_1_F1_list)}")

    print(f"ROUGE 2 Precision: {sum(rouge_2_precision_list)/len(rouge_2_precision_list)}")
    print(f"ROUGE 2 Recall: {sum(rouge_2_recall_list)/len(rouge_2_recall_list)}")
    print(f"ROUGE 2 F1: {sum(rouge_2_F1_list)/len(rouge_2_F1_list)}")

    print(f"ROUGE L Precision: {sum(rouge_L_precision_list)/len(rouge_L_precision_list)}")
    print(f"ROUGE L Recall: {sum(rouge_L_recall_list)/len(rouge_L_recall_list)}")
    print(f"ROUGE L F1: {sum(rouge_L_F1_list)/len(rouge_L_F1_list)}")
    
    
    results = {
        "Rouge_1_Precision": sum(rouge_1_precision_list)/len(rouge_1_precision_list),
        "ROUGE_1_Recall": sum(rouge_1_recall_list)/len(rouge_1_recall_list),
        "ROUGE_1_F1": sum(rouge_1_F1_list)/len(rouge_1_F1_list),
        "ROUGE_2_Precision": sum(rouge_2_precision_list)/len(rouge_2_precision_list),
        "ROUGE_2_Recall": sum(rouge_2_recall_list)/len(rouge_2_recall_list),
        "ROUGE_2_F1": sum(rouge_2_F1_list)/len(rouge_2_F1_list),
        "ROUGE_L_Precision": sum(rouge_L_precision_list)/len(rouge_L_precision_list),
        "ROUGE_L_Recall": sum(rouge_L_recall_list)/len(rouge_L_recall_list),
        "ROUGE_L_F1": sum(rouge_L_F1_list)/len(rouge_L_F1_list),
    }
    
    with open(os.path.join(args.output_dir), "w") as fp:
        json.dump(results, fp)