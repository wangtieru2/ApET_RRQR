import os
import json
import argparse


def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    normalized_answers = []
    for answer in answers:
        text = answer['text']

        # Only keep the first sentence.
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        normalized_answers.append('no' if 'No' in words or 'not' in words or 'no' in words else 'yes')

    label_list = [0 if label == 'no' else 1 for label in label_list]
    pred_list = [0 if answer == 'no' else 1 for answer in normalized_answers]

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP) if TP + FP else 0.0
    recall = float(TP) / float(TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0.0
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'yes_ratio': yes_ratio,
        'tp': TP,
        'fp': FP,
        'tn': TN,
        'fn': FN,
        'samples': len(pred_list),
    }
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]

    category_metrics = {}
    for file in sorted(os.listdir(args.annotation_dir)):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        category_metrics[category] = eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")

    macro_f1 = sum(m['f1'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0
    macro_acc = sum(m['accuracy'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0
    macro_precision = sum(m['precision'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0
    macro_recall = sum(m['recall'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0
    macro_yes_ratio = sum(m['yes_ratio'] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0

    result = {
        'categories': category_metrics,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_yes_ratio': macro_yes_ratio,
        'pope_score': macro_f1 * 100.0,
    }

    print('pope_macro_f1\t{:.4f}'.format(macro_f1))
    print('pope_macro_accuracy\t{:.4f}'.format(macro_acc))
    print('pope_score\t{:.2f}'.format(macro_f1 * 100.0))

    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
