import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, pipeline, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from huggingface_hub import login
import collections
from tqdm.auto import tqdm
import ast

# Define metric for evaluation
metric = load_metric("squad_v2")

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

api_token = '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'
login(api_token)

model_checkpoint = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(tokenizer.is_fast)

max_length = 384
stride = 128
n_best = 20
max_answer_length = 30

# Load your dataset
train_df = pd.read_csv('D:/python_projects/CI/Education/Hugging_face_test_v0.1/train_dataset.csv')
test_df = pd.read_csv('D:/python_projects/CI/Education/Hugging_face_test_v0.1/test_dataset.csv')

train_df['answers'] = train_df['answers'].apply(ast.literal_eval)
test_df['answers'] = test_df['answers'].apply(ast.literal_eval)

# Adjust the columns to match your dataset
train_df = train_df[['question', 'context', 'answers', 'id']]
test_df = test_df[['question', 'context', 'answers', 'id']]

train_df['id'] = train_df['id'].astype(str)
test_df['id'] = test_df['id'].astype(str)

# Convert the DataFrame to a Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(test_df)

# Preprocess the datasets
train_dataset = train_dataset.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=train_dataset.column_names,
)

val_dataset = val_dataset.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=val_dataset.column_names,
)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

args = TrainingArguments(
    output_dir="D:/python_projects/CI/Education/Hugging_face_test_v0.1/legal_Bert_base/results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="D:/python_projects/CI/Education/Hugging_face_test_v0.1/legal_Bert_base/logs",
    logging_steps=100,
    fp16=False,
    prediction_loss_only=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p.predictions[0], p.predictions[1], val_dataset, val_dataset),
)

trainer.train()

results = trainer.evaluate(val_dataset)
print(results)

trainer.save_model("D:/python_projects/CI/Education/Hugging_face_test_v0.1/legal_Bert_base/fine_tuned_model")
tokenizer.save_pretrained("D:/python_projects/CI/Education/Hugging_face_test_v0.1/legal_Bert_base/fine_tuned_model")

# Use the pipeline for inference
question_answerer = pipeline("question-answering", model="D:/python_projects/CI/Education/Hugging_face_test_v0.1/legal_Bert_base/fine_tuned_model", tokenizer="D:/python_projects/CI/Education/Hugging_face_test_v0.1/legal_Bert_base/fine_tuned_model")

def answer_question(question, context):
    response = question_answerer(question=question, context=context)
    return response['answer']

# Example usage
question = "What is the difference between net profit share and royalty basis share?"
context = """Here, it’s especially important to have a solid grasp of exactly what is meant by these terms. People in the music industry may refer to both as royalties
          but there are subtle differences between royalties paid to a producer on a net profits basis and a royalty-basis.
           The main difference is that a net profit share is calculated after all costs and expenses are deducted, while a royalty-basis,
            share is calculated based on total sales only (even though royalty-basis payments won’t start until recording costs are recouped"""
            
answer = answer_question(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")
