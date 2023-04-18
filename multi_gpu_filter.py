import argparse
import json
import os

import jsonlines
import torch.cuda
import transformers
from tqdm import tqdm

from add_filter_relations import prepare_triplet
from utils.async_io import run_async_in_batches, _queue_t, Batch


def gpu_nli(
        rank: int,
        queue_in: _queue_t,
        queue_out: _queue_t,
):
    model_name_or_path = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model = model.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16).to(rank).eval()
    print(f"Loaded model {model_name_or_path} on GPU {rank}!")

    while batch := queue_in.get():
        texts = [item[1] for item in batch]
        articles = [item[0] for item in batch]
        # print(texts[0])
        inputs = tokenizer(texts, return_tensors="pt", max_length=256, padding="longest", truncation=True)
        inputs = inputs.to(rank)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs.logits.softmax(dim=-1)
        scores = scores[..., model.config.label2id['entailment']]
        queue_out.put(Batch.from_batch(batch, list(zip(articles, scores.tolist()))))
    return True


def main(data_dir: str, action: str = 'score', filter_score: float = 0.75, out_file: str = None, batch_size: int = 1024):
    assert out_file is not None
    # with jsonlines.open(f'out_clean/{"/".join(folder_input.split("/")[1:])}.jsonl', mode='w') as writer:
    articles: list[dict] = []
    article_counter = 0
    work = []
    print(f"{data_dir=}")
    for k, j, y in os.walk(data_dir):
        for file_name in y:
            with jsonlines.open(k + '/' + file_name) as reader:
                for i, article in tqdm(enumerate(reader)):
                    previous = []
                    triples_list = []
                    texts = []
                    for triple in article['triples']:
                        if triple['subject']['boundaries'] != None and triple['object']['boundaries'] != None and (
                        triple['subject']['boundaries'], triple['object']['boundaries']) not in previous:
                            previous.append((triple['subject']['boundaries'], triple['object']['boundaries']))
                            triples_list.append(triple)
                            texts.append(prepare_triplet(triple['subject'], triple['object'], article['text'],
                                                         triple["predicate"]))
                        elif (triple['subject']['boundaries'], triple['object']['boundaries']) not in previous:
                            distance = 1000000
                            for entity in article['entities']:
                                if entity['uri'] == triple['subject']['uri']:
                                    if abs(min(triple['object']['boundaries']) - min(
                                            entity['boundaries'])) < distance:
                                        subject_entity = entity
                                        distance = abs(
                                            min(triple['object']['boundaries']) - min(entity['boundaries']))
                            triple['subject'] = subject_entity
                            previous.append((triple['subject']['boundaries'], triple['object']['boundaries']))
                            triples_list.append(triple)
                            texts.append(prepare_triplet(subject_entity, triple['object'], article['text'],
                                                         triple["predicate"]))

                    if len(texts) == 0:
                        continue
                    work.extend((len(articles), text) for text in texts)
                    articles.append(article)
            if len(work) > 0:
                break

    print(f"{len(work)=}")
    outputs = run_async_in_batches(
        work, batch_size=batch_size, async_fn=gpu_nli, n_proc=torch.cuda.device_count(),
    )

    triple_counter = 0
    last_article = -1
    for (article_id, entailment_score) in tqdm(outputs, desc="entailment"):
        if article_id != last_article:
            last_article = article_id
            triple_counter = 0
        articles[article_id]["triples"][triple_counter]["confidence"] = entailment_score
        triple_counter += 1

    if action == "filter":
        for article in tqdm(articles, desc="filter"):
            article["triples"] = [
                triple for triple in article["triples"]
                if triple["confidence"] >= filter_score or triple["predicate"]["uri"] in ["P569", "P570"]
            ]

    with open(out_file, 'w') as f:
        for article in tqdm(articles, desc="json"):
            f.write(json.dumps(article) + "\n")
    print(f"Done!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, required=True)
    args.add_argument('--action', type=str, choices=['score', 'filter'], required=True)
    args.add_argument('--out_file', type=str, required=True)
    args.add_argument('--batch_size', type=int, required=False, default=1024)

    hparams = args.parse_args()
    main(**hparams.__dict__)
