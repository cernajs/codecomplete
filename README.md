## Structure of data

- I decided to use my repository with deep learning projects of various kinds and some very easy dataset containing
40 functions each of length max 8 lines
- for this i created automated dataset that takes path to folder which python files, each file is then separated into functions.
For each function i randomly drop any line except the name and choose that as my fill in the middle target.
- Inside `dataset.py` the dataset the `CodeCompletionDataset` implements pytorch dataset with two helper classes.
`Element` which serves as single function encapsulation. It takes preffix, suffix and middle as strings and returns
on call of `.elements` tokenized input as `input_ids`, `attention_mask` and `target_value`. For purpose of batching
I also created `BatchElement`, which takes `List[Element]` and concatenate them into single padded tensor.

## Inference

- First I tried inference on simple dataset which i tought would be easy task for the model even of this size. I observed
variety of outputs while trying same input. This was easily solved by setting `do_sample=False`, which didnt work well even
with low `temperature` and controlling `top_p` probability. I tested sampling and temperature because it always worked nicely
for me when fine tuning gpt-2 of various sizes. I also set `num_beams=5` so that the model performes beam_seach for even more strict determinism controll.
Lastly i did inference only on cpu. On gpu i sometimes did get different results, which is to be expected, but nothing in comparison
to sampling.

## Results

- When testing on small dataset the only real problem is that the model sometimes overgenerates, which is natural based on
simplicity of the code that is completed, for some lines would be enough to genereta around 4 tokens assuming token beeing
about 2-3 characters long.

```py
Input: def median(lst):
    [MISSING_PART]
        n = len(lst)
    mid = n // 2
    if n % 2 == 0:
        median_value = (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    else:
        median_value = sorted_lst[mid]
    return median_value
Generated:     sorted_lst = sorted(lst) | Target:     sorted_lst = sorted(lst)

            CHRF: 1.0
            | Exact Match: True
            | WER: 0.0
            | BLEU: 1.0
            | IOU: 1.0
            | Cosine Similarity: 1.0
```

- In rare cases as seen above the model exactly generates what was missing, in others it overgenerates

```py
Input: def gcd(a, b):
    while b:
        a, b = b, a % b
    [MISSING_PART]

Generated:
    return a

def lcm(a, b):
    while | Target:     return a

            CHRF: 0.34146341463414637
            | Exact Match: False
            | WER: 0.6666666666666666
            | BLEU: 0.1353352832366127
            | IOU: 0.3333333333333333
            | Cosine Similarity: 0.9503220319747925
```

- Above could still be seen as valid output. Generally with simple dataset like this the model does not have much issues.

- Using my pytorch dataset, the model had much harder time completing what was missing, even tho the context was much bigger
than in my simpler dataset. From what i observed the model was good in generating following.

- Simple code completion of specific function

```py
Input: def main(args: argparse.Namespace) -> dict[str, float]:
    keras.utils.set_random_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    mnist = MNIST(size={"train": 5_000})
    model = keras.Sequential()
    model.add(keras.layers.Rescaling(1 / 255))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(args.dropout))
    for hidden_layer in args.hidden_layers:
        model.add(keras.layers.Dense(hidden_layer, activation="relu"))
        model.add(keras.layers.Dropout(args.dropout))
    model.add(keras.layers.Dense(MNIST.LABELS, activation="softmax"))
    def label_smoothing(gold, alpha=args.label_smoothing):

    [MISSING_PART]
        mnist.train.data['labels'] = label_smoothing(mnist.train.data['labels'])
    mnist.dev.data['labels'] = label_smoothing(mnist.dev.data['labels'])
    mnist.test.data['labels'] = label_smoothing(mnist.test.data['labels'])
    optimizer = keras.optimizers.AdamW(
        weight_decay=args.weight_decay
    )
    optimizer.exclude_from_weight_decay(var_names=['bias'])
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    tb_callback = TorchTensorBoardCallback(args.logdir)
    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}
Generated: return (gold - gold.min()) / | Target: return (gold - gold.min()) / (gold.max() - gold.min())
```

- Above generation was probably stop due to low number of `max_new_tokens`.

- The model also performed well when completing simple metrics

```py
Input: model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(),
<fim_suffix>
    )
tb_callback = TorchTensorBoardCallback(args.logdir)
logs = model.fit(
    mnist.train.data["images"], mnist.train.data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
    callbacks=[tb_callback],
)
return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}

Generated: metrics=[keras.metrics.CategoricalAccuracy()], Target: metrics=[keras.metrics.CategoricalAccuracy()],
```

- Also the model did well when completing parenthesis or single characters on line like so.
- What the model didnt do well was every kind of if/else statements and more complexe lines. In those cases I either
go empty lines or just copy of next line. Sometimes It generated valid code but if forms like `a = a` and so on.


## metrics

- I implemented `chrf` and `exact-match` as was suggested by assignment. I also implemented `BLEU` even tho conceptually close
to `chrf` just for comparison of two different n-gram comparisons. Next I implemented `IOU` as intersection over union. I used
`IOU` in various bounding-box related vision tasks and tought it would add some insight, mainly when our model overgenerates, but
the correct code completion is still in the generated text.

```py
Input: def remove_duplicates(lst):
    unique_list = []
    for item in lst:
    [MISSING_PART]
                unique_list.append(item)
    return unique_list
Generated:
        if item not in unique_list:
            unique_list.append | Target:         if item not in unique_list:

            CHRF: 0.6521739130434783
            | Exact Match: False
            | WER: 0.16666666666666666
            | BLEU: 0.8187307530779819
            | IOU: 0.8333333333333334
            | Cosine Similarity: 0.957021176815033
```

- In above example we can see that while `CHRF` being slightly over 0.6, our `IOU` is high since our model generated correct
and slightly overgenerated
- I also tried `WER` which is word level levenstein or edit distance. I know this metric from speech recognition where it makes
more sense. After trying it I wouldnt probably include it next time since it doesnt really add any new information that cant be infered
from other metrics.
- Last I implemented `Cosine Similarity` metric for cases like following example:

```py
Input: def average(lst):
    if not lst:
        return 0
    [MISSING_PART]
        avg = total / len(lst)
    return avg
Generated:  = sum(lst) / len(lst)
    return total

 | Target:     total = sum(lst)

            CHRF: 0.27450980392156865
            | Exact Match: False
            | WER: 0.8333333333333334
            | BLEU: 0.30934850332660563
            | IOU: 0.5
            | Cosine Similarity: 0.9535203576087952
```

- In above example we can see that generated output isnt really conceptually wrong, but every word similary metric fails
to see that. For that i take the embedding of generated output and target, run them trough embedding and the model. On last
hidden stat i compute `Cosine similarity` to see how the model closely relates those two inputs. We can see that the model
sees these two output closely related even tho they are two different lines of code.

## Further improvements

- If we want to use this model as real code complete, the dataset is prepared of finetuning using HuggingFace trainer API.
- When finetuning we must be carefull that we dont overfit and monitor closely how is our model doing. Mainly on smaller GPT2 like models
the risk of overfitting or making the model babble is from my experience very big
