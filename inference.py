from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from dataset import CodeCompletionDataset

from metrics.chrf import chrf
from metrics.exact_match import exact_match
from metrics.wer import wer
from metrics.bleu import bleu
from metrics.iou import iou
from metrics.cosine_similarity import cosine_similarity

checkpoint = "bigcode/tiny_starcoder_py"
#device = "mps"
device = "cpu"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

#print(tokenizer.pad_token_id) -> returns None

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

dataset = CodeCompletionDataset(
    path = "~/dev/pysample/",
    device = device
)

def collate_fn(batch):
    input_ids = []
    attention_masks = []
    target_values = []

    for batch_element in batch:
        input_id, attention_mask, target_value = batch_element.elements(tokenizer)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        target_values.append(target_value)

    if len(batch) != 1:
        # merge into single padded tensor

        input_ids = [seq for batch in input_ids for seq in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        attention_masks = [seq for batch in attention_masks for seq in batch]
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        target_values = [seq for batch in target_values for seq in batch]
        target_values = torch.nn.utils.rnn.pad_sequence(
            target_values, batch_first=True, padding_value=tokenizer.pad_token_id
        )

    else:
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        target_values = torch.cat(target_values, dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'target_values': target_values
    }

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


for batch in dataloader:
    input_ids, attention_mask, target_value = batch['input_ids'], batch['attention_mask'], batch['target_values']

    #skip empty batches
    if input_ids.shape[0] == 0:
        continue

    for i in range(input_ids.shape[0]):
        _input_ids, _attention_mask = input_ids[i].unsqueeze(0).to(device), attention_mask[i].unsqueeze(0).to(device)

        outputs = model.generate(
            input_ids=_input_ids,
            attention_mask=_attention_mask,
            max_new_tokens=15,
            num_beams=5,
            pad_token_id=tokenizer.pad_token_id,
        )


        output = tokenizer.decode(outputs[0])

        output = output.replace("<fim_prefix>", "").replace("[PAD]", "").replace("<|endoftext|>", "")
        output = output.replace("<fim_suffix>", "[MISSING_PART]")

        input, result = output.split("<fim_middle>")

        target_output = tokenizer.decode(target_value[i], skip_special_tokens=True)

        print(f"Input: {input}")
        print(f"Generated: {result} | Target: {target_output}")
        print(f"""
            CHRF: {chrf(result, target_output)}
            | Exact Match: {exact_match(result, target_output)}
            | WER: {wer(result, target_output)}
            | BLEU: {bleu(result, target_output)}
            | IOU: {iou(result, target_output)}
            | Cosine Similarity: {cosine_similarity(result, target_output, tokenizer, model)}
            """)

        print("\n\n\n")
