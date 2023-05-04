"""Evaluate perplexity.
"""
import fire
import torch

from peft import PeftModel
from torch.utils.data.dataloader import DataLoader
from transformers import LlamaForCausalLM
from tqdm import tqdm
from typing import Optional

from finetune import build_datasets

LOSS_IGNORE_INDEX = torch.nn.CrossEntropyLoss().ignore_index


def evaluate(
    base_model: str,
    data_path: str,
    lora_weights: Optional[str] = None,
    batch_size: int = 32,
):
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    # Defaults duped from finetune#train, except val spl it.
    dataset, _, data_collator = build_datasets(
        base_model=base_model,
        data_path=data_path,
        cutoff_len=256,
        val_set_size=0,
        train_on_inputs=True,
        add_eos_token=False,
        prompt_template_name="alpaca",
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    # On perplexity:
    # Re-weight losses with torch.sum(batch.labels != LOSS_IGNORE_INDEX)?
    # Is therer a better approximation?s
    losses = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            loss = model(
                **{k: v.to(model.device) for k, v in batch.items()}
            ).loss
        losses.append(loss)

    ppl = torch.exp(torch.stack(losses).mean()).item()
    return ppl


if __name__ == "__main__":
    fire.Fire(evaluate)
