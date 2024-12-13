
local_ = False
if local_:
    base_model = "/home/richardarcher/Dropbox/Sci24_LLM_Polarization/project_/weights_local/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    batch_size = 1
else:
    base_model = "/gpfs/home/rka28/alex/all_weights_all_formats/download_from_hf_in_hf_format/Meta-Llama-3.1-8B-Instruct"
    batch_size = 4

# standard python imports
import os
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import torch

# import pandas as pd
import numpy as np

import torch


from tqdm import tqdm

# huggingface libraries

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # HfArgumentParser,
    # TrainingArguments,
    pipeline,
    # logging,
    LlamaForCausalLM
)
from peft import (
#     LoraConfig,
    PeftModel,
#     prepare_model_for_kbit_training,
#     get_peft_model,
)
# from datasets import load_dataset, Dataset
# from trl import SFTTrainer, setup_chat_format

import wandb

import polars as pl
# import pandas as pd

from transformers.pipelines.pt_utils import KeyDataset

from torch.utils.data import Dataset


# def create_prompt(review):
#     system_prompt = f"You read Yelp reviews and return a number (1, 2, 3, 4, or 5) that represents your besst guess of the number of star ratings that were given by that reviewer. Return just the number 1, 2, 3, 4, or 5, with no context, explanation, or special symbols."
#     prompt = f"Here is the review to evaluate: [[[{review}]]]. Remember, you read Yelp reviews and return a number (1, 2, 3, 4, or 5) that represents your besst guess of the number of star ratings that were given by that reviewer. Return just the number 1, 2, 3, 4, or 5, with no context, explanation, or special symbols."
#
#     return system_prompt, prompt

def create_prompt(review):
    system_prompt = f"You read student essays reviews and return a score from 0 to 60 that represents your besst guess of the number of rating given by the grader. Return just the number 0, 1, ..., 60 with no context, explanation, or special symbols."
    prompt = f"Here is the review to evaluate: [[[{review}]]]. You read student essays reviews and return a score from 0 to 60 that represents your besst guess of the number of rating given by the grader. Return just the number 0, 1, ..., 60 with no context, explanation, or special symbols."

    return system_prompt, prompt

def add_prompts_to_df(df):
    lst_system_prompt, lst_prompt = [], []
    for row in df.iter_rows(named=True):
        system_prompt, prompt = create_prompt(row["text"])
        lst_system_prompt.append(system_prompt)
        lst_prompt.append(prompt)
    df = df.with_columns(pl.Series(lst_system_prompt).alias("system_prompt"), pl.Series(lst_prompt).alias("prompt"))
    return df

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        # embeddings: list of numpy arrays or torch tensors
        # labels: list of scalars
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float16)  # or long, depending on your task

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def df_to_dataset(df, batch_size, model, tokenizer):
    model.eval()

    embeddings = []
    labels = []

    rows = df.to_dicts()  # returns a list of row dictionaries
    with torch.no_grad():
        # USE TQDM LOCAL OR THE IX ON THE CLUSTER
        # for i in tqdm(range(0, len(df), batch_size)):
        for i in range(0, len(df), batch_size):
            if i % (batch_size * 1_000) == 0:
                print(f"CURRENTLY OPERATING ON IX={i}/{len(df)}")
                wandb.log({"ix": i})
            batch_rows = rows[i: i + batch_size]

            # Prepare batched input
            batch_messages = [
                [
                    {"role": "system", "content": r["system_prompt"]},
                    {"role": "user", "content": r["prompt"]}
                ]
                for r in batch_rows
            ]

            # Tokenize the entire batch at once
            inputs_message = tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to("cuda")

            # Single forward pass for the entire batch
            with torch.no_grad():
                outputs = model(
                    inputs_message,
                    output_hidden_states=True,
                    return_dict=True
                )
            # Extract embeddings for the entire batch at once
            hidden_states = outputs.hidden_states
            # Convert to float32 before moving to CPU and then NumPy
            embeddings_batch = hidden_states[-2][:, -1, :].to(dtype=torch.float32).cpu().numpy()

            # Add them to a growing list
            for j, r in enumerate(batch_rows):
                embeddings.append(embeddings_batch[j])
                labels.append(r["score"])

        # Convert to a Dataset
        dataset = EmbeddingDataset(np.array(embeddings), labels)
    return dataset



def main():
    run = wandb.init(
        # Set the project where this run will be logged
        project="optim00",
        name="essay_mymethod00"

    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        tokenizer_file=os.path.join(base_model, 'tokenizer.json'),
        tokenizer_config_file=os.path.join(base_model, 'tokenizer_config.json'),
        special_tokens_map_file=os.path.join(base_model, 'special_tokens_map.json'),
        trust_remote_code=True,
        padding_side='left'
    )

    tokenizer.padding_side = 'left'

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Match input dtype

    )

    model = LlamaForCausalLM.from_pretrained(base_model, quantization_config=nf4_config)

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     device_map="auto",
    # #     device_map="balanced",
        # torch_dtype=torch.bfloat16
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     device_map="auto",
    #     # device_map="balanced",
    #     torch_dtype=torch.bfloat16
    # )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id


    df_train = pl.read_csv("data/imported/training.csv")
    df_train = add_prompts_to_df(df_train)
    # save_path = "data/mymethod/training.pt"

    df_test = pl.read_csv("data/imported/testing.csv")
    df_test = add_prompts_to_df(df_test)
    # save_path = "data/mymethod/testing.pt"
    #
    df_val = pl.read_csv("data/imported/val.csv")
    df_val = add_prompts_to_df(df_val)

    # df_train = add_prompts_to_df(df_train)
    # df_test = add_prompts_to_df(df_test)


    # print("NOW OPERATING ON VAL")
    #dataset_val = df_to_dataset(df_val, batch_size, model, tokenizer)
    #print("NOW SAVING VAL")
    #torch.save(dataset_val, "data/mymethod/val.pt")
    #print("VAL SAVED")

    #print("NOW RUNNING TEST")
    #dataset_test = df_to_dataset(df_test, batch_size, model, tokenizer)
    #print("NOW SAVING TEST")
    #torch.save(dataset_test, "data/mymethod/testing.pt")
    #print("TEST SAVED")

    # print("NOW RUNNING TRAIN")
    # dataset_train = df_to_dataset(df_train, batch_size, model, tokenizer)
    # print("NOW SAVING TRAIN")
    # torch.save(dataset_train, "data/training.pt")
    # print("TRAIN SAVED")

    return 1

if __name__ == "__main__":
    main()
