# standard python imports
import os

local_ = False
if local_:
    base_model = "/home/richardarcher/Dropbox/Sci24_LLM_Polarization/project_/weights_local/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    batch_size = 4
else:
    base_model = "/gpfs/home/rka28/alex/all_weights_all_formats/download_from_hf_in_hf_format/Meta-Llama-3.1-8B-Instruct"
    batch_size = 4
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import polars as pl
import torch

# huggingface libraries
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM # , setup_chat_format
from trl.commands.cli import train

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

def main():
    import wandb

    wandb.init(
        project="optim00",  # Change this to your project name
        name="cluster_run_01",
        config={
            "model_name": "quant_for_hpc",
            "task": "response_only",
            "timestamp": "2024.11.18.18_02"
        }
    )

    new_model = "weights/sft/run02"

    PATH_data_to_train_on = "data/1_train_test_split/df_train.csv"
    PATH_data_to_test_on = "data/1_train_test_split/df_test.csv"

    print("TRY BIGGER GPU")

    nf4_config = BitsAndBytesConfig(
        load_in_8bit=True, # NOTE WAS 4 BIT
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Match input dtype
    )
    #
    model = LlamaForCausalLM.from_pretrained(base_model, quantization_config=nf4_config, device_map="auto")

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     device_map="auto",
    #     device_map="balanced",
        # torch_dtype=torch.bfloat16
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        tokenizer_file=os.path.join(base_model, 'tokenizer.json'),
        tokenizer_config_file=os.path.join(base_model, 'tokenizer_config.json'),
        special_tokens_map_file=os.path.join(base_model, 'special_tokens_map.json'),
        trust_remote_code=True
    )

    tokenizer.pad_token_id = 128004  # tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    model.config.pad_token_id = 128004  # tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    # model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, peft_config)

    def print_trainable_params(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        print(f"{trainable_percentage:.2f}% of parameters are trainable")
        print(f"{trainable_params}many parameters are trainable")

    # Call the function after applying PEFT
    print_trainable_params(model)

    def create_prompt(review):
        system_prompt = f"You read student essays reviews and return a score from 0 to 60 that represents your besst guess of the number of rating given by the grader. Return just the number 0, 1, ..., 60 with no context, explanation, or special symbols."
        prompt = f"Here is the review to evaluate: [[[{review}]]]. You read student essays reviews and return a score from 0 to 60 that represents your besst guess of the number of rating given by the grader. Return just the number 0, 1, ..., 60 with no context, explanation, or special symbols."

        return system_prompt, prompt

    df_train = pl.read_csv(PATH_data_to_train_on)
    df_test = pl.read_csv(PATH_data_to_test_on)

    # df_train = df_train.sample(n=100_000, seed=0)
    # df_test = df_test.sample(n=10_000, seed=0)

    lst_system_prompt, lst_prompt = [], []
    for row in df_train.iter_rows(named=True):
        system_prompt, prompt = create_prompt(row["text"])
        lst_system_prompt.append(system_prompt)
        lst_prompt.append(prompt)
    df_train = df_train.with_columns(pl.Series(lst_system_prompt).alias("instruction"),
                                     pl.Series(lst_prompt).alias("input"))
    output = [int(i) for i in df_train["stars"].to_list()]
    df_train = df_train.with_columns(pl.Series(output).alias("output"))

    lst_system_prompt, lst_prompt = [], []
    for row in df_test.iter_rows(named=True):
        system_prompt, prompt = create_prompt(row["text"])
        lst_system_prompt.append(system_prompt)
        lst_prompt.append(prompt)
    df_test = df_test.with_columns(pl.Series(lst_system_prompt).alias("instruction"),
                                   pl.Series(lst_prompt).alias("input"))
    output = [int(i) for i in df_test["stars"].to_list()]
    df_test = df_test.with_columns(pl.Series(output).alias("output"))

    train_dataset = Dataset.from_polars(df_train)
    test_dataset = Dataset.from_polars(df_test)

    max_seq_length_needed = 1_700

    def format_but_not_tokenize(example):
        test = example["instruction"]
        # assert isinstance(test, list), "Input 'example' must be a list, this is probably because formatting function needs >1 eg"
        # assert not isinstance(test, str), "Input 'example' must be a list, not a string"

        output_texts = []

        if isinstance(test, list):
            K_range = len(test)

            for i in range(K_range):
                row_json = [{"role": "system", "content": example['instruction'][i]},
                            {"role": "user", "content": example['input'][i]},
                            {"role": "assistant", "content": example['output'][i]}]
                text = tokenizer.apply_chat_template(row_json, tokenize=False, add_generation_prompt=False)

                output_texts.append(text)

        elif isinstance(test, str):
            # K_range = 1
            row_json = [{"role": "system", "content": example['instruction']},
                        {"role": "user", "content": example['input']},
                        {"role": "assistant", "content": example['output']}]
            text = tokenizer.apply_chat_template(row_json, tokenize=False, add_generation_prompt=False)

            output_texts.append(text)
        else:
            assert False, "ERROR: WHAT IS GOING INTO FORMAT_BUT_NOT_TOKENIZE???"

        return output_texts

    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # Disable KV cache during training

    training_args = SFTConfig(
        max_seq_length=max_seq_length_needed,
        output_dir=new_model,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,  # 4
        # optim="adamw_torch",
        optim="paged_adamw_32bit",
        # optim="paged_adamw_8bit",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=500,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        # TK TK OPTIM1
        bf16=True,  # was false
        group_by_length=True,
        # TK TK OPTIM2
        gradient_checkpointing=True,  # Enable gradient checkpointing
        report_to="wandb",
        run_name="cluster000"
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        formatting_func=format_but_not_tokenize,
        data_collator=collator,
        # ADDED THE BELOW IF IT BREAKS REMOVE IT OR FIX
        # compute_metrics=custom_evals,  # Add this line
    )

    trainer.train()

    return 1

if __name__ == "__main__":
    main()