# standard python imports
import os

# base_model = "../models/llama_base/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
# base_model = "/home/richardarcher/Dropbox/Sci24_LLM_Polarization/project_/weights_local/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
base_model = "/gpfs/home/rka28/alex/all_weights_all_formats/download_from_hf_in_hf_format/Meta-Llama-3.1-8B-Instruct"
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.set_float32_matmul_precision('high')

import polars as pl

# huggingface libraries
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
)
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM # , setup_chat_format

def main():
    import wandb

    wandb.init(
        project="optim00",
        name="essay01",
        config={
            "model_name": "local_training_run_00",
            "task": "response_only",
            "timestamp": "2024.11.18.18_02"
        }
    )

    new_model = "../models/llama_finetuned/"

    PATH_data_to_train_on = "../data/1_clean/training.csv"
    PATH_data_to_test_on = "../data/1_clean/testing.csv"

    nf4_config = BitsAndBytesConfig(
        # load_in_4bit=True,  # NOTE WAS 4 BIT
        load_in_8bit=True,  # NOTE WAS 4 BIT
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Match input dtype
    )
    model = LlamaForCausalLM.from_pretrained(base_model, quantization_config=nf4_config, device_map="auto")

    # print(torch.cuda.is_available())

    # with a bigger gpu could try this
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
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)

    def print_trainable_params(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        print(f"{trainable_percentage:,.2f}% of parameters are trainable")
        print(f"{trainable_params:,.2f} many parameters are trainable")

    print_trainable_params(model)

    def create_prompt(review):
        system_prompt = f"You read student essays reviews and return a score from 0 to 60 that represents your besst guess of the number of rating given by the grader. Return just the number 0, 1, ..., 60 with no context, explanation, or special symbols."
        prompt = f"Here is the review to evaluate: [[[{review}]]]. You read student essays reviews and return a score from 0 to 60 that represents your besst guess of the number of rating given by the grader. Return just the number 0, 1, ..., 60 with no context, explanation, or special symbols."

        return system_prompt, prompt

    df_train = pl.read_csv(PATH_data_to_train_on)
    df_test = pl.read_csv(PATH_data_to_test_on)

    print(f"{df_train.shape=}")
    print(f"{df_test.shape=}")

    lst_system_prompt, lst_prompt = [], []
    for row in df_train.iter_rows(named=True):
        system_prompt, prompt = create_prompt(row["text"])
        lst_system_prompt.append(system_prompt)
        lst_prompt.append(prompt)
    df_train = df_train.with_columns(pl.Series(lst_system_prompt).alias("instruction"),
                                     pl.Series(lst_prompt).alias("input"))
    output = [int(i) for i in df_train["score"].to_list()]
    df_train = df_train.with_columns(pl.Series(output).alias("output"))

    lst_system_prompt, lst_prompt = [], []
    for row in df_test.iter_rows(named=True):
        system_prompt, prompt = create_prompt(row["text"])
        lst_system_prompt.append(system_prompt)
        lst_prompt.append(prompt)
    df_test = df_test.with_columns(pl.Series(lst_system_prompt).alias("instruction"),
                                   pl.Series(lst_prompt).alias("input"))
    output = [int(i) for i in df_test["score"].to_list()]
    df_test = df_test.with_columns(pl.Series(output).alias("output"))

    train_dataset = Dataset.from_polars(df_train)
    test_dataset = Dataset.from_polars(df_test)

    max_seq_length_needed = 1_631

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
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        # optim="paged_adamw_8bit",
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=500,  # Evaluate every 500 steps
        logging_steps=10,
        warmup_steps=500,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        group_by_length=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="ESSAY00"
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        formatting_func=format_but_not_tokenize,
        data_collator=collator,
    )

    trainer.train()

    return "DONE"

if __name__ == "__main__":
    main()