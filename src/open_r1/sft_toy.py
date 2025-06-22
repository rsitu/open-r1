import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format


logger = logging.getLogger(__name__)

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    print("\n===logging setup===\n")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    print(f"log_level: {log_level}")
    logger.info(f"Model parameters {model_args}\n")
    logger.info(f"Script parameters {script_args}\n")
    logger.info(f"Training parameters {training_args}\n")

    print("\n===last checkpoint args===\n")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        print(f"training_args.output_dir: {training_args.output_dir}")
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        print(f"training_args.resume_from_checkpoint: {training_args.resume_from_checkpoint}")
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        print(f"training_args.report_to: {training_args.report_to}")
        init_wandb_training(training_args)
      
    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    print("\n===starting getting datasets===\n")
    dataset = get_dataset(script_args)

    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    if tokenizer.chat_template is None:
        print("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

if __name__ == "__main__":
    print("\n===starting of __main__===\n")
    
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

    print("\n===ending of __main__===\n")
