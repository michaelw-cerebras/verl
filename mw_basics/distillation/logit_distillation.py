from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import TrainerCallback
import re, random, os

os.environ["WANDB_PROJECT"] = "mw_distillation_rl_basics"
os.environ["WANDB_RUN_NAME"] = "gsm8k_qwen7bto0p5b_lora_kl_ce"

class GSM8KSFTDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length, split, instruction):
        super().__init__()
        assert split in ["train", "test"]
        # Load GSM8K from huggingface
        hf_dataset = load_dataset("openai/gsm8k", "main")
        self.data = hf_dataset[split]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instruction = instruction

        self.pad_id = self.tokenizer.pad_token_id
        # In some tokenizers pad_id is None
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id


    def __len__(self):
        return len(self.data)
    

    ''' 
    NOTE:
    Flow for tokenization of SFT QA pairs (question, answer)
        - [Optional] Add instruction in front of raw question
        - messages = [{"role":"user","content":input_text}]
        - prompt = apply_chat_template (messages, tokenize=False) => add headers and make a full prompt string
            <bos>
            <user_header>
            input_text
            <eot>
            <assistant_header>
        - answer_text = answer + EOS token
        - tokenize answer_text and prompt, do not add special tokens
        - set up labels and attention mask
        - MSL truncation and padding
    
    Notice that add_special_tokens is by default True in tokenizer.encode()
      and might double add some special tokens like BOS and EOS. Turn it off.


    '''
    def __getitem__(self, index):
        item = self.data[index]
        input_text = self.instruction + item["question"]
        # NOTE: Add eos_token to let the model learn when to end generation
        output_text = item["answer"] + self.tokenizer.eos_token
        messages = [
            {'role': 'user', 'content': input_text}
        ]

        # apply_chat_template() will usually add <BOS> and <EOS>
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # add assistant header for generation start
        )

        # So in later pipline, tokenizer.encode() might add those again if we don't set add_special_tokens = False
        prompt_input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        response_input_ids = self.tokenizer.encode(
            output_text,
            add_special_tokens=False,
        )

        input_ids = prompt_input_ids + response_input_ids
        labels = [-100] * len(prompt_input_ids) + response_input_ids
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_seq_length - len(input_ids)

        if padding_length < 0:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        '''
        We should not shift if we later use AutoModelForCausalLM / HF Trainer
        because shifting is automatically while calculating loss there.
        Notice, in output.logits things are still not shifted
        '''
        # input_ids = input_ids[:-1]
        # labels = labels[1:]
        return {
            'input_ids': torch.tensor(input_ids), 
            'attention_mask':torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }



def compute_forward_kl(
        student_logits,
        teacher_logits,
        labels,
        padding_id,
        reduce_method,
        temperature=1.0,
):  
    assert reduce_method in ["mean", "sum"]
    # logits: (batch_size, seq_len, vocab_size)
    # logits at timestep t is actually the predicted token distribution for token[t+1]
    # so the last element of logits is the next token of <eos>, which is meaningless
    # same reason for labels
    student_logits = student_logits[:, :-1, :] / temperature # （batch_size, seq_len - 1, vocab_size）
    teacher_logits = teacher_logits[:, :-1, :] / temperature # （batch_size, seq_len - 1, vocab_size）
    labels = labels[:, 1:] # （batch_size, seq_len - 1)

    # Compute 3 variables needed for forward KL calculation
    student_log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)

    vocab_level_kl = teacher_probs * (teacher_log_probs - student_log_probs) # (batch_size, seq_len - 1, vocab_size)
    token_level_kl = vocab_level_kl.sum(-1) # (batch_size, seq_len - 1)
    
    padding_mask = labels.eq(padding_id)
    token_level_kl = token_level_kl.masked_fill_(padding_mask, 0.0)


    if reduce_method == "mean":
        sequence_level_kl = token_level_kl.sum(-1) / (~padding_mask).sum(-1)
    else:
        sequence_level_kl = token_level_kl.sum(-1)
    return sequence_level_kl



def compute_reverse_kl(
        student_logits,
        teacher_logits,
        labels,
        padding_id,
        reduce_method,
        temperature=1.0,
):  
    assert reduce_method in ["mean", "sum"]
    # kl = sum_across_vocab_x(q(x) * [log q(x) - log p(x)])
    student_logits = student_logits[:, :-1, :] / temperature
    teacher_logits = teacher_logits[:, :-1, :] / temperature
    labels = labels[:, 1:]

    student_log_probs = torch.log_softmax(student_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    student_probs = torch.softmax(student_logits, -1, dtype=torch.float32)

    vocab_level_kl = student_probs * (student_log_probs - teacher_log_probs)
    token_level_kl = vocab_level_kl.sum(-1) # (bsz, seq_len - 1)

    padding_mask = labels.eq(padding_id)
    token_level_kl.masked_fill_(padding_mask, 0.0)

    if reduce_method == "sum":
        sequence_level_kl = token_level_kl.sum(-1)
    else:
        sequence_level_kl = token_level_kl.sum(-1) / (~padding_mask).sum(-1)
    return sequence_level_kl


class KDTrainer(Trainer):
    def __init__(
            self,
            student_model = None,
            teacher_model = None,
            use_entropy = False,
            data_collator = None,
            train_dataset = None,
            val_dataset = None,
            tokenizer = None,
            args = None,
    ):
        super().__init__(
            model=student_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )

        self.teacher_model = teacher_model
        self.use_entropy = use_entropy
        self.teacher_model.eval()

        # Metrics accumulator
        # compute_loss() is invoked for every micro-batch, not every global step
        self._kl_loss_accumulator = 0.0
        self._ce_loss_accumulator = 0.0
        self._accumulation_count = 0
        self._last_logged_step = -1


    # NOTE: you must use the same input interface as compute_loss() in the original HF trainer
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, reduce_method="mean"):
        # In KD, inputs already contains prompt_input_ids and response_input_ids
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        cross_entropy_loss = student_outputs.loss
        # For multi-gpu training, cross_entropy_loss might return the loss from only one gpu, 
        # we need to use mean to aggregate
        if cross_entropy_loss.dim() > 0:
            cross_entropy_loss = cross_entropy_loss.mean()

        student_logits = student_outputs.logits # unshifted! (bsz, seq_len, vocab_size)
        teacher_logits = teacher_outputs.logits # unshifted! (bsz, seq_len, vocab_size)

        # In Qwen 2.5 Family, embedding layer and lm_head could be padded for better parallelism.
        # But the actual tokenizer used is the same across all 2.5 models.
        # Therefore a hard truncation can work here.

        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            student_logits = student_logits[:, :, :min(student_logits.shape[-1], teacher_logits.shape[-1])]
            teacher_logits = teacher_logits[:, :, :min(student_logits.shape[-1], teacher_logits.shape[-1])]

            # Second method: pad student
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)

        labels = inputs["labels"]

        sequence_kl = compute_forward_kl(student_logits, teacher_logits, labels, padding_id=-100, reduce_method=reduce_method, temperature=2.0)
        batch_kl = sequence_kl.mean()

        if self.use_entropy:
            loss = 0.5 * batch_kl + 0.5 * cross_entropy_loss
        else:
            loss = batch_kl

        loss = loss / self.args.gradient_accumulation_steps

        self._kl_loss_accumulator += batch_kl.item()
        self._ce_loss_accumulator += cross_entropy_loss.item()
        self._accumulation_count += 1

        #  you must use the same output interface as compute_loss() in the original HF trainer
        return (loss, student_outputs) if return_outputs else loss
    

    def log(self, logs, start_time=None):
        if self._accumulation_count > 0 and self._last_logged_step != self.state.global_step:
            logs["kl_loss"] = self._kl_loss_accumulator / self._accumulation_count
            logs["ce_loss"] = self._ce_loss_accumulator / self._accumulation_count
            
            self._kl_loss_accumulator = 0.0
            self._ce_loss_accumulator = 0.0
            self._accumulation_count = 0
            self._last_logged_step = self.state.global_step
        
        super().log(logs, start_time)



class GSM8KEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, eval_steps=100, num_samples=100):
        """ 
        eval_steps: after every x global_steps, we do one eval
        num_samples: at each eval_step, we randomly sample from the eval set
        """
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_samples = num_samples

    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        model.eval()
        total_score = 0
        total = 0

        rng = random.Random(state.global_step)
        all_indices = list(range(len(self.eval_dataset.data)))
        if self.num_samples < len(all_indices):
            indices = rng.sample(all_indices, self.num_samples)
        else:
            indices = all_indices
        for idx in indices:
            item = self.eval_dataset.data[idx]
            question = self.eval_dataset.instruction + item["question"]
            
            
            ground_truth = self.extract_solution(item["answer"], method="flexible")
            messages = [{"role": "user", "content": question}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
            generated_solution = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            score = self.compute_score(
                solution_str=generated_solution,
                ground_truth=ground_truth,
                method="flexible",
                format_score=0.0,
                score=1.0
            )
            
            total_score += score
            total += 1
        
        accuracy = total_score / total if total > 0 else 0
        if args.report_to and "wandb" in args.report_to:
            import wandb
            wandb.log({
                "eval/accuracy": accuracy, 
                "eval/total_score": total_score, 
                "eval/total": total
            }, step=state.global_step)
        
        print(f"\n[Step {state.global_step}] Eval Accuracy: {accuracy:.4f} ({total_score}/{total})\n")
        
        model.train()

    def extract_solution(self, solution_str, method="flexible"):
        _SOLUTION_CLIP_CHARS = 300
        assert method in ["strict", "flexible"]

        # Optimization: Regular expression matching on very long strings can be slow.
        # For math problems, the final answer is usually at the end.
        # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
        if len(solution_str) > _SOLUTION_CLIP_CHARS:
            solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

        if method == "strict":
            # this also tests the formatting of the model
            solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
            if len(solutions) == 0:
                final_answer = None
            else:
                # take the last solution
                final_answer = solutions[-1].replace(",", "").replace("$", "")
        elif method == "flexible":
            answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
            final_answer = None
            if len(answer) == 0:
                # no reward is there is no answer
                pass
            else:
                invalid_str = ["", "."]
                # find the last number that is not '.'
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
        return final_answer


    def compute_score(self, solution_str, ground_truth, method="flexible", format_score=0.0, score=1.0):
        """The scoring function for GSM8k.

        Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
        Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

        Args:
            solution_str: the solution text
            ground_truth: the ground truth
            method: the method to extract the solution, choices are 'strict' and 'flexible'
            format_score: the score for the format
            score: the score for the correct answer
        """
        answer = self.extract_solution(solution_str=solution_str, method=method)
        if answer is None:
            return 0
        else:
            if answer == ground_truth:
                return score
            else:
                return format_score



if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=256,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, 
        task_type=TaskType.CAUSAL_LM
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.cuda()
    print(student_model.print_trainable_parameters())

    teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    teacher_model.cuda()
    teacher_model.eval()


    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=10, 
                            do_train=True, 
                            per_device_train_batch_size=2,
                            gradient_accumulation_steps=16,
                            logging_steps=5,
                            logging_first_step=True,
                            logging_strategy='steps',
                            report_to='wandb',
                            save_strategy='epoch',
                            save_total_limit=10,
                            bf16=True,
                            learning_rate=0.0005,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            )
    data_collator = DefaultDataCollator()
    train_dataset = GSM8KSFTDataset(
        tokenizer=tokenizer, 
        max_seq_length=512,
        split="train",
        instruction='Let\'s think step by step and output the final answer after "####".'
    )

    test_dataset = GSM8KSFTDataset(
        tokenizer=tokenizer, 
        max_seq_length=512,
        split="test",
        instruction='Let\'s think step by step and output the final answer after "####".'
    )

    eval_callback = GSM8KEvalCallback(
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        eval_steps=50,   
        num_samples=100,
    )

    trainer = KDTrainer(student_model=student_model,
                        teacher_model=teacher_model, 
                        use_entropy=False,
                        args=args, 
                        train_dataset=train_dataset, 
                        tokenizer=tokenizer, 
                        data_collator=data_collator)

    trainer.add_callback(eval_callback)
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')



# CUDA_VISIBLE_DEVICES=4,5 python logit_distillation.py

# Qwen 2.5 family is using the same tokenizer, even vocab size is not the same due to 
# padding in lm_head and nn.embedding for better efficiency

# tok_small = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# tok_big   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

# print("small vocab_size:", tok_small.vocab_size)
# print("big   vocab_size:", tok_big.vocab_size)

# print("len(small tok):", len(tok_small))
# print("len(big tok):  ", len(tok_big))

# print("same tokenizer files:",
#       tok_small.get_vocab() == tok_big.get_vocab())
