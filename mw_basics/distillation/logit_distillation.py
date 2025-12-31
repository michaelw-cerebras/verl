from torch.utils.data import Dataset
from datasets import load_dataset
import torch

def compute_forward_kl(
        student_logits,
        teacher_logits,
        target,
        padding_id,
        reduction,
        temperature=1.0,
):  
    # logits: (batch_size, seq_len, vocab_size)
    # logits at timestep t is actually the predicted token distribution for token[t+1]
    # so the last element of logits is the next token of <eos>, which is meaningless
    student_logits = student_logits[:, :-1, :] / temperature
    teacher_logits = teacher_logits[:, :-1, :] / temperature

    # Compute 3 variables needed for forward KL calculation
    student_log_probs = torch.log_softmax()






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
        # NOTE: Add eos_token to let me model learn when to end generation
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

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
            labels = labels + [-100] * (self.max_seq_length - len(input_ids))
            attention_mask = attention_mask + [0] * (self.max_seq_length - len(input_ids))
        
        '''
        We should not shift if we later use AutoModelForCausalLM / HF Trainer
        because shifting is automatically done there
        '''
        # input_ids = input_ids[:-1]
        # labels = labels[1:]
        return {
            'input_ids': torch.tensor(input_ids), 
            'attention_mask':torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
