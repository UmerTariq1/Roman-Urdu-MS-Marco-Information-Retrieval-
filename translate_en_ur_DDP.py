# To run on GPU number 2 on a cluster , use this command:
# export CUDA_VISIBLE_DEVICES=2; python translate_en_ur_2.py

## D
 
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time, os, re, csv, sys, ftfy, nltk, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group 
import torch.multiprocessing as mp


BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_BEAMS = 8
MAX_SEQ_LEN = 256 # for english_queries.train.tsv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/local/umerbutt/thesis/data/mmarco/garbage/"
# INPUT_FILE = "/local/umerbutt/thesis/data/mmarco/queries/train/english_queries.train.tsv"
# INPUT_FILE = "/local/umerbutt/thesis/data/mmarco/queries/dev/english_queries.dev.tsv"
INPUT_FILE = "/local/umerbutt/thesis/data/mmarco/queries/dev/english_queries.dev.small.tsv"


# Get the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Using {DEVICE} for translation")
print(f"Number of available GPUs: {num_gpus}")

# Check the specific IDs of the GPUs
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")


print("Input file:", INPUT_FILE)
print("Output dir:", OUTPUT_DIR)
print("Batch size:", BATCH_SIZE , " -- Num workers:", NUM_WORKERS, " -- Num beams:", NUM_BEAMS, " -- Max seq len:", MAX_SEQ_LEN)


def ddp_setup(rank, world_size):
    '''
    Args:
        - rank (int): The rank of the current process.
        - world_size (int): The total number of processes.
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    


class MSMarco(Dataset):
    '''
    Pytorch's dataset abstraction for MSMarco.
    '''

    def __init__(self, file_path, target_language="urd"):
        self.documents = self.load_msmarco(file_path)
        
    def __len__(self):
        return len(self.documents)

    def load_msmarco(self, file_path:str):
        '''
        Returns a list with tuples of [(doc_id, doc)].
        It uses ftfy to decode special carachters.
        Also, the special translation token ''>>target_language<<' is
        added to sentences.

        Args:
            - file_path (str): The path to the MSMarco collection file.
        '''
        documents = []
        with open(file_path, 'r', errors='ignore') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for line in tqdm(csv_reader, desc="Reading .tsv file"):
              doc_id = line[0]
              doc_lines = nltk.tokenize.sent_tokenize(ftfy.ftfy(line[1]))
              for doc in doc_lines:
                if len(doc) > 1:
                    # documents.append((doc_id, r'>>{target_language}<< ' + doc))
                    documents.append((doc_id, doc))
        
        return documents

    def __getitem__(self,idx):
        doc_id, doc = self.documents[idx]
        return doc_id, doc
    

def initialize_model_and_tokenizer(ckpt_dir, direction, gpu_id:int):
    '''
    Args:
        - ckpt_dir (str): The path to the model checkpoint.
        - direction (str): The translation direction. It can be "en-indic" or "indic-en".
        - gpu_id (int) / rank : The ID of the GPU to use.
    '''

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # model = model.to(gpu_id)
    if gpu_id == "cuda":
        model.half()

    model.eval()

    model = DDP(model, device_ids=[gpu_id])
    tokenizer = model.to(gpu_id)

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.module.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=MAX_SEQ_LEN, 
                num_beams=NUM_BEAMS,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

def prepare_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )

def load_objects(model_name, input_file, rank):
    en_indic_ckpt_dir = model_name
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic",rank)

    ip = IndicProcessor(inference=True)

    translation_loader = prepare_dataloader(MSMarco(input_file))

    return en_indic_model, en_indic_tokenizer, ip, translation_loader


def main(rank:int, world_size:int, output_file):
    
    ddp_setup(rank, world_size)
    torch.cuda.set_device(rank)
    model, tokenizer, ip, translation_loader = load_objects("ai4bharat/indictrans2-en-indic-1B", INPUT_FILE, rank)

    start = time.time()

    with open(output_file, 'a', encoding='utf-8', errors='ignore') as output:
        for batch in tqdm(translation_loader, desc="Translating..."):
            doc_ids   = batch[0]
            documents = batch[1]
            
            src_lang, tgt_lang = "eng_Latn", "urd_Arab"
            translated_documents = batch_translate(documents, src_lang, tgt_lang, model, tokenizer, ip)

            for doc_id, translated_doc in zip(doc_ids, translated_documents):
                output.write(doc_id + '\t' + translated_doc + '\n')

    print(f"Total Time taken: {time.time() - start:.2f}s")
    print("Done!")

    destroy_process_group()

if __name__ == "__main__":
    '''
    The main function.
    doesnt take arguments. just works with the global variables.
    '''
    world_size = num_gpus

    output_file = OUTPUT_DIR + 'translated_' + INPUT_FILE.split('/')[-1]
    print("Output file:", output_file)   
    
    mp.spawn(main, args=(world_size, output_file), nprocs=world_size)

