# To run on GPU number 2 on a cluster , use this command:
# export CUDA_VISIBLE_DEVICES=2; python translate_en_ur_2.py

 
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time, os, re, csv, sys, ftfy, nltk, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler


BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_BEAMS = 8
MAX_SEQ_LEN = 256 # for english_queries.train.tsv

MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/local/umerbutt/thesis/data/mmarco/garbage/"
# INPUT_FILE = "/local/umerbutt/thesis/data/mmarco/queries/train/english_queries.train.tsv"
# INPUT_FILE = "/local/umerbutt/thesis/data/mmarco/queries/dev/english_queries.dev.tsv"
INPUT_FILE = "/local/umerbutt/thesis/data/mmarco/queries/dev/english_queries.dev.small.tsv"
# INPUT_FILE = "/local/umerbutt/thesis/data/mmarco/collections/english_collection.tsv"

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
    

def initialize_model_and_tokenizer(ckpt_dir, direction,device):
    
    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


    if device == "cuda":
        model.half()    # converts the model's parameters and buffers to half-precision floating-point format 

    model.eval()

    model = nn.DataParallel(model, [0,1])
    # model = model.to(device)

    return tokenizer, model

def batch_translate(batch, src_lang, tgt_lang, model, tokenizer, ip, batch_size, max_seq_len, num_beams, device):
    translations = []
    # no need to iterate over batches anymore, it's already done for you
    # make sure to access items in the batch properly, as it may differ depending on the structure of your dataset
    batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch,
        src=True,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        generated_tokens = model.module.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=max_seq_len, 
            num_beams=num_beams,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

    # Postprocess the translations, including entity replacement
    translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    del inputs
    torch.cuda.empty_cache()

    return translations

def prepare_dataloader(dataset, batch_size, num_workers, gpu_id):
    sampler = DistributedSampler(dataset, num_replicas=num_gpus, rank=gpu_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler
    )

def load_objects(model_name, input_file, batch_size, num_workers, device, gpu_id):
    en_indic_ckpt_dir = model_name
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", device)

    ip = IndicProcessor(inference=True)

    translation_loader = prepare_dataloader(MSMarco(input_file), batch_size, num_workers, gpu_id)

    return en_indic_model, en_indic_tokenizer, ip, translation_loader

def main(gpu_id, model_name, input_file, output_dir, device, batch_size, num_workers, max_seq_len, num_beams):

    model, tokenizer, ip, data_loader = load_objects(model_name, input_file, batch_size, num_workers, device, gpu_id)

    output_file = output_dir + 'translated_' + input_file.split('/')[-1]
    print("Output file:", output_file)


    start = time.time()

    with open(output_file, 'a', encoding='utf-8', errors='ignore') as output:
        for batch in tqdm(data_loader, desc="Translating..."):
            batch_doc_ids   = batch[0]
            batch_documents = batch[1]
            
            src_lang, tgt_lang = "eng_Latn", "urd_Arab"
            translated_documents = batch_translate(batch_documents, src_lang, tgt_lang, model, tokenizer, ip, batch_size, max_seq_len, num_beams, device)
            

            for doc_id, translated_doc in zip(batch_doc_ids, translated_documents):
                output.write(doc_id + '\t' + translated_doc + '\n')

    print(f"Total Time taken: {time.time() - start:.2f}s")


for gpu_id in range(num_gpus):
    main(gpu_id, MODEL_NAME, INPUT_FILE, OUTPUT_DIR, DEVICE, BATCH_SIZE, NUM_WORKERS, MAX_SEQ_LEN, NUM_BEAMS)

print("Done!")
