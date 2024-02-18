# To run on GPU number 2 on a cluster , use this command:
# export CUDA_VISIBLE_DEVICES=2; python en_ur_translate.py

 
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time, os, re, csv, sys, ftfy, nltk, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_BEAMS = 8
MAX_SEQ_LEN = 256 # for english_queries.train.tsv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/local/umerbutt/thesis/data/mmarco/output/"
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
    

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization=None):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

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
        ).to(DEVICE)

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


en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"  # ai4bharat/indictrans2-en-indic-dist-200M
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic")

ip = IndicProcessor(inference=True)

output_file = OUTPUT_DIR + 'translated_' + INPUT_FILE.split('/')[-1]
print("Output file:", output_file)


train_ds = MSMarco(INPUT_FILE)
translation_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS)

start = time.time()
# DataParallel for multiple GPUs
if DEVICE == "cuda":
    en_indic_model = nn.DataParallel(en_indic_model)
en_indic_model = en_indic_model.to(DEVICE)


with open(output_file, 'a', encoding='utf-8', errors='ignore') as output:
    for batch in tqdm(translation_loader, desc="Translating..."):
        doc_ids   = batch[0]
        documents = batch[1]
        
        src_lang, tgt_lang = "eng_Latn", "urd_Arab"
        translated_documents = batch_translate(documents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

        for doc_id, translated_doc in zip(doc_ids, translated_documents):
            output.write(doc_id + '\t' + translated_doc + '\n')

print(f"Total Time taken: {time.time() - start:.2f}s")
print("Done!")
