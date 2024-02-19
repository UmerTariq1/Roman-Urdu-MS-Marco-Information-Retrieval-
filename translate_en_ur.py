# To run on GPU number 2 on a cluster , use this command:
# export CUDA_VISIBLE_DEVICES=2; python translate_en_ur_2.py

 
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time, os, re, csv, sys, ftfy, nltk, argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn



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

    model = model.to(device)
    if device == "cuda":
        model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip, batch_size, max_seq_len, num_beams, device):
    translations = []
    for i in range(0, len(input_sentences), batch_size):
        batch = input_sentences[i : i + batch_size]

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
        ).to(device)

        # Generate translations using the model
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


def prepare_dataloader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

def load_objects(model_name, input_file, batch_size, num_workers, device):
    en_indic_ckpt_dir = model_name
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", device)

    ip = IndicProcessor(inference=True)

    translation_loader = prepare_dataloader(MSMarco(input_file), batch_size, num_workers)

    return en_indic_model, en_indic_tokenizer, ip, translation_loader


def main(model_name, input_file, output_dir, device, batch_size, num_workers, max_seq_len, num_beams):

    model, tokenizer, ip, dataset = load_objects(model_name, input_file, batch_size, num_workers, device)

    output_file = output_dir + 'translated_' + input_file.split('/')[-1]
    print("Output file:", output_file)

    # DataParallel for multiple GPUs
    if device == "cuda":
        model = nn.DataParallel(model)
    model = model.to(device)


    start = time.time()

    with open(output_file, 'a', encoding='utf-8', errors='ignore') as output:
        for batch in tqdm(dataset, desc="Translating..."):
            doc_ids   = batch[0]
            documents = batch[1]
            
            src_lang, tgt_lang = "eng_Latn", "urd_Arab"
            translated_documents = batch_translate(documents, src_lang, tgt_lang, model, tokenizer, ip, batch_size, max_seq_len, num_beams, device)
            

            for doc_id, translated_doc in zip(doc_ids, translated_documents):
                output.write(doc_id + '\t' + translated_doc + '\n')

    print(f"Total Time taken: {time.time() - start:.2f}s")


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Your description here.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers.')
    parser.add_argument('--num_beams', type=int, default=8, help='number of beams for beam search.')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length.')
    parser.add_argument('--model_name', type=str, default="ai4bharat/indictrans2-en-indic-1B", help='model name or path')
    parser.add_argument('--output_dir', type=str, default="/local/umerbutt/thesis/data/mmarco/output/", help='Output directory.')
    parser.add_argument('--input_file', type=str, default="/local/umerbutt/thesis/data/mmarco/collections/english_collection_part_ab.tsv", help='Input tsv file')

    args = parser.parse_args()


    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {DEVICE} for translation")
    print(f"Number of available GPUs: {num_gpus}")

    # Check the specific IDs of the GPUs
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

    print("Model name:", args.model_name)
    print("Input file:", args.input_file)
    print("Output dir:", args.output_dir)
    print("Batch size:", args.batch_size , " -- Num workers:", args.num_workers, " -- Num beams:", args.num_beams, " -- Max seq len:", args.max_seq_len)


    main(args.model_name, args.input_file, args.output_dir, DEVICE, args.batch_size, args.num_workers, args.max_seq_len, args.num_beams)
    print("Done!")
