import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import time

# collections = 8841822 : 120 hours
# dev small = 1048565 : 14.27 hours
# dev = 524285 : 7.1 hours
# train = 50393 : 0.68 hours

BATCH_SIZE = 4  #100 - 37.01s = 0.37s per sentence
BATCH_SIZE = 8  #100 - 14.24s = 0.14s per sentence
BATCH_SIZE = 16 #100 - 8.73s = 0.08s per sentence
BATCH_SIZE = 32 #100 - 6.09s = 0.06s per sentence
BATCH_SIZE = 64 #100 - 4.90s = 0.049s per sentence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None
print(f"Using {DEVICE} for translation")

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
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
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
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
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic", quantization)


ip = IndicProcessor(inference=True)

en_sents = [
    "He has many old books, which he inherited from his ancestors.",
    "She is very hardworking and intelligent, which is why she got all the good marks.",
    "All the kids were having fun at the party and were eating lots of sweets.",
    "My friend has invited me to his birthday party, and I will give him a gift.",
]


print("len(en_sents):", len(en_sents))
src_lang, tgt_lang = "eng_Latn", "urd_Arab"
print("Processing start...")
start = time.time()
ur_translations = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)

print(f"Time taken: {time.time() - start:.2f}s")

print(f"\n{src_lang} - {tgt_lang}")
for input_sentence, translation in zip(en_sents, ur_translations):
    print(f"{src_lang}: {input_sentence}")
    print(f"{tgt_lang}: {translation}")

# flush the models to free the GPU memory
del en_indic_tokenizer, en_indic_model