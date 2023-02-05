# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import T5ForConditionalGeneration, T5Tokenizer

def download_model():
    model_name = "mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    download_model()