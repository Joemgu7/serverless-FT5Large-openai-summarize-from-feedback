from transformers import T5ForConditionalGeneration, T5Tokenizer

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model, tokenizer
    model_name = "mrm8488/flan-t5-large-finetuned-openai-summarize_from_feedback"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model, tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids,
            max_new_tokens=128,
            num_beams=8,
            do_sample=False,
            early_stopping=True,
            use_cache=True,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the results as a dictionary
    return result
