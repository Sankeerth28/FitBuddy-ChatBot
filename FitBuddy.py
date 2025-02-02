
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
import pandas as pd

# Load the saved model and tokenizer
model_path = 'D:\\NLP Project\\bert_intent_model'  # Replace with the path to your saved model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Define a function to predict the intent
def predict_intent(input_text):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

# Load your intents mapping (assuming you have a mapping file)
# For example, if you saved it in a CSV:
# intent_mapping = pd.read_csv('path_to_intents_mapping.csv')  # Replace with your mapping file
intent_mapping = {
    0: "Body Building",
    1: "Meal Plan Recommendation",
    2: "Recommend Meditation or Yoga",
    3: "Suggest Recovery Exercises",
    4: "Weight Loss"
}
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(model_path, sequence, max_length):
    
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print("Fitbuddy: "+tokenizer.decode(final_outputs[0], skip_special_tokens=True))

model1_path = "D:\\NLP Project\\Fitbuddy"
max_len = 150
print("FitBuddy\nHow may I help you with your fitness?")
# Main loop for user input
while True:
    user_input = input("Enter your message (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Good Bye!")
        break
    
    # Get the predicted intent
    predicted_class = predict_intent(user_input)

    # Output the intent
    print(f"FitBuddy: Is the intent behind your input {intent_mapping[predicted_class]}? Please answer yes or no.")
    ip = input()
    while(ip.lower()!='yes' and ip.lower()!='no'):
        print("FitBuddy: Invalid input. Please answer in Yes or No")
        ip=input()
    if ip.lower()=="yes":
        seq = "[Q] "+user_input
        generate_text(model1_path, seq, max_len)
    else:
        print("FitBuddy: Please give your intent: ")
        intent = input()
        seq = "[Q] "+user_input+" My intent is "+ intent
        generate_text(model1_path, seq, max_len)

       
