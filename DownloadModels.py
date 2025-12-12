from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'microsoft/phi-2' # Example model
save_directory = './models_offline/microsoft-phi-2' # Your chosen local folder

# --- On a machine with internet ---
# 1. Download and save the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# 2. (Optional) Zip the folder for easy transfer
# !zip -r offline_bert.zip ./offline_bert

# --- On an offline machine ---
# 3. Load from the local directory
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

# Now you can use 'model' and 'tokenizer' without internet!
