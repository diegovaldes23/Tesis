from transformers import AutoModel, AutoTokenizer

PATH = "/home/diego/Escritorio/Pytesis/models/bert-base-uncased"

print("Cargando tokenizer local...")
tok = AutoTokenizer.from_pretrained(PATH, local_files_only=True)

print("Cargando modelo local...")
model = AutoModel.from_pretrained(PATH, local_files_only=True)

print("Todo OK, se cargó BERT local.")
