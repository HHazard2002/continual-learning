import json
from pathlib import Path
from metrics import get_ll, compute_exact_match
from inference import generate_new_text

# Load JSON file
with open(Path("../configs/prompts.json")) as f:
    config = json.load(f)

label_mappings = config["label_mappings"]
instruction = config["instruction"]
prompt = config["prompt"]

def evaluate(model, tokenizer, test_ds, task, compute_ll=False):
      em = []
      losses = 0

      options_str = ", ".join(label_mappings[task].values())

      for i in range(100):
          sample = test_ds[i]
          numeric_answer = str(sample["label"])  # ensure string for mapping
          text_answer = label_mappings[task][numeric_answer]  # convert to text label

          text1 = sample["text1"]
          text2 = sample["text2"]

          if task == "amazon":
            text = (
              f"{tokenizer.bos_token}{prompt}{instruction[task]} {text1} "
              f"Paragraph: '{text2}'"
            )
          else:
            text = (
              f"{tokenizer.bos_token}{prompt}{instruction[task]} "
              f"Paragraph: '{text1} {text2}'"
            )

          # Generate model prediction
          new_text = generate_new_text(text, model, tokenizer)

          if compute_ll:
            losses += get_ll(text + text_answer, model)

          print(f"Predicted: {new_text}")
          print(f"Expected: {text_answer}")

          predicted = new_text.strip().lower()
          expected = text_answer.strip().lower()

          em.append((new_text, text_answer))

      print(f"The exact match for {task} is {compute_exact_match(em):.2f}")
      print(f"The average loss per sequence for {task} is {losses / 100:.4f}")