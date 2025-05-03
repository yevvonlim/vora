import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from data_module.processor import VoRAProcessor
from models.modeling_vora import VoRAForCausalLM
import yaml 

config_path = "/workspace/VoRAParse/configs/pretrain_sionic_vl_parse.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


model_name = "/workspace/VoRAParse/output/pretrain_sionic_merged"
processor = VoRAProcessor(**config['data']['train']['data_preprocess'])
model = VoRAForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cuda:0")

test_input = {
  "id": "00000000",
  "image_folder": "",
  "frames": [
      "/workspace/VoRAParse/image.jpg",
  ],
  "conversations": [
      {
          "from": "human",
          "value": "<image>"
      },
      {
          "from": "gpt",
          "value": "This image is a ..."
      }
  ]
}

# --- before calling the model:
# flat_conversation = flatten_conversation(conversation)

with torch.inference_mode():
    model_inputs = processor.transform(test_input)
    del model_inputs['prompt'], model_inputs['id'], model_inputs['source'], model_inputs['gt'], model_inputs['question'], model_inputs['data_type']
    model_inputs = {k: v.to("cuda:0") if type(v)==torch.Tensor else v for k, v in model_inputs.items()}
    model_inputs['input_ids'] = model_inputs['input_ids'].unsqueeze(0)
    model_inputs['attention_mask'] = model_inputs['attention_mask'].unsqueeze(0)
    model_inputs['labels'] = model_inputs['labels'].unsqueeze(0)
    gen_kwargs = {"max_new_tokens": 1024, "eos_token_id": processor.tokenizer.eos_token_id}
    model_inputs['vision_placeholder_index'] = processor.vision_placeholder_index
    outputs = model.generate(model_inputs, **gen_kwargs)
    output_text = processor.tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )
    print(output_text)
