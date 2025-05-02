import torch
from transformers import AutoProcessor, AutoModelForCausalLM
model_name = "Hon-Wong/VoRA-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="cuda:0")
SYSTEM_PROMPT = """
You are an expert Vision-Language-Model. An input image will be provided.
Your goals are to:
1. Detect and list **every distinct visual element** in the image, including but not limited to:
   • photos or illustrations  
   • icons or symbols  
   • diagrams/flowcharts  
   • tables  
   • graphs or plots (bar, line, pie, scatter, etc.)
2. For each element, do ALL of the following:
   a. **Classify** the element’s type (photo, flowchart, table, line-graph, etc.).  
   b. **Describe** its visual content concisely and precisely.  
   c. **Transcribe** every piece of legible text that appears inside that element.  
   d. **Summarize key information**, such as variable names, axis labels, units, legends, and the main quantitative or qualitative takeaway.
3. If the image’s text is in a language other than English, **use that same language** when transcribing and describing; otherwise, default to English.
4. Output a well-formed **JSON array** called `"elements"`, where each item has the structure:

```json
{
  "id": "<unique-identifier>",
  "type": "<element-type>",
  "description": "<concise narrative>",
  "text": "<verbatim OCR text (if any)>",
  "key_points": "<bullet-style summary of the main idea or data>"
}
```
5.	After the JSON, provide a one-sentence overall summary of what the entire image communicates.
"""
conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT
            }
        ]
    },
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "/home/VoRA/image.png"
            },
            {
                "type":"text",
                "text":"<image>" 
            }
        ]
    }
]

with torch.inference_mode():
    model_inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True).to(device=model.device)
    gen_kwargs = {"max_new_tokens": 1024, "eos_token_id": processor.tokenizer.eos_token_id}

    outputs = model.generate(model_inputs, **gen_kwargs)
    output_text = processor.tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )
    print(output_text)
