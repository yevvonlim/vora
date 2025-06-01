from datasets import load_dataset
from pathlib import Path
import json

# Root directory where JSONL and frames will be saved
ROOT = Path("/home/ye/VoRAParse/dataset/TableImageTextPair")
FRAMES_DIR = ROOT / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def extract_last_assistant_response(messages) -> str:
    """
    Walks the list of messages (each a dict with 'role' and 'content') in reverse,
    finds the last one where role == 'assistant', and returns its text.
    Handles cases where content is already a dict, a list of blocks, or a JSON string.
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content")

        # If content is a JSON string, parse it
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return content.strip()

        # If content is a dict with a 'text' field, return it
        if isinstance(content, dict) and "text" in content:
            return content["text"].strip()

        # If content is a list of blocks, find the first text block
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text" and "text" in block:
                    return block["text"].strip()

    return ""

def format_batch(batch) -> dict:
    """
    Process a batch of examples:
      - Save each PIL image under frames/{id8}.jpg
      - Extract the last assistant response for each conversation
      - Return lists for 'id', 'frames', and 'conversations' columns
    """
    ids_out = []
    frames_out = []
    convs_out = []

    for raw_id, img, messages in zip(batch["id"], batch["image"], batch["conversations"]):
        # zero-pad to 8 digits
        id_str = str(raw_id).zfill(8)
        img_rel = f"frames/{id_str}.jpg"

        # save the image
        img.save(FRAMES_DIR / f"{id_str}.jpg", format="JPEG")

        # extract last assistant response
        gpt_text = extract_last_assistant_response(messages)

        # build conversation list
        conv = [
            {"from": "human", "value": "<image>"},
            {"from": "gpt",   "value": gpt_text},
        ]

        ids_out.append(id_str)
        frames_out.append([img_rel])
        convs_out.append(conv)

    return {
        "id": ids_out,
        "frames": frames_out,
        "conversations": convs_out,
    }

def preprocess():
    # 1) Load the dataset
    ds = load_dataset("sionic-ai/TableImageTextpairData-replica", split="train", num_proc=128)

    # 2) Map in batched mode
    ds2 = ds.map(
        format_batch,
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        num_proc=48,  # or more if you prefer parallel image writes
    )

    # 3) Save to JSONL
    out_file = ROOT / "annotations/data.json"
    ds2.to_json(str(out_file), orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(ds2)} records to {out_file}")

if __name__ == "__main__":
    preprocess()