from transformers import AutoTokenizer
import torch

def main():
    tok_checkpoint = 'mbart-large-50-many-to-one-mmt-tok'
    model_checkpoint = 'mt_infilling_weights_zh-en_2M_1660497817/epoch4.ckpt'

    torch.cuda.empty_cache()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
    model = torch.load(model_checkpoint).to(device)
    model.eval()

    inp_text = "阿拉伯人身上流淌着一种专制强加给他们的自卑和软弱，正是这种自卑和软弱造成了绝望、愤怒、暴力和偏狭。 <sep> - Arab men and women have shed the sense of humiliation and inferiority that despotism imposed on them – and that fostered desperation, <mask>."
    data = tokenizer(inp_text, return_tensors='pt')
    translated_tokens = model.generate(data['input_ids'].to(device))
    batch_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    print(batch_texts)

if __name__ == "__main__":
    main()
