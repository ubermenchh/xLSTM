#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
import torch
import torch.nn as nn
from tqdm import tqdm

from model import xLSTM

with open("./data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.1 * len(data))
train_data = data[n:]
valid_data = data[:n]

def get_batch(split):
    data = train_data if split == "train" else valid_data 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train(model, loss_fn, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0. 
        num_batches = len(train_data) // (block_size * batch_size)
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")

        for _ in progress_bar:
            batch_input, batch_target = get_batch("train")

            optimizer.zero_grad()
            output, _ = model(batch_input)
            loss = loss_fn(output.view(-1, vocab_size), batch_target.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches 
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            valid_loss = 0. 
            num_val_batches = len(valid_data) // (block_size * batch_size)
            for _ in range(num_val_batches):
                batch_input, batch_target = get_batch("valid")
                output, _ = model(batch_input)
                valid_loss += loss_fn(output.view(-1, vocab_size), batch_target.view(-1)).item()
            avg_valid_loss = valid_loss / num_val_batches 
            print(f"Validation Loss: {avg_valid_loss:.4f}")
        model.train()

    print("Training completed.")

def generate_text(model, start_text, length=200, temperature=1.0):
    model.eval()
    context = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    generated_text = start_text 

    with torch.no_grad():
        for _ in range(length):
            output, _ = model(context)
            probs = (output[0, -1] / temperature).softmax(dim=-1)
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            generated_text += itos[next_char_idx]
            context = torch.cat((context, torch.tensor([[next_char_idx]], device=device)), dim=1)
            if context.size(1) > block_size:
                context = context[:, -block_size:]
    
    return generate_text

if __name__=="__main__":
    block_size = 128 # sequence length
    batch_size = 64
    embed_dim = 128
    hidden_size = 256
    num_layers = 2 
    num_blocks = 3 
    dropout = 0.1 
    lstm_type = "slstm"
    learning_rate = 1e-3 
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = xLSTM(vocab_size, embed_dim, hidden_size, num_layers, num_blocks, dropout, lstm_type, device)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train(model, loss_fn, optimizer, num_epochs)

    sample_text = generate_text(model, start_text="The ", length=1024, temperature=0.7)
    print("Generated Text: \n\n")
    print(sample_text)
