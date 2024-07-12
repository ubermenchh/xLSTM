import torch 
import torch.nn as nn 
import torch.nn.functional as F

class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu"):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.device = device 

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size, device=device))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size, device=device))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, input, hx):
        h, c, n, m = hx 
        gates = input @ self.weight_ih.T + h @ self.weight_hh.T + self.bias 

        z_tilde, i_tilde, f_tilde, o_tilde = gates.chunk(4, 1)

        z = torch.tanh(z_tilde)
        i = torch.exp(i_tilde)
        f = torch.exp(f_tilde)
        o = torch.sigmoid(o_tilde)

        m_t = torch.maximum(torch.log(f) + m, torch.log(i))
        i_prime = torch.exp(torch.log(i) - m_t)
        f_prime = torch.exp(torch.log(f) + m - m_t)

        c = f_prime * c + i_prime * z 
        n = f_prime * n + i_prime 
        h_tilde = c / n 
        h = o * h_tilde 

        return h, c, n, m_t

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, device="cpu"):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.device = device 
        
        self.layers = nn.ModuleList([
            sLSTMCell(input_size if i == 0 else hidden_size, hidden_size, device)
            for i in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, input, hidden_state=None):
        bs, seq_len, _ = input.size()

        if hidden_state is None:
            hidden_state = [(
                torch.zeros(bs, self.hidden_size, device=self.device),
                torch.zeros(bs, self.hidden_size, device=self.device),
                torch.ones (bs, self.hidden_size, device=self.device),
                torch.zeros(bs, self.hidden_size, device=self.device)
            ) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c, n, m = hidden_state[layer_idx]
                h, c, n, m = layer(x, (h, c, n, m))
                hidden_state[layer_idx] = (h, c, n, m)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)

        return torch.stack(outputs, dim=1), hidden_state 

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, device="cpu"):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.device = device 

        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size, device=device))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size, device=device))
        self.bias = nn.Parameter(torch.randn(3 * hidden_size, device=device))

        self.w_q = nn.Linear(input_size, hidden_size, device=device)
        self.w_k = nn.Linear(input_size, hidden_size, device=device)
        self.w_v = nn.Linear(input_size, hidden_size, device=device)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)

        nn.init.zeros_(self.w_q.bias)
        nn.init.zeros_(self.w_k.bias)
        nn.init.zeros_(self.w_v.bias)

    def forward(self, input, hx):
        h, c = hx 
        gates = input @ self.weight_ih.T + h @ self.weight_hh.T + self.bias 

        i, f, o = gates.chunk(1, 3)

        i = torch.exp(i) 
        f = torch.exp(f)
        o = torch.sigmoid(o)

        q = self.w_q(input)
        k = self.w_k(input)
        v = self.w_v(input)

        c = f.unsqueeze(2) * c + i.unsqueeze(2) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        h = o * torch.bmm(q.unsqueeze(1), c).squeeze(1)

        return h, c 

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, device="cpu"):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.device = device 

        self.layers = nn.ModuleList([
            mLSTMCell(input_size if i == 0 else hidden_size, hidden_size, device=device)
            for i in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input, hidden_state=None):
        bs, seq_len, _ = input.size()
        
        if hidden_state is None:
            hidden_state = [(
                torch.zeros(bs, self.hidden_size, device=self.device),
                torch.zeros(bs, self.hidden_size, device=self.device)
            ) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            x = input[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c = hidden_state[layer_idx]
                h, c = layer(x, (h, c))
                hidden_state[layer_idx] = (h, c)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h 
            outputs.append(x)

        return torch.stack(outputs, dim=1), hidden_state

class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, lstm_type="slstm", device="cpu"):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.lstm_type = lstm_type 
        self.device = device 

        if self.lstm_type == "slstm":
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout, device)
        if self.lstm_type == "mlstm":
            self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout, device)

        self.norm = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, input, hidden_state=None):
        lstm_output, hidden_state = self.lstm(input, hidden_state)
        output = self.act(lstm_output)
        output = self.norm(output)
        output = self.proj(output)
        output = self.dropout_layer(output + input)
        return output, hidden_state

class xLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_blocks, dropout=0.0, lstm_type="slstm", device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size 
        self.embed_dim = embed_dim 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.num_blocks = num_blocks 
        self.lstm_type = lstm_type 
        self.device = device 

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            xLSTMBlock(embed_dim, hidden_size, num_layers, dropout, lstm_type, device)
            for _ in range(self.num_blocks)
        ])
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, input, hidden_state=None):
        embed_seq = self.embedding(input)
        if hidden_state is None:
            hidden_state = [None] * self.num_blocks 
        output_seq = embed_seq 
        for i, block in enumerate(self.blocks):
            output_seq, hidden_state[i] = block(output_seq, hidden_state[i])
        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_state


if __name__=="__main__":
    vocab_size = 1000
    embed_dim = 128 
    hidden_size = 64 
    num_layers = 2 
    num_blocks = 3 
    dropout = 0.1 
    lstm_type = "slstm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = xLSTM(vocab_size, embed_dim, hidden_size, num_layers, num_blocks, dropout, lstm_type, device=device)
    model.to(device)

    bs, seq_len = 4, 32 
    input_data = torch.randint(0, vocab_size, (bs, seq_len)).to(device)

    output, hidden_state = model(input_data)

    print(f"Input Shape: {input_data.shape}")
    print(f"Output Shape: {output.shape}")
