# Bigram Language Model

This project implements a simple Bigram Language Model using PyTorch, inspired by Andrej Karpathy's "Let's Make GPT" video. The model learns to predict the next token in a sequence based on the current token using embeddings and transformer blocks.

## Features
- Token and positional embeddings
- Transformer-based processing with multiple layers
- Layer normalization and output projection
- Cross-entropy loss computation for training
- Autoregressive text generation

## Model Architecture
The `BigramLanguageModel` consists of:
- **Token Embedding Table**: Maps vocabulary tokens to embeddings.
- **Position Embedding Table**: Encodes positional information.
- **Transformer Blocks**: Stack of transformer-based layers.
- **Final Layer Normalization**: Improves training stability.
- **Linear Output Layer**: Converts embeddings to logits for vocabulary prediction.

## Training Process
1. Initialize the model and move it to the selected device.
2. Create an AdamW optimizer with a specified learning rate.
3. Train for `max_iters` iterations by:
   - Evaluating loss at intervals.
   - Sampling batches of training data.
   - Computing loss and updating model parameters.
4. Generate text using the trained model.

## Code Example
### Model Initialization
```python
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
```

### Training Loop
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### Text Generation
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```

## Dependencies
- Python
- PyTorch

## Credits
This implementation is based on Andrej Karpathy's "Let's Make GPT" series.

## License
This project is available under the MIT License.

