# Ringtone Generation using Variational Autoencoder (VAE)

# --- Imports and Setup ---
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from constants import SEED, TIME_STEP, MAX_TIMESTEPS, LATENT_DIM, MIDI_FOLDER, EPOCHS, PITCH_RANGE, BATCH_SIZE
from utils import pad_embeddings, process_midi_dataset
from vae import VAE
from dataset import MIDIDataset

np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Loss ---
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Training ---
def train(model, dataloader, optimizer, epochs=EPOCHS):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.view(batch.size(0), -1)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = loss_function(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return loss_history

# --- Generation ---
def generate(model, n_samples=5):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM)
        samples = model.decoder(z).view(n_samples, MAX_TIMESTEPS, PITCH_RANGE[1]-PITCH_RANGE[0]).numpy()
    return samples

def embedding_to_midi(embedding, out_file):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for t, timestep in enumerate(embedding):
        for p, val in enumerate(timestep):
            if val > 0.3:
                note = pretty_midi.Note(velocity=int(val*127), pitch=p + PITCH_RANGE[0], start=t*TIME_STEP, end=(t+1)*TIME_STEP)
                instrument.notes.append(note)
    midi.instruments.append(instrument)
    midi.write(out_file)

if __name__ == '__main__':
    # --- Execution Pipeline ---
    embeddings = process_midi_dataset(MIDI_FOLDER)
    padded = pad_embeddings(embeddings)
    dataset = MIDIDataset(padded)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = VAE(input_dim=MAX_TIMESTEPS * (PITCH_RANGE[1] - PITCH_RANGE[0]), latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    loss_history = train(model, dataloader, optimizer)
    
    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    samples = generate(model, n_samples=3)
    
    for i, emb in enumerate(samples):
        embedding_to_midi(emb, f"generated_ringtone_{i}.mid")
    
    # --- Visualize a generated piano roll ---
    plt.figure(figsize=(12, 4))
    plt.imshow(samples[0].T, aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Velocity')
    plt.title('Generated Ringtone (Piano Roll)')
    plt.xlabel('Time Step')
    plt.ylabel('Pitch')
    plt.show()
