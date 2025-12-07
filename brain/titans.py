"""
Titans Architecture: Neural Memory with Test-Time Training
Implements a security agent that learns and adapts to input patterns in real-time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import hashlib


class NeuralMemory(nn.Module):
    """
    MLP-based long-term memory vector.
    Architecture: Linear -> ReLU -> Linear
    """
    
    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32, vocab_size: int = 100):
        super(NeuralMemory, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # MLP layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural memory.
        
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        # Embed input tokens
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Pass through MLP
        hidden = self.fc1(embedded)
        hidden = self.relu(hidden)
        logits = self.fc2(hidden)  # [batch_size, seq_len, vocab_size]
        
        return logits


class SecurityAgent:
    """
    Titans-based security agent that maintains session-specific neural memory
    and performs test-time training to adapt to new inputs.
    """
    
    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32, vocab_size: int = 100, learning_rate: float = 0.001):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        # Initialize neural memory
        self.model = NeuralMemory(embed_dim, hidden_dim, vocab_size)
        
        # Optimizer for test-time training
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function for surprise calculation
        self.criterion = nn.CrossEntropyLoss()
        
    def _tokenize(self, text: str) -> torch.Tensor:
        """
        Simple character-level tokenization with modulo mapping to vocab.
        
        Args:
            text: Input text string
            
        Returns:
            Tensor of token indices
        """
        # Convert characters to integers and map to vocab size
        tokens = [ord(c) % self.vocab_size for c in text]
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    
    def calculate_surprise(self, text: str) -> float:
        """
        Calculate surprise score (gradient loss) for the input text.
        Higher surprise indicates anomalous input.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Surprise score (float)
        """
        if not text or len(text) < 2:
            return 0.0
        
        # Tokenize input
        tokens = self._tokenize(text)
        
        # Prepare input and target for next-token prediction
        input_tokens = tokens[:, :-1]  # All but last token
        target_tokens = tokens[:, 1:]   # All but first token
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tokens)  # [1, seq_len-1, vocab_size]
        
            # Reshape for loss calculation
            logits_flat = logits.reshape(-1, self.vocab_size)
            target_flat = target_tokens.reshape(-1)
            
            # Calculate cross-entropy loss as surprise metric
            loss = self.criterion(logits_flat, target_flat)
            surprise_score = loss.item()
        
        return surprise_score
    
    def update_memory(self, text: str) -> None:
        """
        Perform test-time training to update neural memory based on new input.
        This allows the model to adapt to expanding contexts.
        
        Args:
            text: Input text to learn from
        """
        if not text or len(text) < 2:
            return
        
        # Tokenize input
        tokens = self._tokenize(text)
        
        # Prepare input and target
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        
        # Training mode
        self.model.train()
        
        # Forward pass
        logits = self.model(input_tokens)
        
        # Reshape for loss calculation
        logits_flat = logits.reshape(-1, self.vocab_size)
        target_flat = target_tokens.reshape(-1)
        
        # Calculate loss
        loss = self.criterion(logits_flat, target_flat)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def is_anomalous(self, surprise_score: float, threshold: float = 5.0) -> bool:
        """
        Determine if a surprise score indicates an anomaly.
        
        Args:
            surprise_score: Calculated surprise score
            threshold: Anomaly threshold (default: 5.0)
            
        Returns:
            True if anomalous, False otherwise
        """
        return surprise_score > threshold


class SessionManager:
    """
    Manages session-specific SecurityAgent instances.
    """
    
    def __init__(self):
        self.sessions: Dict[str, SecurityAgent] = {}
    
    def get_or_create_agent(self, session_id: str) -> SecurityAgent:
        """
        Retrieve existing agent for session or create a new one.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            SecurityAgent instance for the session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = SecurityAgent()
        return self.sessions[session_id]
    
    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.sessions)
