import unittest
import torch
from transformers import LlamaConfig
from llama_model import LlamaForCausalLM

class TestLlamaModel(unittest.TestCase):

    def setUp(self):
        self.config = LlamaConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=1000
        )
        self.model = LlamaForCausalLM(self.config)
        self.input_ids = torch.randint(0, 1000, (1, 10))  # batch_size=1, seq_len=10

    def test_forward_pass(self):
        outputs = self.model(self.input_ids)
        self.assertEqual(outputs.shape, (1, 10, self.config.vocab_size))

    def test_loss_computation(self):
        labels = torch.randint(0, 1000, (1, 10))
        loss, logits = self.model(self.input_ids, labels=labels)
        self.assertIsNotNone(loss)
        self.assertEqual(logits.shape, (1, 10, self.config.vocab_size))

    def test_attention_heads(self):
        self.assertEqual(self.config.num_attention_heads, 4)

if __name__ == '__main__':
    unittest.main()
