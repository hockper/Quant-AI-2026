"""The dictionary of market words.

This is the heart of the whole idea. The encoder hands it a vector — a rich,
continuous description of what happened — and the codebook answers with the
single closest **word** from a fixed dictionary of (say) 512.

That is the bottleneck that forces the machine to be economical. It cannot
describe every day uniquely; it must decide that *these* days are the same kind
of day and *those* are a different kind. What survives that squeeze is structure.

The dictionary is not designed by us. It starts as random noise and is dragged,
word by word, toward whatever the data actually looks like.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    """A fixed set of words, learned from the data by moving averages.

    We do not train the words with gradients. Instead each word slowly drifts
    toward the average of everything that chose it — which is stabler, and is what
    keeps the dictionary from collapsing onto a handful of words.
    """

    def __init__(self, words: int, width: int, decay: float = 0.99,
                 commitment: float = 0.25, diversity: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.words = words
        self.decay = decay
        self.commitment = commitment
        self.diversity = diversity
        self.eps = eps

        dictionary = torch.randn(words, width)
        # Buffers, not Parameters: these are updated by averaging, not by gradients.
        self.register_buffer("dictionary", dictionary)
        self.register_buffer("usage", torch.zeros(words))
        self.register_buffer("running_sum", dictionary.clone())

    def forward(self, z: torch.Tensor) -> dict:
        """Snap each vector in `z` [B, width] to its nearest word.

        Returns the snapped vectors, which word each one chose, and how healthy the
        dictionary looks.
        """
        # A clone is REQUIRED. The moving-average update below writes to the
        # dictionary in place, and PyTorch needs the version it saw on the way in to
        # compute gradients. Without this, training corrupts silently.
        dictionary = self.dictionary.clone()

        # Squared distance from every vector to every word, without building the
        # full difference tensor: ||z||² - 2·z·w + ||w||²
        distance = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ dictionary.t()
            + dictionary.pow(2).sum(1)
        )
        chose = distance.argmin(dim=1)                       # [B] -- the word ids
        snapped = dictionary[chose]                          # [B, width]

        if self.training:
            self._drift_words_toward(z.detach(), chose)

        # The encoder must be pushed to produce vectors that are ALREADY close to a
        # word, rather than relying on the snap to clean up after it.
        commitment_loss = self.commitment * F.mse_loss(z, snapped.detach())

        diversity_loss = self.diversity * self._crowding(distance)

        # Snapping is not differentiable, so we pass the gradient straight through it:
        # forward gives the snapped vector, backward acts as if nothing happened.
        snapped = z + (snapped - z).detach()

        return {
            "snapped": snapped,
            "ids": chose,
            "commitment_loss": commitment_loss,
            "diversity_loss": diversity_loss,
            "perplexity": self._perplexity(chose),
        }

    def _crowding(self, distance: torch.Tensor) -> torch.Tensor:
        """Punish a dictionary that crowds onto a few words. (STORM, eq. 4.)

        Left alone, a VQ-VAE collapses: a handful of words win everything early and
        the rest are never chosen again. Reviving dead words fights that from one
        side; this fights it from the other.

        We take a SOFT vote — how strongly each vector leans toward each word — average
        those votes over the batch, and penalise the result for being lopsided. A
        dictionary where every word gets its fair share of the vote has high entropy;
        one that crowds onto a few words has low entropy. Minimising `p·log p` is
        maximising that entropy.

        This is not merely decorative. The soft vote depends on the DISTANCES between
        the encoder's output and the words, so it carries a real gradient back into
        the encoder — telling it to spread out. (An orthogonality penalty on the
        dictionary itself, which the paper also uses, would do nothing here: our
        dictionary is updated by moving averages, not gradients, so nothing could
        act on it.)
        """
        vote = F.softmax(-distance, dim=1)          # [batch, words]
        share = vote.mean(dim=0)                    # each word's share of the vote
        return (share * torch.log(share + 1e-10)).sum()

    def _drift_words_toward(self, z: torch.Tensor, chose: torch.Tensor) -> None:
        """Nudge each chosen word toward the average of what chose it."""
        hits = F.one_hot(chose, self.words).type(z.dtype)     # [B, words]
        counts = hits.sum(0)                                  # how many chose each word
        totals = hits.t() @ z                                 # sum of what chose each word

        self.usage.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.running_sum.mul_(self.decay).add_(totals, alpha=1 - self.decay)

        # Laplace smoothing: a word nobody chose keeps a tiny weight instead of
        # dividing by zero.
        total = self.usage.sum()
        smoothed = (self.usage + self.eps) / (total + self.words * self.eps) * total
        self.dictionary.copy_(self.running_sum / smoothed.unsqueeze(1))

    def _perplexity(self, chose: torch.Tensor) -> torch.Tensor:
        """How many words are effectively in use.

        1.0 means every day got the same word (the dictionary collapsed — the model
        learned nothing). Close to the dictionary size means it is using its full
        vocabulary. This is the single most useful number to watch while training.
        """
        counts = torch.bincount(chose, minlength=self.words).float()
        p = counts / counts.sum().clamp(min=1)
        return torch.exp(-(p * torch.log(p + 1e-10)).sum())

    @torch.no_grad()
    def revive_dead_words(self, z: torch.Tensor, threshold: float = 1e-3) -> int:
        """Restart words nobody is using, on top of real data.

        A VQ-VAE has a nasty failure mode: early on, a few words win everything and
        the rest are never chosen again — so most of the dictionary is wasted. When a
        word goes unused we drop it onto an actual encoder output, giving it a
        realistic place to start competing from.

        Returns how many words were revived.
        """
        dead = self.usage < threshold
        n = int(dead.sum())
        if n == 0 or z.shape[0] == 0:
            return 0
        picks = torch.randint(0, z.shape[0], (n,), device=z.device)
        self.dictionary[dead] = z[picks]
        self.running_sum[dead] = z[picks]
        self.usage[dead] = 1.0
        return n
