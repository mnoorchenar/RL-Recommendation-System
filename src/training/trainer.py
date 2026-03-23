"""
Offline RL trainer for the ad recommendation DQN.

Converts logged impression data into (s, a, r, s′) transitions,
fills the replay buffer, then runs batched TD-learning updates.
"""
import time
import numpy as np


class AdTrainer:
    def __init__(self, agent, dataset, max_seq_len: int = 10):
        self.agent       = agent
        self.dataset     = dataset
        self.max_seq_len = max_seq_len
        self.history     = {'losses': [], 'epochs': 0}

    # ------------------------------------------------------------------

    def _build_transitions(self) -> list:
        transitions = []
        for uid, seq in self.dataset.user_sequences.items():
            if len(seq) < 2:
                continue
            user_feat = self.dataset.get_user_features(uid)
            ctx_feat  = self.dataset.get_context_features()   # default: current time

            for t in range(len(seq) - 1):
                history      = seq[:t]
                ad_id, reward, clicked, converted = seq[t]
                next_history = seq[:t + 1]
                done         = t == len(seq) - 2

                state      = self.agent.get_state(history,      user_feat, ctx_feat)
                next_state = self.agent.get_state(next_history, user_feat, ctx_feat)

                transitions.append((state, int(ad_id), float(reward), next_state, done))

        return transitions

    # ------------------------------------------------------------------

    def train_offline(self, n_epochs: int = 5, callback=None) -> dict:
        t0 = time.time()
        n_users = len(self.dataset.user_sequences)
        print(f"[Trainer] Building transitions from {n_users} users…")

        transitions = self._build_transitions()
        print(f"[Trainer] {len(transitions):,} transitions → filling replay buffer")

        for s, a, r, ns, d in transitions:
            self.agent.store(s, a, r, ns, d)

        for epoch in range(n_epochs):
            steps       = min(600, len(self.agent.buffer) // max(1, self.agent.batch_size))
            epoch_losses = []

            for _ in range(steps):
                loss = self.agent.train_step()
                if loss is not None:
                    epoch_losses.append(loss)

            avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.history['losses'].append(round(avg, 6))
            self.history['epochs'] += 1

            print(
                f"[Trainer] Epoch {epoch + 1}/{n_epochs} | "
                f"loss={avg:.5f} | ε={self.agent.epsilon:.4f} | "
                f"buffer={len(self.agent.buffer):,}"
            )

            if callback:
                callback({
                    'epoch':        epoch + 1,
                    'total_epochs': n_epochs,
                    'loss':         avg,
                    'epsilon':      self.agent.epsilon,
                    'progress':     (epoch + 1) / n_epochs,
                })

        elapsed = time.time() - t0
        print(f"[Trainer] Done in {elapsed:.1f}s")
        return self.history
