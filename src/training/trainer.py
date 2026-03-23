"""
Offline RL trainer.

Converts logged user-item interactions into (s, a, r, s') transitions,
fills the DQN replay buffer, then runs mini-batch gradient updates.
"""
import time


class Trainer:
    def __init__(self, agent, dataset, max_seq_len: int = 10):
        self.agent = agent
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.history = {'losses': [], 'epochs': 0}

    # ------------------------------------------------------------------

    def _build_transitions(self) -> list:
        transitions = []
        for user_id, seq in self.dataset.user_sequences.items():
            if len(seq) < 2:
                continue
            for t in range(len(seq) - 1):
                history    = seq[:t]
                action, reward = seq[t]
                next_history   = seq[:t + 1]
                done = t == len(seq) - 2

                state      = self.agent.get_state(history,      self.max_seq_len)
                next_state = self.agent.get_state(next_history, self.max_seq_len)
                transitions.append((state, action, reward, next_state, done))
        return transitions

    # ------------------------------------------------------------------

    def train_offline(self, n_epochs: int = 5, callback=None) -> dict:
        t0 = time.time()
        print(f"[Trainer] Building transitions from {len(self.dataset.user_sequences)} users…")
        transitions = self._build_transitions()
        print(f"[Trainer] {len(transitions)} transitions → filling replay buffer")

        for s, a, r, ns, d in transitions:
            self.agent.store(s, a, r, ns, d)

        for epoch in range(n_epochs):
            steps = min(500, len(self.agent.buffer) // max(1, self.agent.batch_size))
            epoch_losses = []
            for _ in range(steps):
                loss = self.agent.train_step()
                if loss is not None:
                    epoch_losses.append(loss)

            avg = sum(epoch_losses) / max(1, len(epoch_losses))
            self.history['losses'].append(round(avg, 6))
            self.history['epochs'] += 1

            print(
                f"[Trainer] Epoch {epoch + 1}/{n_epochs} | "
                f"loss={avg:.5f} | ε={self.agent.epsilon:.4f}"
            )

            if callback:
                callback({
                    'epoch':        epoch + 1,
                    'total_epochs': n_epochs,
                    'loss':         avg,
                    'epsilon':      self.agent.epsilon,
                    'progress':     (epoch + 1) / n_epochs,
                })

        print(f"[Trainer] Done in {time.time() - t0:.1f}s")
        return self.history
