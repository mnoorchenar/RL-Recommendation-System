"""
RL Recommendation System — Flask application entry point.

On startup:
  1. Generates (or loads) synthetic MovieLens-style data
  2. Initialises a DQN agent with GRU state encoder
  3. Runs offline training in a background thread
  4. Serves the web UI + REST API on port 7860
"""
import os
import sys
import threading

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))

from src.data.dataset import MovieLensDataset
from src.models.dqn_agent import DQNAgent
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
dataset: MovieLensDataset | None = None
agent: DQNAgent | None = None
trainer: Trainer | None = None
evaluator: Evaluator | None = None

training_status: dict = {
    'running':       False,
    'epoch':         0,
    'total_epochs':  0,
    'loss':          None,
    'epsilon':       1.0,
    'progress':      0.0,
    'complete':      False,
    'metrics':       None,
    'error':         None,
}


# ---------------------------------------------------------------------------
# Background training
# ---------------------------------------------------------------------------

def _train_background(n_epochs: int) -> None:
    global training_status
    training_status.update(
        running=True, complete=False, error=None,
        total_epochs=n_epochs, epoch=0, progress=0.0,
    )

    def cb(info: dict):
        training_status.update(
            epoch=info['epoch'],
            progress=info['progress'],
            loss=round(info['loss'], 6),
            epsilon=round(info['epsilon'], 5),
        )

    try:
        trainer.train_offline(n_epochs=n_epochs, callback=cb)
        metrics = evaluator.evaluate(n_users=200)
        training_status['metrics'] = metrics
        training_status['complete'] = True
    except Exception as exc:
        training_status['error'] = str(exc)
        print(f"[Train] Error: {exc}")
    finally:
        training_status['running'] = False


# ---------------------------------------------------------------------------
# System initialisation
# ---------------------------------------------------------------------------

def initialize() -> None:
    global dataset, agent, trainer, evaluator

    print("[Init] Loading dataset…")
    dataset = MovieLensDataset(n_users=500, n_items=200)
    print(
        f"[Init] {dataset.n_users_actual} users, {dataset.n_items_actual} items | "
        f"train={len(dataset.train_df)}, test={len(dataset.test_df)}"
    )

    agent = DQNAgent(
        n_items=dataset.n_items_actual,
        embed_dim=16,
        hidden_dim=32,
        lr=1e-3,
        batch_size=32,
        buffer_capacity=30_000,
    )
    trainer   = Trainer(agent, dataset)
    evaluator = Evaluator(agent, dataset)

    t = threading.Thread(target=_train_background, args=(5,), daemon=True)
    t.start()
    print("[Init] Background training started.")


# ---------------------------------------------------------------------------
# UI route
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route('/api/status')
def api_status():
    ds = None
    if dataset is not None:
        ds = {
            'n_users': dataset.n_users_actual,
            'n_items': dataset.n_items_actual,
            'n_train': len(dataset.train_df),
            'n_test':  len(dataset.test_df),
        }
    return jsonify({'training': training_status, 'dataset': ds})


@app.route('/api/users')
def api_users():
    if dataset is None:
        return jsonify({'error': 'Initialising…'}), 503
    users = sorted(dataset.user_sequences.keys())[:100]
    return jsonify({'users': users})


@app.route('/api/recommend/<int:user_id>')
def api_recommend(user_id: int):
    if agent is None:
        return jsonify({'error': 'Initialising…'}), 503

    k = request.args.get('k', 10, type=int)
    history = dataset.get_user_history(user_id, max_len=20)

    recs = agent.get_top_k_recommendations(history, k=k)
    recommendations = [
        {**dataset.get_item_info(item_id), 'score': round(score, 4)}
        for item_id, score in recs
    ]

    history_display = [
        {**dataset.get_item_info(item_id), 'liked': bool(reward)}
        for item_id, reward in history[-15:]
    ]

    return jsonify({
        'user_id':         user_id,
        'history':         history_display,
        'recommendations': recommendations,
        'epsilon':         round(agent.epsilon, 4),
    })


@app.route('/api/train', methods=['POST'])
def api_train():
    if training_status['running']:
        return jsonify({'message': 'Already training'}), 409
    n_epochs = (request.json or {}).get('epochs', 5)
    t = threading.Thread(target=_train_background, args=(n_epochs,), daemon=True)
    t.start()
    return jsonify({'message': 'Training started', 'epochs': n_epochs})


@app.route('/api/training_history')
def api_training_history():
    losses = trainer.history['losses'] if trainer else []
    return jsonify({'losses': losses, 'epsilon': agent.epsilon if agent else 1.0})


@app.route('/api/metrics')
def api_metrics():
    if training_status['metrics']:
        return jsonify(training_status['metrics'])
    if agent and not training_status['running']:
        metrics = evaluator.evaluate(n_users=200)
        training_status['metrics'] = metrics
        return jsonify(metrics)
    return jsonify({'message': 'Training in progress — metrics will appear after completion.'})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=7860, debug=False)
