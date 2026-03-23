"""
RL Ad Recommendation System — Flask application.

Startup sequence
----------------
1. Generate (or load) synthetic ad / user / impression data
2. Initialise DQN agent with AdStateEncoder
3. Launch offline training in a background thread
4. Serve the web UI + REST API on port 7860
"""
import os
import sys
import threading

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))

from src.data.ad_dataset import AdDataset
from src.models.dqn_agent import AdDQNAgent
from src.training.evaluator import AdEvaluator
from src.training.trainer import AdTrainer

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
dataset:   AdDataset    | None = None
agent:     AdDQNAgent   | None = None
trainer:   AdTrainer    | None = None
evaluator: AdEvaluator  | None = None

training_status: dict = {
    'running':      False,
    'epoch':        0,
    'total_epochs': 0,
    'loss':         None,
    'epsilon':      1.0,
    'progress':     0.0,
    'complete':     False,
    'metrics':      None,
    'error':        None,
}


# ---------------------------------------------------------------------------
# Background training
# ---------------------------------------------------------------------------

def _train_background(n_epochs: int) -> None:
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
# Init
# ---------------------------------------------------------------------------

def initialize() -> None:
    global dataset, agent, trainer, evaluator

    print("[Init] Loading ad dataset…")
    dataset = AdDataset(n_users=300, n_ads=100)
    print(
        f"[Init] {dataset.n_users_actual} users | {dataset.n_ads_actual} ads | "
        f"{len(dataset.impressions_df):,} impressions"
    )

    agent = AdDQNAgent(
        n_ads=dataset.n_ads_actual,
        user_feat_dim=21,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=30_000,
    )
    trainer   = AdTrainer(agent, dataset)
    evaluator = AdEvaluator(agent, dataset)

    t = threading.Thread(target=_train_background, args=(5,), daemon=True)
    t.start()
    print("[Init] Background training started.")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# API — status & metadata
# ---------------------------------------------------------------------------

@app.route('/api/status')
def api_status():
    ds = None
    if dataset is not None:
        a   = dataset.analytics
        ds  = {
            'n_users':      dataset.n_users_actual,
            'n_ads':        dataset.n_ads_actual,
            'n_train':      len(dataset.train_df),
            'n_test':       len(dataset.test_df),
            'impressions':  a['total_impressions'],
            'ctr':          a['ctr'],
            'cvr':          a['cvr'],
            'ecpm':         a['ecpm'],
        }
    return jsonify({'training': training_status, 'dataset': ds})


@app.route('/api/users')
def api_users():
    if dataset is None:
        return jsonify({'error': 'Initialising…'}), 503
    users = sorted(dataset.user_sequences.keys())[:100]
    return jsonify({'users': users})


# ---------------------------------------------------------------------------
# API — recommendations
# ---------------------------------------------------------------------------

@app.route('/api/recommend/<int:user_id>')
def api_recommend(user_id: int):
    if agent is None:
        return jsonify({'error': 'Initialising…'}), 503

    k       = request.args.get('k', 10, type=int)
    hour    = request.args.get('hour', None, type=int)
    dow     = request.args.get('dow',  None, type=int)

    history   = dataset.get_user_history(user_id, max_len=20)
    user_feat = dataset.get_user_features(user_id)
    ctx_feat  = dataset.get_context_features(hour, dow)
    profile   = dataset.get_user_profile(user_id)

    recs = agent.get_top_k_recommendations(history, user_feat, ctx_feat, k=k)

    enriched_recs = []
    for r in recs:
        info = dataset.get_ad_info(r['ad_id'])
        enriched_recs.append({
            **info,
            'q_score':   r['q_score'],
            'p_click':   r['p_click'],
            'p_convert': r['p_convert'],
        })

    history_display = []
    for ad_id, reward, clicked, converted in history[-15:]:
        info = dataset.get_ad_info(ad_id)
        history_display.append({**info, 'clicked': bool(clicked), 'converted': bool(converted)})

    return jsonify({
        'user_id':         user_id,
        'profile':         profile,
        'history':         history_display,
        'recommendations': enriched_recs,
        'epsilon':         round(agent.epsilon, 4),
    })


# ---------------------------------------------------------------------------
# API — analytics
# ---------------------------------------------------------------------------

@app.route('/api/analytics')
def api_analytics():
    if dataset is None:
        return jsonify({'error': 'Initialising…'}), 503
    return jsonify(dataset.analytics)


@app.route('/api/advertisers')
def api_advertisers():
    if dataset is None:
        return jsonify({'error': 'Initialising…'}), 503
    return jsonify({'advertisers': dataset.analytics.get('by_advertiser', [])})


# ---------------------------------------------------------------------------
# API — training
# ---------------------------------------------------------------------------

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
