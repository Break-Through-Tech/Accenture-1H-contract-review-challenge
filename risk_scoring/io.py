"""Load predictions and source text without ground-truth labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / 'results/training/15ep/legal-bert-focal-optuna/test_predictions.parquet'
DEFAULT_CHUNKS = ROOT / 'data/processed/test_chunks_labeled.parquet'
DEFAULT_CATEGORIES = ROOT / 'data/processed/categories.json'
KEYS = ['title', 'chunk_id']
TEXT_COLUMNS = KEYS + ['chunk_text', 'char_start', 'char_end']


def load_categories(path=DEFAULT_CATEGORIES):
    categories = json.loads(Path(path).read_text())
    if (not isinstance(categories, list) or len(categories) != 41
            or any(not isinstance(c, str) or not c for c in categories)
            or len(set(categories)) != 41):
        raise ValueError('Expected 41 unique category names')
    return categories


def join_predictions(predictions, chunks, categories):
    """Require a complete one-to-one join; never copy annotation columns."""
    if (not predictions.columns.is_unique or
            set(predictions.columns) != set(KEYS + categories)):
        raise ValueError('Prediction columns must exactly match keys and categories')
    if not chunks.columns.is_unique or not set(TEXT_COLUMNS) <= set(chunks.columns):
        raise ValueError('Missing or duplicate text columns')
    chunks = chunks[TEXT_COLUMNS].copy()
    for frame in (predictions, chunks):
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError('Missing or duplicate chunk keys')
        if not frame.title.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError('Invalid contract title')
        if not pd.api.types.is_integer_dtype(frame.chunk_id) or (frame.chunk_id < 0).any():
            raise ValueError('chunk_id must be a nonnegative integer')
    for category in categories:
        values = predictions[category]
        if (not pd.api.types.is_numeric_dtype(values) or
                not np.isfinite(values.to_numpy()).all() or
                not values.between(0, 1).all()):
            raise ValueError(f'Invalid probabilities: {category}')
    for column in ('char_start', 'char_end'):
        if chunks[column].isna().any() or not pd.api.types.is_integer_dtype(chunks[column]):
            raise ValueError('Offsets must be integers')
    if not chunks.chunk_text.map(lambda x: isinstance(x, str)).all():
        raise ValueError('Missing or invalid chunk text')
    if ((chunks.char_start < 0).any() or (chunks.char_end <= chunks.char_start).any()
            or ((chunks.char_end - chunks.char_start) != chunks.chunk_text.str.len()).any()):
        raise ValueError('Invalid source offsets or text length')
    joined = predictions.merge(chunks, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Prediction/text key sets differ')
    return joined.drop(columns='_merge').sort_values(KEYS).reset_index(drop=True)


def load_prediction_chunks(predictions_path=DEFAULT_PREDICTIONS,
                           chunks_path=DEFAULT_CHUNKS, categories_path=DEFAULT_CATEGORIES):
    categories = load_categories(categories_path)
    # Read text columns only, excluding ground-truth labels.
    chunks = pd.read_parquet(chunks_path, columns=TEXT_COLUMNS)
    return join_predictions(pd.read_parquet(predictions_path), chunks, categories), categories


def load_validation_inputs(probabilities_path, train_chunks_path, titles_path, summary_path, metrics_path):
    """Reconstruct notebook order and cross-check saved category metrics."""
    from sklearn.metrics import average_precision_score

    categories = load_categories()
    summary = json.loads(Path(summary_path).read_text())
    if summary.get('mode') != 'full' or summary.get('config', {}).get('subset', 'missing') is not None:
        raise ValueError('Validation reconstruction requires full mode without a subset')
    if summary.get('run') != 'legal-bert-focal-optuna':
        raise ValueError('Unexpected model run')
    split = json.loads(Path(titles_path).read_text())
    titles = split['val_titles']
    if len(titles) != len(set(titles)) or len(titles) != split['n_val_contracts']:
        raise ValueError('Invalid validation title list')
    train = pd.read_parquet(train_chunks_path)
    if not set(titles) <= set(train.title):
        raise ValueError('Validation titles missing from training chunks')
    # Match build_split exactly: filtering preserves the original Parquet order.
    validation = train[train.title.isin(set(titles))].reset_index(drop=True)
    probabilities = np.load(probabilities_path, allow_pickle=False)
    if probabilities.shape != (len(validation), len(categories)):
        raise ValueError('Validation probability dimensions disagree')
    predictions = pd.DataFrame(probabilities, columns=categories)
    predictions.insert(0, 'chunk_id', validation.chunk_id)
    predictions.insert(0, 'title', validation.title)
    chunks = validation[TEXT_COLUMNS].copy()
    join_predictions(predictions, chunks, categories)
    metrics = pd.read_csv(metrics_path, index_col=0)
    if list(metrics.index) != categories:
        raise ValueError('Saved validation category order differs')
    labels = validation[categories].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid validation category labels for alignment audit')
    if not np.array_equal(labels.sum(axis=0), metrics.val_positives.to_numpy()):
        raise ValueError('Validation positive counts differ from saved run')
    average_precision = np.array([average_precision_score(labels[:, index], probabilities[:, index])
                                  for index in range(len(categories))])
    if not np.allclose(average_precision, metrics.val_ap.to_numpy(), rtol=0, atol=1e-10):
        raise ValueError('Saved validation AP does not match reconstructed row/category order')
    import hashlib
    audit = {
        'method': 'Notebook order reconstruction plus all-category saved AP and positive-count checks',
        'row_order': 'Original train Parquet order filtered by val_titles, reset_index(drop=True); no sorting or shuffle',
        'category_order': categories, 'rows': len(validation), 'contracts': validation.title.nunique(),
        'max_ap_difference': float(np.max(np.abs(average_precision - metrics.val_ap.to_numpy()))),
        'ordered_keys_sha256': hashlib.sha256(validation[KEYS].to_json(orient='records').encode()).hexdigest(),
        'limitation': 'Original array has no row IDs or source hashes; code trace and metric agreement corroborate reconstruction, not a cryptographic proof of historical identity.',
        'labels_usage': 'CUAD category labels used only to audit alignment; excluded from exported inputs and risk scoring.',
    }
    return predictions, chunks, audit
