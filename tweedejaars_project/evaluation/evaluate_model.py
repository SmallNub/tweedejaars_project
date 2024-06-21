import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from tweedejaars_project.evaluation.metrics import compute_income_score_added, compute_income_score_naive
from tweedejaars_project.evaluation import adjust_pred_realtime, adjust_pred_consistency, adjust_pred_conform


def evaluate_income(
    train_func,
    test_func,
    train_in: pd.DataFrame,
    train_out: pd.Series,
    test_in: pd.DataFrame,
    test_df: pd.DataFrame,
    version="target",
    repeat=100
):
    """Evaluates a model by repeatedly running a simplified version of the income score."""
    def step(train_in, train_out, test_in, realtime, action, price, naive):
        # Train
        model = train_func(train_in, train_out)
        # Test and adjust predictions
        pred_adj, pred = test_func(model, test_in)
        pred_adj = adjust_pred_realtime(realtime, pred_adj)

        # Make consistent first (then conform (not relevant for income))
        # pred_con = adjust_pred_consistency(pred)

        # Make conform first then consistent
        # pred = adjust_pred_conform(pred)
        # pred_con2 = adjust_pred_consistency(pred)

        # Compute scores
        preds = [pred_adj]
        scores = [compute_income_score_added(action, price, p.to_numpy(copy=True), naive) for p in preds]
        return model, scores, pred

    # Predefined
    realtime = test_df[f"{version}_two_sided_ptu_realtime"].to_numpy(copy=True)
    action = test_df["naive_strategy_action"].to_numpy(copy=True)
    price = test_df["settlement_price_realized"].to_numpy(copy=True)
    naive = compute_income_score_naive(action, price)

    results = Parallel(n_jobs=-1)(delayed(step)(train_in, train_out, test_in, realtime, action, price, naive) for _ in range(repeat))

    # Number of outputs
    n_out = len(results[0][1])

    # Outputs (model, best score, score x repeat, best base prediction) x n_out
    out = [[None, -np.inf, [], None] for _ in range(n_out)]

    # Iterate every result
    for model, scores, preds in results:
        # Iterate every score
        for i, score in enumerate(scores):
            out[i][2].append(score)  # Save the score for every output
            # Save the best score and the corresponding model for the corresponding output type
            if score > out[i][1]:
                out[i][0] = model
                out[i][1] = score
                out[i][3] = preds

    # Get the overall best
    best = [None, -np.inf, None]
    for i in range(n_out):
        if out[i][1] > best[1]:
            best[:2] = out[i][:2]
            best[2] = out[i][3]

    for i in range(n_out):
        out[i][2] = pd.Series(out[i][2], name=str(i))
        print(out[i][2].describe())
    return best, out
