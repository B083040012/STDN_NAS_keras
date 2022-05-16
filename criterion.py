import numpy as np

# eval pick and drop together by rmse and mape
def eval_together(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask) == 0:
        return -1
    mape = np.mean(np.abs(y[mask] - pred_y[mask]) / y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))

    return rmse, mape

def eval_rmse(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask) == 0:
        return -1
    rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))

    return rmse

# eval pick and drop seperately
def eval_lstm(y, pred_y, threshold):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    # pickup part
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]) / pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask])))
    # dropoff part
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(
            np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]) / dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask])))

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)