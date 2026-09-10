import numpy as np
def regression_metrics(y_true,y_pred):
    e=y_pred-y_true
    mae=float(np.mean(np.abs(e))); mse=float(np.mean(e**2))
    mean_temp_error=float(np.mean(e))
    max_temp_error=float(np.max(np.abs(e)))
    true_r=np.diff(y_true,axis=1); pred_r=np.diff(y_pred,axis=1)
    max_rise_rate_error=float(np.max(np.abs(np.max(pred_r,axis=1)-np.max(true_r,axis=1))))
    peak_error=float(np.mean(np.abs(np.max(y_pred,axis=1)-np.max(y_true,axis=1))))
    return {"MAE":mae,"MSE":mse,"MeanTempError":mean_temp_error,
            "MaxTempError":max_temp_error,"MaxRiseRateError":max_rise_rate_error,
            "PeakError":peak_error}
