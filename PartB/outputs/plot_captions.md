# Plot Captions

## rf_actual_vs_predicted_scatter.png
Random Forest actual vs predicted scatter plot. Points closer to the 45-degree reference line indicate more accurate predictions.

## rf_actual_vs_predicted_line.png
Random Forest actual and predicted price line plot for the first 100 test samples. This helps compare trend similarity between true and predicted values.

## rf_residual_plot.png
Random Forest residual plot. Residuals should be distributed around the zero line; large patterns may indicate bias or underfitting.

## rf_residual_distribution.png
Random Forest residual distribution histogram. A distribution centered near zero suggests balanced prediction errors.

## rf_feature_importance.png
Top Random Forest feature importance chart. Larger importance scores indicate features that contribute more strongly to the model's decisions.

## xgb_actual_vs_predicted_scatter.png
XGBoost actual vs predicted scatter plot. Better predictions cluster around the 45-degree reference line.

## xgb_actual_vs_predicted_line.png
XGBoost actual and predicted price line plot for the first 100 test samples. This visualizes how closely the model follows real price movements.

## xgb_residual_plot.png
XGBoost residual plot. A random spread around zero indicates a healthier error pattern.

## xgb_residual_distribution.png
XGBoost residual distribution histogram. Narrower spread generally indicates more consistent predictions.

## xgb_feature_importance.png
Top XGBoost feature importance chart. It shows which input variables most influence predicted house price.

## correlation_heatmap.png
Correlation heatmap of the training features. Darker or more intense values show stronger linear relationships between variables.

## model_comparison_mae.png
Model comparison chart for MAE. Lower values indicate better average prediction accuracy.

## model_comparison_rmse.png
Model comparison chart for RMSE. Lower values indicate smaller large-error penalties.

## model_comparison_r2.png
Model comparison chart for R². Higher values indicate that the model explains more variance in house prices.

## model_comparison_mape.png
Model comparison chart for MAPE. Lower percentages indicate smaller average relative error.

