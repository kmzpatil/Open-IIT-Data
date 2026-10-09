
## 6. End-to-End Machine Learning Pipeline
To ensure the most optimal spatial predictions for unvisited addresses, we engineered a professional, object-oriented Machine Learning Pipeline (Solutions/geocoding_engine.py) that incorporates:

1. **AddressParser**: Uses NLP and Regex logic to extract latent geographical features such as pincode and house_no directly from the raw ddress_text.
2. **FraudDetector**: Hard-filters agent spoofing (repetitive photo_hash) and GPS teleportation (check-ins >1.5km from baseline).
3. **SpatialModel (Ensemble Regressors)**: We implemented and benchmarked several advanced ML algorithms to predict the coordinate displacement vector (delta_x, delta_y) of unvisited locations:
   - **HistGradientBoosting** (MAE: ~433m, RMSE: ~859m)
   - **RandomForestRegressor** (MAE: ~426m, RMSE: ~871m)
   - **ExtraTreesRegressor** (MAE: ~424m, RMSE: ~867m)
   - **Ensemble Average** (MAE: ~421m, RMSE: ~860m)

### Final Results & ML Conclusion
Despite engineering highly complex models, empirical validation against the surveyed_addresses.csv ground truth revealed that the **Conditional Baseline Fallback** strategy remains mathematically superior for Mean Absolute Error (MAE: **404.57m** vs ML Ensemble: **421m**). 

The ML models attempt to correct the baseline coordinates but inadvertently amplify noise for localized addresses. Thus, a true data-driven approach dictated that the pipeline defaults to the Fallback strategy, yielding the absolute closest predictions possible while maintaining the flexibility to deploy ML algorithms via configuration.
