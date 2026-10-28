# Leakage Checklist

- [x] Chronological splits on `TransactionDT`; boundaries stored as absolute dt.
- [x] Preprocessor `fit` receives **only** the training fold; `transform` applies
      training-fold statistics to val/test.
- [x] Preprocessor statistics proven invariant to val/test content
      (`test_preprocessor_statistics_invariant_to_val_test`).
- [x] Rolling aggregates use strictly prior rows (`<` target `TransactionDT`);
      ties do not see each other (`test_rolling_aggregates_are_strictly_prior`,
      `test_rolling_aggregates_exclude_current_row`).
- [x] Deleting future rows does not change earlier features
      (`test_no_future_influence_on_features`).
- [x]