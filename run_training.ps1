Write-Host "=== Experiment 1: 1ch BatchNorm ===" -ForegroundColor Green
python train_acdc_1ch.py `
  --lm1-coord-weight 3.0 --lm1-heatmap-weight 2.0 `
  --sep-margin 0.15 --sep-weight 5.0

Write-Host "=== Experiment 2: 1ch GroupNorm ===" -ForegroundColor Green
python train_acdc_1ch.py `
  --group-norm `
  --lm1-coord-weight 3.0 --lm1-heatmap-weight 2.0 `
  --sep-margin 0.15 --sep-weight 5.0

Write-Host "=== Experiment 3: 2ch BatchNorm ===" -ForegroundColor Green
python train_acdc_2ch.py `
  --min-lm1-confidence 0.6 `
  --lm1-coord-weight 3.0 --lm1-heatmap-weight 2.0 `
  --sep-margin 0.15 --sep-weight 5.0 `
  --seg-dropout-prob 0.3

Write-Host "=== Experiment 4: 2ch GroupNorm ===" -ForegroundColor Green
python train_acdc_2ch.py `
  --group-norm --min-lm1-confidence 0.6 `
  --lm1-coord-weight 3.0 --lm1-heatmap-weight 2.0 `
  --sep-margin 0.15 --sep-weight 5.0 `
  --seg-dropout-prob 0.3

Write-Host "=== All Training Complete ===" -ForegroundColor Yellow