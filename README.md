# Speaker Recognition Metrics

## Datasets

### Analysis

#### Output 'xvector.txt' format

```powershell
Get-Content "data/xvector.txt" | select -First 50 > "data/xvector-sample.txt" 
```

## Metrics

### Equal Error Rate (EER)

### minDCF (Minimum Detection Cost Function (DCF)

### Calibrated Log-Likelihood Ratio (Cllr)

## References

- <https://github.com/clovaai/voxceleb_trainer>
- <https://github.com/bsxfan/PYLLR>
