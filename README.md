# Transferability assesment for sugical phase recognition
This is the repository for the paper "Transferability assesment for sugical phase recognition"

Instructions:

1. Save all the embeddings in the given dataset folders at root such that: AutoLapro->Model->Frame->Embedding.npy, label.npy
2. To compute a score run `compute_metric.py -d $dataset -me $logme` I use shorthand for AutoLapro as `a` & RAMIE as `r`
3. This will save the metric results in  root as `$metric_score_$dataset.json`
4. Processing instructions for the score file can be found in `processing_results.ipynb`

   
