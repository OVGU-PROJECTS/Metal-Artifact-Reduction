# Evaluation

Test-set inference and scoring 

## Test set
Run once for head and once for body (needs CUDA):

    python -m evaluation.run_inference --test_dir ../../test_images_1000/head \
        --ckpt results/pconv_fullstack/checkpoints/epoch_175.pt \
        --out_dir evaluation/predictions/pconv_fullstack_bestckpt/head
    python -m evaluation.score_images \
        --pred_dir evaluation/predictions/pconv_fullstack_bestckpt/head --fov head


## AAPM submission
In "aapm-score/", fo the 29 clinical cases:
- "prepare_clinical_inputs.py": make the inputs
- "create_case_valid_stats.py": their normalization
- "predscoring.py": the submission

