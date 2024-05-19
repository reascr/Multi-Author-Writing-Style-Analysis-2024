How to train embeddings classifier

1. Load the embedding-classifier to your kaggle
2. Make sure As input for the file you use the lp-data (https://www.kaggle.com/datasets/ronjastern/lp-data/data)
3. Chose the right model configurations (emb_aug_a emb_aug_b emb_org_a emb_org_b)
4. Chose what test set you want to test on 
5. Let the notebook run using GPU (save version-> save and run all + advanced settings -> GPU as accelerator)
6. Hopefully the model runs for 12 hours and crashes than because of the time limit 
7. Download the model.pth file in your notebook output 
8. Create a new dataset and the model.pth file as input 
9. Use this created dataset as input for the "test-model" notebook. 
10. Report test results :)
11. Repeat for all three test sets (don't let it run on the full test set!)