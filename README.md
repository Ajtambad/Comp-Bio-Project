# TCR - Epitope binding affinity prediction

## What's the point?

T-cell receptor (TCR) and epitope binding is a crucial aspect of the immune response, driving the body’s ability to recognize and combat pathogens and abnormal cells. TCRs bind to specific peptide fragments, or epitopes, presented on the surface of cells by major histocompatibility complex (MHC) molecules. This interaction determines whether a T cell activates to trigger immune responses, influencing processes like infection control, tumor surveillance, and autoimmunity. Understanding TCR-epitope binding is vital for advancing immunotherapy, vaccine design, and personalized medicine, as it helps identify targetable antigens and optimize therapeutic strategies for various diseases.

## Objective
Tha main goal of this project is to predict the binding affinity between TCRs and Epitopes in order to understand whether they are likely to bind or any alternate measures are to be taken.

## Methodology

1) Embeddings are generated using a pre-trained model that has learned the intricacies of protein sequences and their inherent meaning. I used the BERT-base-TCR model for this purpose and generated the necessary embeddings for both TCR and Epitope splits.
2) Trained a classifier model using the embeddings by splitting the data into training and validation sets, and evaluated it's performance with metrics like - accuracy, ROC, AUC, etc. I used two different approaches:
- A simple Neural Network.
- A Neural Network with skip connections and five fold cross validation splits, to ensure fair split of data into training and validation sets.

## Results

Obtained an accuracy of around **75%** and AUC of **80%** on average for both approaches. 

  
