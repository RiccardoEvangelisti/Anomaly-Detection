## Anomaly Detection on High-Performance Computing real data using Autoencoders
The study focuses on using **Autoencoder Neural Networks** for **Anomaly Detection** in **High-Performance Computing (HPC) data** from the **Marconi100 supercomputer**. The main objective is to develop a model capable of differentiating between normal and anomalous node states and to detect anomalies before manual checks by system administrators. The research highlights the effectiveness of an overcomplete autoencoder, which achieved high precision, recall, and F1 scores, demonstrating the model's accuracy in identifying anomalies. Notably, the model was able to detect anomalies up to **90 minutes** before the manual system administrators intervention.

#### Usage
- _Report_HPC_Anomaly_Detection.pdf_ describes the resarch in detail
- _Data_Exploration.ipynb_ contains the analysis and resampling of the original data
- The original data is downloadable [here](https://zenodo.org/records/7590583)
- _semi_supervised/_ folder contains the program
- _papers/_ folder contains articles on the topic and code examples
- _query_tool_ is a python module, necessary for manipulating the original data
