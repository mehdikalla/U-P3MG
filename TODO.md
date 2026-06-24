# TODO list for the project
Each time you complete a task, please check it off the list below, putting an "x" in the brackets. This will help us keep track of our progress and ensure that we cover all necessary aspects of the project.

## Refactoring and Documentation should be done by mid May — ✅ Completed. 
**Fully completed by Mehdi.**

- [x] Complete the overview of the project in section A "README.md".
- [x] Rename the repository to "U-P3MG".
- [x] Follow the instructions in section B.2 to install Mamba in the README.md file.
- [x] Create an `env.yml` file with the necessary dependencies for the project.
- [x] Create a `config.yaml` file with the necessary configuration for training and evaluation.
- [x] Provide a script for training the model that uses the configuration file and handles GPU if available.
- [x] Provide a script for evaluating the model that uses the configuration file and handles GPU if available.
- [x] Provide a script to run traditional algorithms for comparison with the U-P3MG model.
- [x] Complete the instructions on how to train the model in section D of the README.md file.
- [x] Provide instructions on how to visualize the results and analyze the performance of the model in section E of the README.md file.
- [x] Provide a code to simulate signals
- [x] By creating the scripts for training, evaluation, and running traditional algorithms, remove the slurm scripts since they will no longer be needed.

## Future Work

### **1. Implementing and Evaluating Models** should be done by the end of June.
- [x] Fix one loss function and one metric for evaluation and use them consistently throughout the project.
- [x] Fix the learning rate and the optimizer for all the models you will implement.
- [x] Implement the P3MG traditional algorithm and evaluate its performance on the easiest dataset.
- [x] Implement the HQ traditional algorithm and evaluate its performance on the easiest dataset.
- [x] Implement the ISTA traditional algorithm and evaluate its performance on the easiest dataset.
- [x] Implement the PMMS traditional algorithm and evaluate its performance on the easiest dataset.
- [x] Implement the U-HQ model and evaluate its performance on the easiest dataset.
- [x] Implement the U-ISTA model and evaluate its performance on the easiest dataset.
- [x] Implement the U-P3MG model and evaluate its performance on all datasets.

Once we have completed the above tasks, we can consider fully deep learning approaches and compare them with the U-P3MG model.

The following are some of the deep learning approaches token from the U-HQ paper that we can implement and compare with the U-P3MG model:
- [ ] Implement FCU-Net and evaluate its performance on the easiest dataset.
- [ ] Implement ResUNet and evaluate its performance on the easiest dataset.

### **2. Evaluating Models on Different Datasets** should be done by mid July.
Once the evaluation is done on the easiest dataset, we can move on to the more challenging datasets and evaluate the performance of all models on those datasets as well.

- [ ] Dataset 2: two gaussians and by diversification of the noise level.
- [ ] Dataset 3: three or more gaussians and by diversification of the noise level.

We should have three levels of noise for each dataset: low, medium, and high. This will allow us to evaluate the robustness of the models under different noise conditions.

### **3. Ablation Study** should be done by the end of August.
- [ ] Ablation study: Fix the intern number of iterations and evaluate the performance of the U-P3MG model with different numbers of extern iterations.
- [ ] Ablation study: Fix the number of extern iterations and evaluate the performance of the U-P3MG model with different numbers of intern iterations.
- [ ] Ablation study: Evaluate the performance of the U-P3MG model with different configurations of the model, such as different numbers of layers for the MLP.

Each time you complet an evaluation for the ablation study, please make sure to keep the results and metrics in a well organized manner so that we can easily analyze the results and draw conclusions from the ablation study.

### **4. Writing the Paper** should be done by mid September.
- [ ] Start writing the paper by mid July, and aim to submit it to a conference by the end of September. We can start by writing the introduction and related work sections, and then move on to the methodology and experiments sections.
