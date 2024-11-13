# Summary
This folder contains three notebook(.ipynb) files:-
## 1. Fed-D_PSI
This folder contains the notebook file for the implementation of the following paper **Federated Dropout – A Simple Approach for Enabling Federated Learning on Resource Constrained Devices** (*https://arxiv.org/pdf/2109.15258*)

<!---### A. Problem Statement
Federated Learning involves training models across multiple devices while keeping data localized to enhance privacy. However, two main bottlenecks hinder its effectiveness:

Communication Bottleneck: High-dimensional model updates require significant bandwidth.

Computation Bottleneck: Resource-constrained devices struggle with the intensive computations needed for model updates.
### B. FedDrop Concept
FedDrop aims to mitigate these bottlenecks by utilizing a dropout mechanism to create subnets from a global model, allowing devices to train smaller models instead of the full-scale DNN. This approach includes several key features:

Subnet Distribution: The server generates multiple subnets using dropout techniques, each assigned to different devices. This allows devices to work with smaller models, thereby reducing computational demands.

Subnet Aggregation: After local updates, the server aggregates these subnets to update the global model, ensuring that all updates contribute to improving overall performance.

C2 Awareness: The dropout rates for each device are adapted based on their communication and computation capabilities, optimizing performance while managing latency.

### C. Implementation Steps
The implementation of FedDrop follows these sequential steps in each training round:

Subnet Generation: The server randomly generates subnets using dropout, assigning one subnet per device.

Model Downloading: Devices download their respective subnets from the server.

Local Model Updating: Each device updates its local subnet using its local dataset.

Local Model Uploading: Updated subnets are sent back to the server.

Global Model Updating: The server aggregates the updated subnets to refine the global model.
--->
This paper presents the Federated Dropout (FedDrop) method to address both computation and communication challenges in Federated Learning (FL), especially on resource-constrained devices.
The method is based on dropout, which randomly deactivates neurons in a neural network to reduce model size. FedDrop adapts dropout rates per device, creating submodels of different sizes that match device capabilities.
In each round, the server generates several subnets with distinct dropout rates. Each subnet is assigned to a device, which trains and uploads the updated submodel.
The server aggregates the results from all devices, updating the global model. This adaptive dropout approach helps reduce resource demands while maintaining model accuracy, addressing the overfitting issues typical in uniform dropout​
## 2. FedAvg_PSI
This folder contains the notebook file for the implementation of the following paper **Communication-Efficient Learning of Deep Networks from Decentralized Data** (*https://arxiv.org/pdf/1602.05629*)

This foundational work in federated learning introduces FedAvg, a method to train a global model across decentralized data by allowing devices to compute updates locally.
FedAvg combines local stochastic gradient descent on client devices with periodic aggregation at a central server. The clients perform several rounds of local updates and send only the model updates (not raw data) to the server.
By averaging these local updates, the server updates the global model. FedAvg is effective in handling non-IID data distributions across devices, reducing communication requirements significantly compared to synchronous gradient descent, which demands more frequent updates​
<!---Methodology Overview
1. Federated Learning Framework
The authors define Federated Learning as a decentralized approach where:

Data remains on devices: Each participating device (client) has its local dataset, which is never uploaded to a central server.

Model updates are shared: Instead of sending raw data, clients compute updates to the global model based on their local data and send only these updates to the server.

2. FederatedAveraging Algorithm
The core of the methodology is the FederatedAveraging algorithm, which consists of the following steps:

Client Selection: In each communication round, a random subset of clients is selected to participate in the training
process. This helps manage communication costs and ensures efficiency.

Model Downloading: The server sends the current global model parameters to the selected clients.

Local Training: Each client performs several iterations of local SGD on its dataset to compute an update to the model parameters. This local training is crucial as it allows clients to leverage their unique data distributions.

Model Uploading: After local training, each client sends its model update back to the server.

Model Aggregation: The server aggregates all received updates by averaging them to form an updated global model.

Iteration: This process repeats for multiple communication rounds until convergence or until a predefined stopping criterion is met.

3. Handling Non-IID and Unbalanced Data
The authors address two significant challenges in FL:

Non-IID Data: The data across clients may not be identically distributed, meaning each client's dataset may represent different user behaviors and preferences.

Unbalanced Data: Some clients may have significantly more data than others, leading to potential biases in model updates.

To mitigate these issues, FederatedAveraging effectively utilizes local updates that reflect individual client distributions while maintaining robustness through model averaging.

4. Communication Efficiency
The methodology emphasizes reducing communication rounds, which are a primary constraint in FL due to limited bandwidth on mobile devices. The authors demonstrate that by increasing local computation (e.g., more SGD iterations per client before sending updates), they can significantly decrease the number of required communication rounds—reporting reductions by factors of 10 to 100 compared to traditional synchronized SGD approaches.
--->
## 3. HeteroFL_PSI
This folder contains the notebook file for the implementation of the following paper **HETEROFL: COMPUTATION AND COMMUNICATION EFFICIENT FEDERATED LEARNING FOR HETEROGENEOUS CLIENTS** (*https://arxiv.org/pdf/2010.01264*)

HeteroFL is proposed as a framework that allows each client device to train a model that varies in complexity based on its own computational and communication capabilities, while still contributing to a unified global model.
Unlike FedAvg, which requires all clients to use the same model architecture, HeteroFL supports model heterogeneity by creating multiple "shrunken" versions of the global model (with fewer parameters). Devices with lower capacity receive and train simpler models.
The global model is aggregated by combining these heterogeneous models at each communication round. A "masking trick" and static batch normalization (sBN) are introduced to manage model variance and ensure stable updates, making HeteroFL adaptable to non-IID data with varying client resources​
<!---1. Problem Definition
The authors identify the limitations of traditional FL methods, which typically assume that all local models share the same architecture as the global model. This assumption restricts the complexity of the global model to accommodate the least capable client, leading to inefficiencies in both computation and communication.
2. HeteroFL Framework
The proposed HeteroFL framework allows for heterogeneous local models, meaning that clients can have different model architectures and complexities while still contributing to a unified global model. The key components of this framework include:
A. Model Heterogeneity
Adaptive Subnetwork Allocation: Clients are assigned subnetworks based on their computational capabilities. This allows each client to train a model that suits its resources while contributing to a common global model.
Model Complexity Levels: The framework defines multiple levels of computation complexity, enabling clients to operate at different levels based on their capabilities. This flexibility helps optimize performance across diverse devices.
B. Global Model Aggregation
Weighted Model Averaging: The server aggregates model parameters from clients based on their assigned complexity levels. This process ensures that smaller models can still benefit from updates made by larger models while maintaining stability in global aggregation.
Dynamic Parameter Selection: The method allows for the selection of subsets of global model parameters for local training, which helps in reducing communication overhead and enhancing efficiency.
3. Training Process
The training process in HeteroFL involves several key steps:
Client Selection: A subset of clients is selected randomly for each communication round.
Model Distribution: The server distributes the current global model parameters to selected clients.
Local Training: Each client trains its local model using its private data for several epochs, adjusting its parameters based on local computations.
Parameter Uploading: After local training, clients send their updated parameters back to the server.
Global Aggregation: The server aggregates these updates into a new global model using weighted averaging based on the complexity levels of each client.
4. Static Batch Normalization (sBN)
To address privacy concerns associated with Batch Normalization (BN), which requires sharing running statistics, the authors introduce Static Batch Normalization (sBN). In this approach:
Clients do not track running statistics during training but normalize their batch data locally.
After training, the server queries clients to update global BN statistics cumulatively, ensuring privacy while stabilizing optimization.
5. Scaler Module
To manage discrepancies in scale between different local models:
A Scaler module is introduced that adjusts outputs from local models during training. This module ensures that representations are appropriately scaled before being passed through normalization and activation layers.
This adjustment allows for seamless integration of various local models into a single global inference model without compromising performance.--->

