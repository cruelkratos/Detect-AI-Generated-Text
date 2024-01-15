Using NLP Pipeline and Models like Naive Bayes with Stochastic Gradient Descent I was able to get 91.1% accuracy in predicting if the text was AI generated or human written


![image](https://github.com/cruelkratos/Detect-AI-Generated-Text/assets/116339436/7dfcdda5-97a6-46b1-9477-962149ee44e9)


<h1>Approach</h1>
<h2>Dataset</h2>
Upon initial inspection, it became evident that the supplied data exhibited a significant skew, rendering it unsuitable for effective model training. Recognizing this limitation, I opted to integrate an external dataset, specifically Large Language Models, to ensure a more balanced representation. The external dataset featured a balanced ratio of AI-generated content to human-written text. This strategic augmentation not only addressed the imbalance issue but also enriched the training data, fostering a more robust and comprehensive learning experience for the model.
<h2>Tokenizer</h2>
To process the words we need to convert them to tokens so that we can convert them to vectors to predict distances between the words and hence use that to check for LLM detection. 
I used the BytePair Encoding strategy which involves merging words with close frequencies together until we reach a desired vocab size( which i had set to 30,000 in my case) . After the tokenizer was trained i applied them to the train and the test datasets. 

<h2>TF-IDF Vectorization</h2>
Now we convert the tokens to vectors so now we can effectively calculate the distances between them for this I use the technique of TF-IDF Vectorization.


$$\text{TF(t,d)} =  \frac{\text{Number of times term t appears in document d}}{\text{Total number of terms in document d}}$$


$$\text{IDF(t,d)} =  \log{(1 + \frac{\text{Total number of documents in the collection D}}{\text{Number of documents containing term t}})}$$

<h2>Model Selection</h2>
In the process of model experimentation, I discovered that Naive Bayes consistently outperformed other models, yielding an impressive accuracy of 91.1\%. Stochastic Gradient Descent, while still commendable, achieved a slightly lower score of 88.6\%. To leverage the strengths of both models, I decided to implement a Voting Classifier. This ensemble approach involved combining the predictive abilities of the Gradient Descent Algorithm and the Naive Bayes Algorithm with varying weights. By fine-tuning these weights, I aimed to harness the complementary strengths of each model, ultimately enhancing the overall predictive performance of the system. This strategic combination not only showcased the versatility of ensemble methods but also allowed me to achieve a more robust and reliable model for the given task.

<h3>Naive Bayes</h3>
Naive Bayes’ is a classification algorithm(often dubbed as Idiot’s Bayes’) which is based on the Bayes’ Theorem. The algorithm makes the bold assumption that the probability of all posteriors are independent given the class priors. This obviously is a false assumption but it tends to work good enough and speeds up the pipeline (source: Andrew Ng Course CSN229).
<h3>Gradient Descent</h3>
As the name suggest we use the $\nabla$ operator here to find the direction with the maximum slope and move in the negative direction along it , until we reach the local optima. We Try to minimize the MSE(Mean Squared Error) function .
<h1>Learnings</h1>
in this project, I dived into the world of natural language processing (NLP) and machine learning. I learned how to process text through the NLP pipeline, breaking it down into smaller parts using tokenization. Decision trees became my go-to for understanding patterns in data, while text preprocessing taught me how to clean and prepare messy text for analysis. Vectorization, turning words into numbers, became a key step for machine learning. I got hands-on with various classification algorithms like Support Vector Machines, Naive Bayes, Gradient Descent, and Logistic Regression. Each algorithm had its strengths, and applying them in real-world scenarios showed me the practical side of these concepts. Overall, this project was a hands-on journey into the practical world of NLP and machine learning.
