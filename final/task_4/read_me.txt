4.	Topic Discovery and Custom Feature Scoring [25 Marks]: 2 or 9/8 Athina,Alex
The purpose of this section is to explore whether the language used in social-media posts can reveal meaningful topical structure and whether a custom-designed scoring mechanism can improve the representation of topic-related information.
•	Identify and describe the most important terms associated with each selected topic. You may use an appropriate approach such as term frequency, TF-IDF, n-grams, topic modelling, or another justified method. Your analysis should explain why the selected terms are informative for distinguishing the chosen topics. 
[8 Marks]
•	Design and implement a custom word or phrase scoring mechanism that assigns a numerical importance score to terms within your corpus. The scoring mechanism may be based on frequency, topic-specific frequency, TF-IDF, class association, positional information, or a justified combination of such signals. You must clearly explain the scoring logic, including any thresholds, weights, or rules used. 
Examples of possible scoring approaches include:
•	assigning a score to each term based on its frequency within a specific topic relative to its frequency across the full corpus;
•	deriving a score from TF-IDF values;
•	combining topic-specific term frequency with sentiment association;
•	assigning scores to bi-grams rather than individual words (unigrams); 
•	calculating a document-level score as the sum or average of the scores assigned to the terms it contains.
[9 Marks]
•	Apply the scoring mechanism to create one or more additional features that can be used in the topic-classification stage. The score must be calculated using training data only and then applied to the test data, in order to avoid information leakage. 
[4 Marks]
•	Compare the custom scoring representation with a standard baseline representation, such as Bag-of-Words or TF-IDF. Discuss whether the additional scoring features improve, reduce, or do not materially affect topic-classification performance. Your discussion should consider both predictive performance and interpretability. 
