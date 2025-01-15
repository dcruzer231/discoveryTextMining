# Text Mining from legal Discovery Documents
Project description. Law documents known as discovery requests are given to lawyers from law firms.  These documents contain a list of questions that the lawyer must answer and send back to the law firm.  A lawyer receives a discovery request for each client that they take on.  Many common questions are repeated across many documents.  A lawyer wanted to keep track of the most common questions asked by law firms.  These questions can be worded differently but ask a similar question.

In response to this challenge, I designed software that can keep track of the number of requests and cluster them to count similar questions.  This software was able to parse questions from a pdf file with 94% accuracy and use transformer ML models to extract the semantics from each question.  The questions were clustered using hierarchical clustering.

The output of the program is a csv file that listed each question with a cluster number assigned to it.
