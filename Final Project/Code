<font color = black face = "bold" size = 6>Final Project</font>


<font color = black face = "bold" size = 5>ENPM 808W</font>


<font color = black face = "bold" size = 5>By : Abhiram Dapke and Neeraj Sathe</font>


<font color = black face = "bold" size = 4>Background</font>  

Diabetes, is a group of metabolic disorders in which there are high blood sugar levels over a prolonged period. Symptoms of high blood sugar include frequent urination, increased thirst, and increased hunger. If left untreated, diabetes can cause many complications. Acute complications can include diabetic ketoacidosis, hyperosmolar hyperglycemic state, or death. Serious long-term complications include cardiovascular disease, stroke, chronic kidney disease, foot ulcers, and damage to the eyes.
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

<font color = black face = "bold" size = 4>Objective</font>  

The goal of this project was to leverage techniques and methods in R and create a regression model to accurately predict whether or not the patients in the dataset have diabetes or not.
we will use the following approaches:
1. Lasso Regression
2. Logistic Regression using glm
3. Random Forest Method
4. Decision Tree approach


<font color = black face = "bold" size = 4>Write-up</font>  

This write-up will explain the code/techniques that were used, but the following items address what was explicitly requested in the project rubric:

-   Why this is a good idea: This was a good idea because the dataset contained features that obviously could impact the intensity of a patient being diabetic. We were able to identify these attributes and sucessfully predict whether the patient had diabetes or not and how the factors affected her.  The dataset also provided an opportunity to explore and identify which pieces were important to either include or ignore. We included the data of patients taken in between the years 1960-1990. We were able to clean the data and extract only the most useful predictors to perform our analyses.    

-  What you did: We first identified the factors which were significantly affecting the patient and had a direct impact on the patient's bloodpressure, Insulin and Glucose level which are key factors in determining diabetes. Initially, We plotted graphs representing the the no. of pregnancies, age, glucose, bloodpressure of patients with/without diabetes. Then we plotted a box plot depicting the Diabetes Pedigree Function with respect to the age and divided the age into 5 age groups. After that, we plotted histograms, box plot and a line plot representing the data which is essential for a proper data visualization. Next, we applied 4 techniques on the data, namely, Lasso regression, logistic regression, random forest method and decision tree approach to compare the accuracy of each of them with one another. These approaches have their graphs plotted as well. Finally, we compared them and found out that decision tree approach gives the best result with respect to accuracy.  

-  Who did what: Abhiram did the regression analysis for lasso regression and random forest method while Neeraj explored the concepts of decision tree and logistic regression. We both contributed in the initial analysis of the histograms and box plots. Neeraj generated the variable correlation matrix and created the power point presentation while Abhiram wrote the write-up and commments related to the code.  

-  Whether your technique worked or not: Our techniques worked and actually were better predictors than what was originally expected. Though we thought and wrote in the proposal that we would only implement the logistic regression and decision tree techniques, we ended up implementing Random forest approach as well as Lasso regression and compared all of them. It was truly engaging and very fascinating to compare 4 methods and find out the best one.   


The rest of this script shows the code, plots generated, as well some supplementary text for the sections of the code. This .Rmd file has all the comments and descriptions in the rubric and can be considered as a write-up.  


<font color = black face = "bold" size = 4>About the data</font>

Pregnancies: Number pregnancies  
Glucose: Blood Glucose concentration  
Blood Pressure: Diastolic blood pressure (mm Hg)  
Skin Thickness: Triceps skin fold thickness (mm)  
Insulin: Insulin Level (mu U/ml)  
BMI: Body mass index  
Diabetes Pedigree Function: Diabetes pedigree function  
Age: In years  
Outcome: Yes or No  


```{r}


#Importing the libraries
library("dplyr")
library("rpart")
library("grid")
library("lattice")
library("tidyverse")
library("Amelia")
library("cwhmisc")
library('corrgram')
library('corrplot')
library("InformationValue")
library("ggplot2")
library("cowplot")
library("e1071")
library("caret")
library("purrr")
library("glmnet")
library("tidyr")
library("tree")
library("randomForest")
library("reshape2")
theme_set(theme_cowplot())


diabetes = read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Project\\diabetes.csv")
summary(diabetes)
```


The summary shows the mean, quartile etc values of the variables if they are numeric. Here, we can see that the output is either 0 or 1, where 0 means that patient does not have diabetes and 1 means patient have diabetes. Also, we can see that few columns contain the value 0 which is not possible. For example, Blood pressure can never be 0 for a given data, hence we need to clean the data.



```{r}

diabetes$Outcome <- factor(diabetes$Outcome)

# removing those observation rows with 0 in any of the variables
for (i in 2:6) {
      diabetes <- diabetes[-which(diabetes[, i] == 0), ]
}

# modify the data column names slightly for easier typing
names(diabetes)[7] <- "dpf"
names(diabetes) <- tolower(names(diabetes))

str(diabetes)
print(paste0("number of observations = ", dim(diabetes)[1]))
print(paste0("number of predictors = ", dim(diabetes)[2]))



ggplot(diabetes,aes(pregnancies, fill = outcome)) + geom_histogram(position = "dodge", bins = 8) + labs(title = "Number of pregnancies with and without diabetes")

ggplot(diabetes,aes(glucose, fill = outcome)) + geom_histogram(position = "dodge", bins = 8) + labs(title = "Number of persons having glucose with/without diabetes")

ggplot(diabetes,aes(age, fill = outcome)) + geom_histogram(position = "dodge", bins = 8) + labs(title = "Age of people with/without diabetes")

ggplot(diabetes,aes(bloodpressure, fill = outcome)) + geom_histogram(position = "dodge", bins = 8) + labs(title = "Blood Pressure of patients with/without diabetes")

ggplot(diabetes,aes(cut(age,breaks=5),y=dpf,fill=cut(age,breaks=5)))+geom_boxplot()+scale_fill_brewer(palette="RdBu")
```


The number of observations remaining is 392, with 8 columns of variables and 1 column of response. We're going to look at some simple plots to better understand the data. It is easy to do since we only have 8 variables to work with. Plots do not need to look nice at this stage since we're looking to understand the data.

```{r}
par(mfrow = c(2, 2))

# the $ notation can be used to subset the variable you're interested in.
hist(diabetes$pregnancies)
hist(diabetes$age)
hist(diabetes$glucose)
hist(diabetes$bmi)
```

The graphs show some of the distributions of the variables. Age and number of times pregnant are not normal distributions as expected since the underlying population should not be normally distributed either. This 392 observations are just a sample of the original population. On the other hand, the glucose level and BMI seem to follow a normal distribution. When performing any analysis, it is always good to know what is the distribution of the data so all the assumptions for different tests or models can be met.  



```{r}
par(mfrow = c(1, 2))

# boxplot
with(diabetes, boxplot(dpf ~ outcome, 
                       ylab = "Diabetes Pedigree Function", 
                       xlab = "Presence of Diabetes",
                       main = "Figure A",
                       outline = FALSE))

# subsetting based on response
with <- diabetes[diabetes$outcome == 1, ]
without <- diabetes[diabetes$outcome == 0, ]

# density plot
plot(density(with$glucose), 
     xlim = c(0, 250),
     ylim = c(0.00, 0.02),
     xlab = "Glucose Level",
     main = "Figure B",
     lwd = 2)
lines(density(without$glucose), 
      col = "red",
      lwd = 2)
legend("topleft", 
       col = c("black", "red"), 
       legend = c("With Diabetes", "Without Diabetes"), 
       lwd = 2,
       bty = "n")

# simple two sample t-test with unequal variance
t.test(with$dpf, without$dpf)
```


In here, we can use other plots as well such as boxplot or density plot to look at the difference in data and comparing the diabetic and non diabetic patients. In the above figure, we can see that the distribution has slightly shifted to left and with the we can conclude that the blood glucose level is lower for the people with no diabetes.



```{r}
cor_melt <- melt(cor(diabetes[, 1:8]))
cor_melt <- cor_melt[which(cor_melt$value > 0.5 & cor_melt$value != 1), ]
cor_melt <- cor_melt[1:3, ]
cor_melt
```

We can also create a table of the correlations between the variables, and keep only those pairs with correlation values higher than 0.5. However, this is not a good indicator of correlations between the variables as there might be some other unknown interaction effects not taken into account to. Next, we'll use LASSO regression to fit a model for this data set, and perform simple predictions by splitting the data set into training and validation set.  




```{r}
# creating a random set of observations to be in the training set
set.seed(100)
inTrain <- sample(x = seq(1, 392), size = 294, replace = FALSE)

# preparing the inputs for the function cv.glmnet()
# you can use ?glmnet to understand more
x <- model.matrix(outcome ~ . - 1, data = diabetes)
y <- diabetes$outcome

# model fitting with lasso (alpha = 1)
# since response is binary, we'll set the [family = "binomial"] in the argument
# lasso regression also perform variable selection to determine which are the important variables

fit.lasso.cv <- cv.glmnet(x[inTrain, ], y[inTrain], alpha = 1, family = "binomial")
plot(fit.lasso.cv)

print(paste0("minimum binomial deviance = ", round(min(fit.lasso.cv$cvm), 3)))
print(paste0("log(lambda) with minimum binomial deviance = ", round(log(fit.lasso.cv$lambda.min), 3)))
coef(fit.lasso.cv)

# prediction with the validation data set
pred <- predict(fit.lasso.cv, newx = x[-inTrain, ])
pred <- exp(pred) / (1 + exp(pred))
pred <- ifelse(pred >= 0.5, 1, 0)
table(pred, y[-inTrain])

# calculate the accuracy
correct_pred <- sum(table(pred, y[-inTrain])[c(1, 4)])
total <- length(y[-inTrain])
acc <- correct_pred / total
print(paste0("accuracy = ", round(acc, 3)))
```


Using a portion of the data set as the training data, we fitted a lasso regression model for the data. By analysing the same, we can say that out of all the variables, BMI, Glucose level , Age and DPF are the most important factors to indicate the presence or absence of diabetes. With Lasso regression, we got an accuracy of 0.755.


We can also use other techniques like Logistic Regression, Random Forest and Decision Tree. It completely depends on the data that is given to us.    


Fitting with logistic regression using glm(). We need to take the exp() of the predicted values in order to get the probability response. Plotting the model gives some diagnostic plots needed to identify those observations with high leverages or that are outliers. These points may have to be removed for the model to be fitted better to the training set.  


```{r}
fit.glm <- glm(outcome ~ ., data = diabetes[inTrain, ], family = binomial)
pred.glm.logistic <- predict(fit.glm, diabetes[-inTrain, ])
pred.glm <- exp(pred.glm.logistic) / (1 + exp(pred.glm.logistic))
pred.glm <- as.integer(pred.glm >= 0.5)

par(mfrow = c(2, 2))
plot(fit.glm)
```


Random forest. There are a handful of tuning parameters that can be adjusted to ensure a better fit using random forest, namely the mtry and ntree. However, we did not go into details on adjusting these parameters.


```{r}
set.seed(123)
fit.rf <- randomForest(outcome ~ .,
                       diabetes[inTrain, ],
                       mtry = 3, # number of predictors to use for generation of tree 
                       ntree = 500, # number of trees to create
                       importance = TRUE)
pred.rf <- predict(fit.rf, diabetes[-inTrain, ])
confusionMatrix(pred.rf, y[-inTrain])[2:3]
importance(fit.rf)
varImpPlot(fit.rf)
```


Finally, we have implemented Decision Tree and found out that it has the highest Accuracy. Below is the tree that was generated.



```{r}
set.seed(123)
fit.tree <- tree(outcome ~ ., 
                 data = diabetes[inTrain, ])
pred.tree <- predict(fit.tree, diabetes[-inTrain, ], type = "class")
confusionMatrix(pred.tree, y[-inTrain])[2:3]
plot(fit.tree)
text(fit.tree, pretty = 0)
```


<font color = black face = "bold" size = 4>Conclusion and Key findings</font>

- One of the key factor that shows an obvious relation with diabetes is the presence of high glucose in blood.
- We found out that the factor Insulin significantly affected the prediction of diabetes between females.
- We did not find and obvious relation between Age and Diabetes offset.
- The Pedi Function or DPF also does not show and direct relation with Diabetes offset.
- With this, we can say that diabetes is not hereditary or we need more data to predict it accurately.
- Moreover, We have a limited data. To predict the diabetes offset more accurately, we need way more data than the one that we have.
- This data was collected in 1990's and there have been many new medical enhancements that can affect diabetes offset. 
