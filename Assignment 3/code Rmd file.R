
---
output:
  html_document: default
  pdf_document: default
---

<font color = black face = "bold" size = 6>Assignment 3</font>


<font color = black face = "bold" size = 5>By : Abhiram Dapke</font>


<font color = black face = "bold" size = 4>Import the required Libraries</font>

```{r}
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(e1071)
library(pROC)
library(cowplot)
library(ROCR)
library(gbm)
library(lattice)
library(grid)
library(cwhmisc)
theme_set(theme_cowplot())
```

<font color = black face = "bold" size = 4> 1. Download and store the Data </font>

<font color = black face = "bold" size = 4>Saving the qbtrain data as train.data</font>

```{r}
train.data <- read.csv('C:\\Users\\abhir\\Desktop\\ENPM-808W\\HW3\\qb.train.csv')
str(train.data)
```

<font color = black face = "bold" size = 4>Saving the qbtest data as test.data</font>

```{r}
test.data <- read.csv('C:\\Users\\abhir\\Desktop\\ENPM-808W\\HW3\\qb.test.csv')
str(test.data)
```

<font color = black face = "bold" size = 4>Create a new Variable :  <i>parent_match</i></font>

<font color = black face = "bold" size = 4> the variable parent_match stores the value to be TRUE if the value in the brackets is in the text otherwise it stores FALSE </font>

```{r}
parent_match <- function(page, text) {
  start <- cpos(page, "(")
  end <- cpos(page, ")")
  if (!is.na(start) && !is.na(end)) {
    search <- substring(page, start + 1, end - 1)
    return(grepl(tolower(search), tolower(text), fixed=TRUE))
  } else {
    return(FALSE)
  }
}
```

<font color = black face = "bold" size = 4> Scale the Train Data </font>

<font color = black face = "bold" size = 4> the variables obs_len, body_score and inlinks have high values, so we scale these variables for train data  </font>

```{r}
train.data$obs_len <- apply(train.data, 1, function(x) {nchar(x['text'])})
train.data$scale_len <- scale(train.data$obs_len)

train.data$scale_score <- scale(train.data$body_score)

train.data$parent_match <- apply(train.data, 1, function(x) {parent_match(x['page'], x['text'])})

train.data$log_links <- scale(log(as.numeric(train.data$inlinks) + 1))
```

<font color = black face = "bold" size = 4> Scale the Test Data </font>

<font color = black face = "bold" size = 4> the variables obs_len, body_score and inlinks have high values, so we scale these variables for test data  </font>

```{r}
test.data$obs_len <- apply(test.data, 1, function(x) {nchar(x['text'])})
test.data$scale_len <- scale(test.data$obs_len)

test.data$scale_score <- scale(test.data$body_score)

test.data$parent_match <- apply(test.data, 1, function(x) {parent_match(x['page'], x['text'])})

test.data$log_links <- scale(log(as.numeric(test.data$inlinks) + 1))
```

<font color = black face = "bold" size = 4>Dividing the Train Data </font>

<font color = black face = "bold" size = 4> trainset and testset of train data </font>

```{r}
index <- 1:nrow(train.data)
testindex <- sample(index, trunc(length(index)/5))
testset <- train.data[testindex,]
trainset <- train.data[-testindex,]
```

<font color = black face = "bold" size = 4> Most Frequent Baseline (mfc_baseline) </font>

```{r}
mfc_baseline <- sum(testset$corr == "False") / nrow(testset)

```



<font color = black face = "bold" size = 5> 2.(a) Different Classifiers </font>

<font color = black face = "bold" size = 5> I. SVM</font>

<font color = black face = "bold" size = 4> Create a function for calculating SVM</font>

```{r}
svm.per <- function(df, name, model, test) {
  svm.pred <- predict(model, test)
  svm.table <- table(pred = svm.pred, true=test$corr)
  df <- rbind(df, data.frame(model=c(name), score=c(classAgreement(svm.table)$diag)))
  return(df)
}
```

<font color = black face = "bold" size = 4> <i> i. Radial Kernel </i> </font>

<font color = black face = "bold" size = 4> Calculating accuracy for all types of models</font>

```{r}
svm.result <- data.frame(model=c("MFC"), score=c(mfc_baseline))

svm.result <- svm.per(svm.result, "body_score", svm(corr ~ body_score, data=trainset), testset)
svm.result <- svm.per(svm.result, "scale_score", svm(corr ~ scale_score, data=trainset), testset)
svm.result <- svm.per(svm.result, "obs_len", svm(corr ~ obs_len, data=trainset), testset)
svm.result <- svm.per(svm.result, "score+len", svm(corr ~ obs_len + body_score, data=trainset), testset)
svm.result <- svm.per(svm.result, "paren+len", svm(corr ~ obs_len + parent_match, data=trainset), testset)
svm.result <- svm.per(svm.result, "parent_match", svm(corr ~ parent_match, data=trainset), testset)
svm.result <- svm.per(svm.result, "score+parent_match", svm(corr ~ scale_score + parent_match, data=trainset), testset)
svm.result <- svm.per(svm.result, "score+len+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset), testset)
svm.result <- svm.per(svm.result, "links", svm(corr ~ inlinks, data=trainset), testset)
svm.result <- svm.per(svm.result, "loglinks", svm(corr ~ log_links, data=trainset), testset)
svm.result <- svm.per(svm.result, "score+len+links+parent_match", svm(corr ~ scale_len + scale_score + log_links + parent_match, data=trainset), testset)
svm.result <- svm.per(svm.result, "score+links+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset), testset)
svm.result
```

<font color = black face = "bold" size = 4> Finding the model with maximum accuracy</font>

```{r}
svm.result[which.max(svm.result$score),] 
```


<font color = black face = "bold" size = 4> <i> ii. Linear Kernel </i> </font>

<font color = black face = "bold" size = 4> Calculating accuracy for all types of models</font>

```{r}
svm.lk.result <- data.frame(model=c("MFC"), score=c(mfc_baseline))

svm.lk.result <- svm.per(svm.lk.result, "body_score", svm(corr ~ body_score, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "scale_score", svm(corr ~ scale_score, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "obs_len", svm(corr ~ obs_len, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "score+len", svm(corr ~ obs_len + body_score, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "paren+len", svm(corr ~ obs_len + parent_match, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "parent_match", svm(corr ~ parent_match, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "score+parent_match", svm(corr ~ scale_score + parent_match, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "score+len+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "links", svm(corr ~ inlinks, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "loglinks", svm(corr ~ log_links, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "score+len+links+parent_match", svm(corr ~ scale_len + scale_score + log_links + parent_match, data=trainset, kernel = 'linear'), testset)
svm.lk.result <- svm.per(svm.lk.result, "score+links+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset, kernel = 'linear'), testset)
svm.lk.result
```

<font color = black face = "bold" size = 4> Finding the model with maximum accuracy</font>

```{r}
svm.lk.result[which.max(svm.lk.result$score),] 
```

<font color = black face = "bold" size = 4> <i> iii. Polynomial Kernel </i> </font>

<font color = black face = "bold" size = 4> Calculating accuracy for all types of models</font>

```{r}
svm.pk.result <- data.frame(model=c("MFC"), score=c(mfc_baseline))

svm.pk.result <- svm.per(svm.pk.result, "body_score", svm(corr ~ body_score, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "scale_score", svm(corr ~ scale_score, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "obs_len", svm(corr ~ obs_len, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "score+len", svm(corr ~ obs_len + body_score, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "paren+len", svm(corr ~ obs_len + parent_match, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "parent_match", svm(corr ~ parent_match, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "score+parent_match", svm(corr ~ scale_score + parent_match, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "score+len+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "links", svm(corr ~ inlinks, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "loglinks", svm(corr ~ log_links, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "score+len+links+parent_match", svm(corr ~ scale_len + scale_score + log_links + parent_match, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result <- svm.per(svm.pk.result, "score+links+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset, kernel = 'polynomial'), testset)
svm.pk.result
```

<font color = black face = "bold" size = 4> Finding the model with maximum accuracy</font>

```{r}
svm.pk.result[which.max(svm.pk.result$score),] 
```

<font color = black face = "bold" size = 4> <i> iv. Sigmoid Kernel </i> </font>

<font color = black face = "bold" size = 4> Calculating accuracy for all types of models</font>

```{r}
svm.sk.result <- data.frame(model=c("MFC"), score=c(mfc_baseline))

svm.sk.result <- svm.per(svm.sk.result, "body_score", svm(corr ~ body_score, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "scale_score", svm(corr ~ scale_score, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "obs_len", svm(corr ~ obs_len, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "score+len", svm(corr ~ obs_len + body_score, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "paren+len", svm(corr ~ obs_len + parent_match, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "parent_match", svm(corr ~ parent_match, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "score+parent_match", svm(corr ~ scale_score + parent_match, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "score+len+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "links", svm(corr ~ inlinks, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "loglinks", svm(corr ~ log_links, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "score+len+links+parent_match", svm(corr ~ scale_len + scale_score + log_links + parent_match, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result <- svm.per(svm.sk.result, "score+links+parent_match", svm(corr ~ scale_len + scale_score + parent_match, data=trainset, kernel = 'sigmoid'), testset)
svm.sk.result
```

<font color = black face = "bold" size = 4> Finding the model with maximum accuracy</font>

```{r}
svm.sk.result[which.max(svm.sk.result$score),] 
```

<font color = black face = "bold" size = 4> Comparing accuracies of all the SVM Kernels </font>

<font color = black face = "bold" size = 4><i> Table containing accuracy values for all svm models </i> </font>

```{r}
svm.accuracies <- cbind(svm.result, svm.lk.result$score, svm.pk.result$score, svm.sk.result$score)
names(svm.accuracies) <- c('Model', 'svm.RadialKernel.accuracy','svm.LinearKernel.accuracy','svm.Polynomial.accuracy', 'svm.Sigmoid.accracy')
svm.accuracies

```

<font color = black face = "bold" size = 4> Within all the Kernels , Radial kernel has better accuracies and its maximum acuracy is for the model score+len and accuracy is 0.8148607</font>

<font color = black face = "bold" size = 4> In Radial SVM the Model <i>score+len</i> has highest accuracy so predicted values for that model is as follows </font>


```{r}
mod.svm <-  svm(corr ~ scale_len + scale_score, data=train.data)
summary(mod.svm)
```
<font color = black face = "bold" size = 4> Predicting mod.svm on test Data </font>

```{r}

pred.svm <- data.frame(cbind(test.data$row, predict(mod.svm,test.data)))
names(pred.svm )<- c('row','corr')
pred.svm$corr[pred.svm$corr==1]<- 'FALSE'
pred.svm$corr[pred.svm$corr==2]<- 'TRUE'
head(pred.svm,10)
write.csv(pred.svm,'svm.csv',row.names = F)
```


<font color = black face = "bold" size = 5> II. Decision Tree</font>

<font color = black face = "bold" size = 4> Create a function for calculating accuracy for all Decision tree Models</font>

```{r}
dt.per <- function(df, name, model, test) {
  dt.pred <- predict(model, test, type = 'class')
  confmat <- table(test$corr,dt.pred)
   acc <- sum(diag(confmat))/sum(confmat)
  df <- rbind(df, data.frame(model=c(name), accuracy=c(acc)))
  return(df)
}
```

<font color = black face = "bold" size = 4> Calculating accuracy for all types of models</font>

```{r}
dt.result <- data.frame(model=c("MFC"), accuracy=c(mfc_baseline))
dt.result <- dt.per(dt.result,"body_score",rpart(corr ~ body_score, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"scale_score",rpart(corr ~ scale_score, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"obs_len",rpart(corr ~ obs_len, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"score+len",rpart(corr ~ obs_len + body_score, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"paren+len",rpart(corr ~ obs_len + parent_match, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"parent_match",rpart(corr ~ parent_match, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"score+parent_match",rpart(corr ~ scale_score + parent_match, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"score+len+parent_match",rpart(corr ~ scale_len + scale_score + parent_match, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"links",rpart(corr ~ inlinks, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"loglinks",rpart(corr ~ log_links, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"score+len+links+parent_match",rpart(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data = trainset), testset)
dt.result <- dt.per(dt.result,"score+links+parent_match",rpart(corr ~ scale_len + scale_score + parent_match, method = 'class', data = trainset), testset)

dt.result

```

<font color = black face = "bold" size = 4> Finding the model with maximum accuracy</font>

```{r}
dt.result[which.max(dt.result$accuracy),] 
```

<font color = black face = "bold" size = 4> In Decision Tree Model <i>score+len+links+parent_match</i> has highest accuracy so predicted values for that model is as follows </font>


```{r}
mod.dt <-  rpart(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data=train.data)
printcp(mod.dt)
```

<font color = black face = "bold" size = 4> Predicting mod.dt on test Data </font>

```{r}

pred.dt <- data.frame(cbind(test.data$row, predict(mod.dt,test.data,type = 'class')))
names(pred.dt )<- c('row','corr')
pred.dt$corr[pred.dt$corr==1]<- 'FALSE'
pred.dt$corr[pred.dt$corr==2]<- 'TRUE'
head(pred.dt,10)
write.csv(pred.dt,'dTree.csv',row.names = F)
```


<font color = black face = "bold" size = 5> III. Logistic Model</font>

<font color = black face = "bold" size = 4> Create a function for calculating accuracy for all Logistic Models</font>

```{r}
logm.per <- function(df, name, model, test) {
  logm.pred <- predict(model, test, type = 'response')
  confmat <- table(test$corr,logm.pred > 0.5)
   acc <- sum(diag(confmat))/sum(confmat)
  df <- rbind(df, data.frame(model=c(name), accuracy=c(acc)))
  return(df)
}
```

<font color = black face = "bold" size = 4> Calculating accuracy for all types of models</font>

```{r}
logm.result <- data.frame(model=c("MFC"), accuracy=c(mfc_baseline))
logm.result <- logm.per(logm.result,"body_score",glm(corr ~ body_score, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"scale_score",glm(corr ~ scale_score, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"obs_len",glm(corr ~ obs_len, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"score+len",glm(corr ~ obs_len + body_score, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"paren+len",glm(corr ~ obs_len + parent_match, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"parent_match",glm(corr ~ parent_match, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"score+parent_match",glm(corr ~ scale_score + parent_match, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"score+len+parent_match",glm(corr ~ scale_len + scale_score + parent_match, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"links",glm(corr ~ inlinks, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"loglinks",glm(corr ~ log_links, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"score+len+links+parent_match",glm(corr ~ scale_len + scale_score + log_links + parent_match, family = binomial(link = 'logit'), data = trainset), testset)
logm.result <- logm.per(logm.result,"score+links+parent_match",glm(corr ~ scale_len + scale_score + parent_match, family = binomial(link = 'logit'), data = trainset), testset)

logm.result

```

<font color = black face = "bold" size = 4> Finding the model with maximum accuracy</font>

```{r}
logm.result[which.max(logm.result$accuracy),] 
```

<font color = black face = "bold" size = 4> In Logistic Model, the Model <i>score+len+parent_match</i> has highest accuracy so predicted values for that model are: </font>


```{r}
mod.logm <-  glm(corr ~ scale_len + scale_score + parent_match, family = binomial(link = 'logit'), data=train.data)
summary(mod.logm)
```

<font color = black face = "bold" size = 4> Predicting mod.logm on test Data </font>

```{r}

pred.logm <- data.frame(cbind(test.data$row, predict(mod.logm, test.data, type = 'response')>0.5))
names(pred.logm )<- c('row','corr')
pred.logm$corr[pred.logm$corr==0]<- 'FALSE'
pred.logm$corr[pred.logm$corr==1]<- 'TRUE'
head(pred.logm,10)
write.csv(pred.logm,'logModel.csv',row.names = F)
```

<font color = black face = "bold" size = 4><i> Table containing accuracy values for all models with all three types of classifiers : SVM, Logistic Model and Decision Tree  </i> </font>

```{r}
accuracies <- cbind(svm.accuracies ,logm.result$accuracy, dt.result$accuracy)
names(accuracies) <- c('Model', 'svm.RadialKernel.accuracy','svm.LinearKernel.accuracy','svm.Polynomial.accuracy', 'svm.Sigmoid.accracy','logReg_Accuracy', 'DTree_Accuracy')
accuracies
```

<font color = black face = "bold" size = 5>2(b) Error Analysis </font>

<font color = black face = "bold" size = 5>I. SVM </font>

<font color = black face = "bold" size = 4><i>We chose SVM kernel to be Radial as it gave best result.</i> </font>

<font color = black face = "bold" size = 4>Model selected is scale_len+scale_score </font>

<font color = black face = "bold" size = 4>Creating confusion matrix for SVM Model </font>

```{r}
e.p1 <- predict(svm(corr ~ scale_len + scale_score, data=trainset),testset)
tab1 <- table(Predicted = e.p1, Actual = testset$corr)
tab1
```

<font color =black face = "bold" size = 4>Making ROC curve and finding values of accuracy,error, sensitivity, precision, false positive rate and Area under Curve</font>

```{r}

m1 <-svm(corr ~ scale_len + scale_score , data=trainset, probability=T)
p1 <- predict(m1,testset, probability = T)
pred1 <- prediction(attr(p1,'probabilities')[, -2],testset$corr)
roc1 <- performance(pred1,'tpr','fpr')
auc1 <- performance(pred1,'auc')
auc1 <- round(unlist(slot(auc1,'y.values')),4)
a1 <- (tab1[1,1]+ tab1[2,2])/sum(tab1)
er1 <- (tab1[1,2]+tab1[2,1])/sum(tab1)
s1 <- tab1[2,2]/(tab1[2,1]+tab1[2,2])
p1 <- tab1[2,2]/(tab1[1,2]+tab1[2,2])
fpr1 <- tab1[1,2]/(tab1[1,1]+tab1[1,2])
#plot
df1 <- data.frame('SVM', a1, er1, s1, p1, fpr1,auc1)
names(df1) <- c('Type of Classification', 'Accuracy', 'Error Rate', 'Sensitivity', 'Precision', 'False Positive Rate','Area Under Curve')
plot(roc1,col='black',xlab = '1- Specificity', ylab ='Sensitivity')
legend(0.4,0.2,round(auc1,4),title='AUC',cex=0.8)
abline(a=0,b=1)
```

<font color =black face = "bold" size = 4> Creating new variables like errorType and predType  </font>

```{r}
new.testset.svm <- testset
new.svm <-svm(corr ~ scale_len + scale_score, data=trainset)
new.p.svm <- predict(new.svm,new.testset.svm)
new.testset.svm$predval <- new.p.svm
```


```{r}

for (i in 1:(nrow(new.testset.svm)-1)) {
  if(new.testset.svm$predval[i] == 'True' && new.testset.svm$corr[i] == 'True')
  {
    new.testset.svm$errorType[i] <- 'TP'
  }
  else if(new.testset.svm$predval[i] == 'True' && new.testset.svm$corr[i] == 'False')
  {
    new.testset.svm$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.svm$predval[i] == 'False' && new.testset.svm$corr[i] == 'True')
  {
    new.testset.svm$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.svm$errorType[i] <- 'TN'
  }
}


for (i in 1:(nrow(new.testset.svm)-1)) {
  if(new.testset.svm$errorType[i] == 'TP' || new.testset.svm$errorType[i] == 'TN')
  {
    new.testset.svm$predType[i] <- 'Correct Prediction'
  }
  else 
  {
     new.testset.svm$predType[i] <- 'Wrong Prediction'
  } 
  
}
head(new.testset.svm)
```

```{r}
f.svm <- data.frame(table(new.testset.svm$errorType))
f.svm
```

<font color = black face = "bold" size = 5>II. Decision Tree </font>

<font color = black face = "bold" size = 4>Model selected is score+len+links+parent_match  </font>

<font color = black face = "bold" size = 4>Creating confusion matrix for Decision Tree Model </font>

```{r}
e.p2 <- predict(rpart(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data=trainset),testset, type = 'class')
tab2 <- table(Predicted = e.p2, Actual = testset$corr)
tab2

```

<font color =black face = "bold" size = 4>Making ROC curve and finding values of accuracy,error, sensitivity, precision, false positive rate and Area under Curve</font>

```{r}
m2 <-rpart(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data=trainset)
p2 <- predict(m2,testset, type = 'prob')
auc2 <- as.numeric(auc(testset$corr, p2[,2]))
a2 <- (tab2[1,1]+ tab2[2,2])/sum(tab2)
er2 <- (tab2[1,2]+tab2[2,1])/sum(tab2)
s2 <- tab2[2,2]/(tab2[2,1]+tab2[2,2])
p2 <- tab2[2,2]/(tab2[1,2]+tab2[2,2])
fpr2 <- tab2[1,2]/(tab2[1,1]+tab2[1,2])
df2 <- data.frame('Decision Tree', a2, er2, s2, p2, fpr2, auc2)
names(df2) <- c('Type of Classification', 'Accuracy', 'Error Rate', 'Sensitivity', 'Precision', 'False Positive Rate','Area Under Curve')
df2

#plot ROC
m2 <-rpart(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data=trainset)
p2 <- predict(m2,testset, type = 'prob')
plot(roc(testset$corr,p2[,2]),col='black', ylab='sensitivity', xlab = 'Specificity')
legend(0.4,0.2,round(auc2,4),title='AUC',cex=0.8)
```

<font color =black face = "bold" size = 4> Creating new variables like errorType and predType  </font>

```{r}
new.testset.dt <- testset
new.dt <-rpart(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data=trainset)
new.p.dt <- predict(new.dt,new.testset.dt, type = 'class')
new.testset.dt$predval <- new.p.dt

```


```{r}

for (i in 1:(nrow(new.testset.dt)-1)) {
  if(new.testset.dt$predval[i] == 'True' && new.testset.dt$corr[i] == 'True')
  {
    new.testset.dt$errorType[i] <- 'TP'
  }
  else if(new.testset.dt$predval[i] == 'True' && new.testset.dt$corr[i] == 'False')
  {
    new.testset.dt$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.dt$predval[i] == 'False' && new.testset.dt$corr[i] == 'True')
  {
    new.testset.dt$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.dt$errorType[i] <- 'TN'
  }
}

for (i in 1:(nrow(new.testset.dt)-1)) {
  if(new.testset.dt$errorType[i] == 'TP' || new.testset.dt$errorType[i] == 'TN')
  {
    new.testset.dt$predType[i] <- 'Correct Prediction'
  }
  else 
  {
     new.testset.dt$predType[i] <- 'Wrong Prediction'
  } 
  
}

head(new.testset.dt)
```

```{r}
f.dt <- data.frame(table(new.testset.dt$errorType))
f.dt
```

<font color = black face = "bold" size = 5>III. Logistic Model </font>

<font color = black face = "bold" size = 4>Model selected is score+len+parent_match  </font>

<font color = black face = "bold" size = 4>Creating confusion matrix for Logistic Model </font>

```{r}
e.p3 <- (predict(glm(corr ~ scale_len + scale_score + parent_match, family = binomial(link = 'logit'), data=trainset),testset, type = 'response'))>0.5
tab3 <- table(Predicted = e.p3, Actual = testset$corr)
tab3


```

<font color =black face = "bold" size = 4>Making ROC curve and finding values of accuracy,error, sensitivity, precision, false positive rate and Area under Curve</font>

```{r}
p3 <- predict(glm(corr ~ scale_len + scale_score + parent_match, family = binomial(link = 'logit'), data=trainset),testset, type = 'response')
pred <- prediction(p3,testset$corr)
roc3 <- performance(pred,'tpr','fpr')
auc3 <- performance(pred,'auc')
auc3 <- round(unlist(slot(auc3,'y.values')),4)
a3 <- (tab3[1,1]+ tab3[2,2])/sum(tab3)
er3 <- (tab3[1,2]+tab3[2,1])/sum(tab3)
s3 <- tab3[2,2]/(tab3[2,1]+tab3[2,2])
p3 <- tab3[2,2]/(tab3[1,2]+tab3[2,2])
fpr3 <- tab3[1,2]/(tab3[1,1]+tab3[1,2])
df3 <- data.frame('Logistic Model', a3, er3, s3, p3, fpr3, auc3)
names(df3) <- c('Type of Classification', 'Accuracy', 'Error Rate', 'Sensitivity', 'Precision', 'False Positive Rate','Area Under Curve')

#plot
plot(roc3, col='black',xlab = '1- Specificity', ylab ='Sensitivity')
abline(a=0,b=1)
legend(0.4,0.2,auc3,title='AUC',cex=0.8)
```

<font color =black face = "bold" size = 4> Creating new variables like errorType and predType  </font>

```{r}
new.testset.lm <- testset
new.lm <-glm(corr ~ scale_len + scale_score + parent_match, family = binomial(link = 'logit'), data=trainset)
new.p.lm <- (predict(new.lm,new.testset.lm,type = 'response'))>0.5
new.testset.lm$predval <- new.p.lm
```


```{r}

for (i in 1:(nrow(new.testset.lm)-1)) {
  if(new.testset.lm$predval[i] == 'TRUE' && new.testset.lm$corr[i] == 'True')
  {
    new.testset.lm$errorType[i] <- 'TP'
  }
  else if(new.testset.lm$predval[i] == 'TRUE' && new.testset.lm$corr[i] == 'False')
  {
    new.testset.lm$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.lm$predval[i] == 'FALSE' && new.testset.lm$corr[i] == 'True')
  {
    new.testset.lm$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.lm$errorType[i] <- 'TN'
  }
}


for (i in 1:(nrow(new.testset.lm)-1)) {
  if(new.testset.lm$errorType[i] == 'TP' || new.testset.lm$errorType[i] == 'TN')
  {
    new.testset.lm$predType[i] <- 'Correct Prediction'
  }
  else 
  {
     new.testset.lm$predType[i] <- 'Wrong Prediction'
  } 
  
}

head(new.testset.lm)
```

```{r}
f.lm <- data.frame(table(new.testset.lm$errorType))
f.lm

```

<font color =black face = "bold" size = 4> Combining Accuracy, Error Rate, Sensitivity, Precision, False Positive Rate and Area Under Curve for all the classifiers   </font>

```{r}

error.analysis.df <- rbind(df1, df2, df3)
error.analysis.df
```

<font color =black face = "bold" size = 4> SVM has better accuracy, precision and Area Under Curve as compared to other classifiers  </font>

<font color =black face = "bold" size = 4> Combining Error Type</font>

```{r}

co.df <- cbind(f.svm[1:2,],f.dt[1:2,2],f.lm[1:2,2])
names(co.df) <- c('Error Type','SVM value','DecisionTree value','LogisticModel value')
co.df
```

<font color =black face = "bold" size = 4> SVM has least Type II error as compared to all other classifiers </font>

<font color =black face = "bold" size = 4>Graphical Representation of error</font>

```{r fig.height = 18, fig.width = 18, fig.align = "center"}

svm.plt <-ggplot(data= new.testset.svm, aes(x=predType, fill=errorType))+ geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ ggtitle('SVM')

dt.plt <- ggplot(data= new.testset.dt, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Decision Tree')

lm.plt <- ggplot(data= new.testset.lm, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Logistic Model')

plot_grid(svm.plt,dt.plt,lm.plt)
```

<font face = "bold" size = 4> <b>Above classifiers have following Patterns: </b> </font>

<font color =black face = "bold" size = 4>1. Type II error for SVM is less as compablack to other classifiers but this error can be improved by ensemble techniques like Bagging, Boosing , Random Forest, etc. </font>

<font color =black face = "bold" size = 4>2. Total error for svm and decision tree is almost same. </font>

<font color =black face = "bold" size = 4>3. Area under the curve for both svm and Decison tree is almost same but it can be increased. </font>

<font color =black face = "bold" size = 4>4. All three classifiers give maximum accuracy for different models. But it is seen that scale_score and Scale_len is common between all three classifiers </font>

<font color =black face = "bold" size = 4>5. We need a classifier which can boot the performance of these classifiers </font>

<font color = black face = "bold" size = 5> 3(a) </font>

<font color = black face = "bold" size = 5>Improving Predictions </font>

<font face = "bold" size = 4> <b>Ways to Improve the classifier:  </b></font>

<font color =black face = "bold" size = 4>1. SVM and Decision Tree give almost same result but decision tree is more reliable and its performance can be enhanced using ensemble methods, where many trees are fit and predictions are aggregated accross trees. Example include bagging, boosting and RandomForest.</font>

<font color =black face = "bold" size = 4>2.Amongst these methods, we choose Random Forest for improving the predictions </font>

<font face = "bold" size = 4><b> Random Forest is chosen as a classifier to improve the prediction because:  </b></font>

<font color =black face = "bold" size = 4>Random Forest produces large number of decision trees. This approach first takes a random sample of data and identifies a key set of features to grow each decision tree. These decision trees then have their OUt of Bag error determined. Then collection of decision trees are compared to find joint set of variables that creates strongest classification model.</font>

<font color =black face = "bold" size = 4><i>Assuming that Random Forest improves the performance we can combine it with next best classifier i.e SVM and try to improve the performance further</i></font>

<font color =black face = "bold" size = 4>Checking Whether Random Forest Improves the performance or not.</font>

<font color = black face = "bold" size = 5> I. Random Forest</font>

<font color = black face = "bold" size = 4> Creating a function for calculating accuracy for all Random Forest Models</font>

```{r}
rf.per <- function(df, name, model, test) {
  rf.pred <- predict(model, test, type = 'class')
  confmat <- table(test$corr,rf.pred)
   acc <- sum(diag(confmat))/sum(confmat)
  df <- rbind(df, data.frame(model=c(name), accuracy=c(acc)))
  return(df)
}
```

<font color = black face = "bold" size = 4> Calculating accuracy for all types of models</font>

```{r}
rf.result <- data.frame(model=c("MFC"), accuracy=c(mfc_baseline))
rf.result <- rf.per(rf.result,"body_score",randomForest(corr ~ body_score, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"scale_score",randomForest(corr ~ scale_score, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"obs_len",randomForest(corr ~ obs_len, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"score+len",randomForest(corr ~ obs_len + body_score, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"paren+len",randomForest(corr ~ obs_len + parent_match, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"parent_match",randomForest(corr ~ parent_match, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"score+parent_match",randomForest(corr ~ scale_score + parent_match, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"score+len+parent_match",randomForest(corr ~ scale_len + scale_score + parent_match, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"links",randomForest(corr ~ inlinks, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"loglinks",randomForest(corr ~ log_links, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"score+len+links+parent_match",randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data = trainset), testset)
rf.result <- rf.per(rf.result,"score+links+parent_match",randomForest(corr ~ scale_len + scale_score + parent_match, method = 'class', data = trainset), testset)

rf.result

```

<font color = black face = "bold" size = 4> Finding the model with maximum accuracy</font>

```{r}
rf.result[which.max(rf.result$accuracy),] 
```

<font color = black face = "bold" size = 4> In Random Forest Model <i>score+len+links+parent_match</i> has highest accuracy so pblackicted values for that model is as follows </font>

```{r}
mod.rf <-  randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data=train.data)
mod.rf

```

<font color = black face = "bold" size = 4> Predicting mod.rf on test Data </font>

```{r}

pred.rf <- data.frame(cbind(test.data$row, predict(mod.rf,test.data,type = 'class')))
names(pred.rf )<- c('row','corr')
pred.rf$corr[pred.rf$corr==1]<- 'FALSE'
pred.rf$corr[pred.rf$corr==2]<- 'TRUE'
head(pred.rf,10)
write.csv(pred.rf,'randomForest.csv',row.names = F)
```

<font color = black face = "bold" size = 5>error Analysis for Random Forest</font>


<font color = black face = "bold" size = 4>Model selected is score+len+links+parent_match  </font>

<font color = black face = "bold" size = 4>Creating confusion matrix for random forest Classifier </font>

```{r}
e.p4 <- predict(randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data= trainset),testset, type = 'class')
tab4 <- table(Predicted = e.p4, Actual = testset$corr)
tab4

```

<font color =black face = "bold" size = 4>Making ROC curve and finding values of accuracy,error, sensitivity, precision, false positive rate and Area under Curve</font>

```{r}
m4 <-randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data= trainset)
p4 <- predict(m4,testset, type = 'prob')
auc4 <-as.numeric( auc(testset$corr, p4[,2]))
a4 <- (tab4[1,1]+ tab4[2,2])/sum(tab4)
er4 <- (tab4[1,2]+tab4[2,1])/sum(tab4)
s4 <- tab4[2,2]/(tab4[2,1]+tab4[2,2])
p4 <- tab4[2,2]/(tab4[1,2]+tab4[2,2])
fpr4 <- tab4[1,2]/(tab4[1,1]+tab4[1,2])
df4 <- data.frame('Random Forest', a4, er4, s4, p4, fpr4,auc4)
names(df4) <- c('Type of Classification', 'Accuracy', 'Error Rate', 'Sensitivity', 'Precision', 'False Positive Rate', 'Area Under Curve')
m4 <-randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data= trainset)
p4 <- predict(m4,testset, type = 'prob')
plot(roc(testset$corr,p4[,2]),col='black', ylab='sensitivity')
legend(0.4,0.2,round(auc4,4),title='AUC',cex=0.8)
```

<font color =black face = "bold" size = 4> Creating new variables like errorType and predType  </font>

```{r}
new.testset.rf <- testset
new.rf <-randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data= trainset,proximity=T)
new.p.rf <- predict(new.rf,new.testset.rf)
new.testset.rf$predval <- new.p.rf
```


```{r}

for (i in 1:(nrow(new.testset.rf)-1)) {
  if(new.testset.rf$predval[i] == 'True' && new.testset.rf$corr[i] == 'True')
  {
    new.testset.rf$errorType[i] <- 'TP'
  }
  else if(new.testset.rf$predval[i] == 'True' && new.testset.rf$corr[i] == 'False')
  {
    new.testset.rf$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.rf$predval[i] == 'False' && new.testset.rf$corr[i] == 'True')
  {
    new.testset.rf$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.rf$errorType[i] <- 'TN'
  }
}

for (i in 1:(nrow(new.testset.rf)-1)) {
  if(new.testset.rf$errorType[i] == 'TP' || new.testset.rf$errorType[i] == 'TN')
  {
    new.testset.rf$predType[i] <- 'Correct Prediction'
  }
  else 
  {
     new.testset.rf$predType[i] <- 'Wrong Prediction'
  } 
  
}

head(new.testset.rf)
```

```{r}
f.rf <- data.frame(table(new.testset.rf$errorType))
f.rf

```

<font color =black face = "bold" size = 4>Comparing ROC graph for all four classifiers</font>


```{r fig.height = 10, fig.width = 18, fig.align = "center"}
par(mfrow = c(2,2))

plot(roc1,col='black',xlab = '1- Specificity', ylab ='Sensitivity', main='SVM ROC')
legend(0.4,0.2,round(auc1,4),title='AUC',cex=0.8)
abline(a=0,b=1)

m2 <-rpart(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data=trainset)
p2 <- predict(m2,testset, type = 'prob')
plot(roc(testset$corr,p2[,2]),col='black', ylab='sensitivity', xlab = 'Specificity', main='Decision Tree ROC')
legend(0.4,0.2,round(auc2,4),title='AUC',cex=0.8)

#plot
plot(roc3, col='black',xlab = '1- Specificity', ylab ='Sensitivity', main='Logistic Model ROC')
abline(a=0,b=1)
legend(0.4,0.2,auc3,title='AUC',cex=0.8)

m4 <-randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data= trainset)
p4 <- predict(m4,testset, type = 'prob')
plot(roc(testset$corr,p4[,2]),col='black', ylab='sensitivity', xlab = 'Specificity', main='Random Forest ROC')
legend(0.4,0.2,round(auc4,4),title='AUC',cex=0.8)


```

<font color =black face = "bold" size = 4> From the graph it can be clearly seen that the roc for Decision tree is better than that of any other classifiers as it is reaching more towards 1. Also the AUC has increased to 0.88 for Random Forest Classifier</font>

<font color =black face = "bold" size = 4> Combining Error Type for all four classifiers</font>

```{r}

com.df <- cbind(co.df,f.rf[1:2,2])
names(com.df) <- c('Error Type','SVM value','DecisionTree value','LogisticModel value','Random Forest Value')
com.df
```

<font color =black face = "bold" size = 4> It can be seen that in Random Forest classifier  the Type II error has decreased and the Type _I error has increased. But the sum of both the errors is almost equal to the Type II error of svm. So Random Forest Classifier gives less error. Also, Type II error is more critical as ompablack to Type I error which is decreased by using Random forest Classifier </font>

<font color = black face = "bold" size = 4> Graphical Representation of Error</font>

```{r fig.height = 18, fig.width = 22, fig.align = "center"}

svm.plt <-ggplot(data= new.testset.svm, aes(x=predType, fill=errorType))+ geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ ggtitle('SVM')

dt.plt <- ggplot(data= new.testset.dt, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Decision Tree')

lm.plt <- ggplot(data= new.testset.lm, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Logistic Model')

lm.rf <- ggplot(data= new.testset.rf, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Random Forest')


plot_grid(svm.plt,dt.plt,lm.plt,lm.rf)
```


<font color = black face = "bold" size = 5> It can be seen that Random Forest classifier  not only decreases the Type II error and the over all error but it also increases the True Positives in the system thereby improving overall accuracy </font>


<font color = black face = "bold" size = 5> Combining Random Forest to prev data frame </font>


```{r}
error.analysis.df.comb <- rbind(error.analysis.df,df4)
error.analysis.df.comb
```

<font color =black face = "bold" size = 4> The accuracy of the model has increased and the error rate has decreased. Also the area under the curve has increased. But the sensitivity has decreased as compared to svm. To improve the classifier further we can combine the two best classifiers i.e svm and Random Forest</font>

<font color = black face = "bold" size = 5> II. Combined Random Forest and SVM Classifier</font>

<font color = black face = "bold" size = 4> Combining RandomForest and SVM models having maximum accuracy</font>

```{r}
c.mod.svm <- predict(svm(corr ~ scale_len + scale_score, data=trainset),testset)

c.mod.randFor <- predict(randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data= trainset),testset)
```

```{r}
num.svm <- as.numeric(c.mod.svm)
num.randFor<- as.numeric(c.mod.randFor)
comb.d <- round(0.5*num.randFor+ 0.5*num.svm )
comb.d[comb.d==2] <- 'True'
comb.d[comb.d==1] <- 'False'
comb.d <- as.factor(comb.d)

```

<font color = black face = "bold" size = 4> Finding the Confusion Matrix for combined classifier model</font>

```{r}
tc <-table(comb.d,testset$corr)
tc
```
<font color = black face = "bold" size = 4> Finding the accuracy</font>

```{r}

acc.c <- sum(diag(tc))/sum(tc)
acc.c
```

<font color = black face = "bold" size = 4> The combined Random Forest and SVM classifier has highest accuracy so predicted values for that model is as follows </font>

```{r}
mod.svm.c <- predict(svm(corr ~ scale_len + scale_score, data=train.data),test.data)

mod.randFor.c <- predict(randomForest(corr ~ scale_len + scale_score + log_links + parent_match, method = 'class', data= train.data),test.data)

```

<font color = black face = "bold" size = 4> Predicting mod.dt on test Data </font>

```{r}
n.svm <- as.numeric(mod.svm.c)
n.randFor <- as.numeric(mod.randFor.c)

c.data <-round(0.5*n.randFor+ 0.5*n.svm )
pred.c.data <- data.frame(cbind(test.data$row, c.data))
names(pred.c.data )<- c('row','corr')
pred.c.data$corr[pred.c.data$corr == 1]<-1
pred.c.data$corr[pred.c.data$corr==2]<-2
head(pred.c.data,10)
pred.c.data$corr <- as.double(pred.c.data$corr)
str(pred.c.data$corr)
write.csv(pred.c.data,'combined.csv',row.names = F)


```

<font color =black face = "bold" size = 4>Finding values of accuracy,error, sensitivity, precision, false positive rate and Area under Curve</font>

```{r}
aucc <- auc4
ac <- (tc[1,1]+ tc[2,2])/sum(tc)
erc <- (tc[1,2]+tc[2,1])/sum(tc)
sc <- tc[2,2]/(tc[2,1]+tc[2,2])
pc <- tc[2,2]/(tc[1,2]+tc[2,2])
fprc <- tc[1,2]/(tc[1,1]+tc[1,2])
df5 <- data.frame('SVM and Random Forest Combination', ac, erc, sc, pc, fprc,aucc)
names(df5) <- c('Type of Classification', 'Accuracy', 'Error Rate', 'Sensitivity', 'Precision', 'False Positive Rate', 'Area Under Curve')
df5
```

<font color =black face = "bold" size = 4> Creating new variables like errorType and predType  </font>

```{r}
new.testset.comb <- testset
new.testset.comb$predval <- comb.d
```


```{r}

for (i in 1:(nrow(new.testset.comb)-1)) {
  if(new.testset.comb$predval[i] == 'True' && new.testset.comb$corr[i] == 'True')
  {
    new.testset.comb$errorType[i] <- 'TP'
  }
  else if(new.testset.comb$predval[i] == 'True' && new.testset.comb$corr[i] == 'False')
  {
    new.testset.comb$errorType[i] <- 'FP- Type I error'
  } 
  else if(new.testset.comb$predval[i] == 'False' && new.testset.comb$corr[i] == 'True')
  {
    new.testset.comb$errorType[i] <- 'FN - Type II error'
  } 
  else
  {
    new.testset.comb$errorType[i] <- 'TN'
  }
}

for (i in 1:(nrow(new.testset.comb)-1)) {
  if(new.testset.comb$errorType[i] == 'TP' || new.testset.comb$errorType[i] == 'TN')
  {
    new.testset.comb$predType[i] <- 'Correct Prediction'
  }
  else 
  {
     new.testset.comb$predType[i] <- 'Wrong Prediction'
  } 
  
}

head(new.testset.comb)
```

```{r}
f.comb <- data.frame(table(new.testset.comb$errorType))
f.comb

```

<font color =black face = "bold" size = 4> Combining Error Type for all four classifiers</font>

```{r}

com.df.c <- cbind(com.df,f.comb[1:2,2])
names(com.df.c) <- c('Error Type','SVM value','DecisionTree value','LogisticModel value','Random Forest Value','SVM and Random Forest Combined')
com.df.c
```

<font color =black face = "bold" size = 4> It can be seen that the combined Random Forest and SVM classifier has further decreased the Type II error and increased the Type _I error. But the sum of both the errors is almost equal to total error of RandomForest.</font>

<font color = black face = "bold" size = 5> 3(a) Plot explaining why info is useful: </font>

<font color =black face = "bold" size = 4> Graphical Representation of Error</font>

```{r fig.height = 18, fig.width = 22, fig.align = "center"}

pdf(file = 'plot1.pdf',width = 22,height = 18)

svm.plt <-ggplot(data= new.testset.svm, aes(x=predType, fill=errorType))+ geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ ggtitle('SVM')

dt.plt <- ggplot(data= new.testset.dt, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Decision Tree')

lm.plt <- ggplot(data= new.testset.lm, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Logistic Model')

rf.plt <- ggplot(data= new.testset.rf, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Random Forest')

rf.n.svm.plt <- ggplot(data= new.testset.comb, aes(x=predType, fill=errorType))+geom_text(aes(label=..count..),stat= 'count',position = position_dodge(0.9),color = 'black',vjust=-0.2, size = 6)+geom_bar(position = 'dodge')+ggtitle('Random Forest and SVM')


plot_grid(svm.plt,dt.plt,lm.plt,rf.plt,rf.n.svm.plt)

dev.off()
```


<font color =black face = "bold" size = 4> Combined SVM and Random Forest Classifier slightly decreases the Type II error. As the combined classifier decreases the overall error along with increasing the True Positives. Combining two different classifiers was really important.  </font>

<font color = black face = "bold" size = 5> 3(b) how much does the Feature improve the classifier: </font>

<font color =black face = "bold" size = 4> Combining Random Forest to prev data frame </font>


```{r}
error.analysis.df.combination <- rbind(error.analysis.df.comb,df5)
error.analysis.df.combination
```

<font color = black face = "bold" size = 4> <b>Improvement in performance after combining Random Forest and SVM classifier: </b></font>

<font color =black face = "bold" size = 4> 1. Accuracy slightly increased.  </font>

<font color =black face = "bold" size = 4> 2. Error rate slightly decreased.  </font>

<font color =black face = "bold" size = 4> 3. Precision increased .  </font>

<font color =black face = "bold" size = 4> 4. False Positive Rate decreased.  </font>
