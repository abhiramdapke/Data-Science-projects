
---
title: "Homework-2"
author: "Abhiram Dapke"
---

```{r}
setwd('C:\\Users\\abhir\\Desktop\\ENPM-808W\\HW2')
getwd()

### Question 2 and 3

library(dplyr)



train.data <- read.csv('house_train.csv')
head(train.data,5)



test.data <- read.csv('house_test.csv')
head(test.data,5)



train.model <- lm(price2013~state,data = train.data)
summary(train.model)



train.model$coefficients[1]


coef.train.data <- data.frame(train.model$coefficients)
names(coef.train.data) <- 'Regression Values'
head(coef.train.data,5)


new.coef.train.state.data <- coef.train.data[-1, ,drop = F]
head(new.coef.train.state.data,5)


max.state <- new.coef.train.state.data[which.max(new.coef.train.state.data$`Regression Values`),,drop=F] 
max.state


min.state <- new.coef.train.state.data[which.min(new.coef.train.state.data$`Regression Values`),,drop=F] 
min.state



DC <- data.frame(state='DC')
avgDC  <- predict(train.model,DC)
avgDC


WV <- data.frame(state = 'WV')
avgWV <- predict(train.model,WV)
avgWV


train.model.on.state.and.County <- lm(price2013~state+county,data=train.data)
summary(train.model.on.state.and.County)


coef.train.data.state.and.county <- data.frame(train.model.on.state.and.County$coefficients)
names(coef.train.data.state.and.county) <- 'Regression Values'
head(coef.train.data.state.and.county,5)


new.coef.train.data.state.and.county <- coef.train.data.state.and.county[-1,, drop=F]
head(new.coef.train.data.state.and.county,5)



max.regression.value.at <- new.coef.train.data.state.and.county[which.max(new.coef.train.data.state.and.county$`Regression Values`),,drop=F]
max.regression.value.at


min.regression.value.at <- new.coef.train.data.state.and.county[which.min(new.coef.train.data.state.and.county$`Regression Values`),,drop=F]
min.regression.value.at



model.poverty <- lm(price2013~poverty, data = train.data)
summary(model.poverty)


poverty.prediction <- predict(model.poverty, test.data)
result.poverty <- as.data.frame(poverty.prediction)
names(result.poverty) <- c('predicted2013.wrt.poverty')
head(result.poverty,5)



model.price2007 <- lm(price2013~price2007, data = train.data)
summary(model.price2007)



price2007.prediction <- predict(model.price2007, test.data)
result.price2007 <- data.frame(cbind(test.data$id,price2007.prediction))
names(result.price2007) <- c('id','predicted2013.wrt.2007price')
head(result.price2007,5)


finalpred1 <- result.price2007
names(finalpred1) <- c('id','prediction')
write.csv(finalpred1,file = 'price2013 prediction based on price2007 .csv',row.names = F)



model.state <- lm(price2013~state, data = train.data)
summary(model.state)


state.prediction <- predict(model.state, test.data)
result.state <- as.data.frame(state.prediction)
names(result.state) <- c('predicted2013.wrt.state')
head(result.state,5)



model.price2007.poverty <- lm(price2013~price2007+poverty, data = train.data)
summary(model.price2007.poverty)


price2007.poverty.prediction <- predict(model.price2007.poverty, test.data)
result.price2007.poverty <- as.data.frame(cbind(test.data$id ,price2007.poverty.prediction))
names(result.price2007.poverty) <- c('id','predicted2013.wrt.2007price.and.poverty')
head(result.price2007.poverty,5)



finalpred2 <- result.price2007.poverty
names(finalpred2) <- c('id','prediction')
write.csv(finalpred2,file = 'price2013 prediction based on price2007 and poverty.csv',row.names = F)


model.price2007.state <- lm(price2013~price2007+state, data = train.data)
summary(model.price2007.state)


price2007.state.prediction <- predict(model.price2007.state, test.data)
result.price2007.state <- as.data.frame(cbind(test.data$id,price2007.state.prediction))
names(result.price2007.state) <- c('id','predicted2013.wrt.2007price.and.state')
head(result.price2007.state,5)



finalpred3 <- result.price2007.state
names(finalpred3) <- c('id','prediction')
write.csv(finalpred3,file = 'price2013 prediction based on price2007 and state.csv',row.names = F)
```

# price of 2013 with poverty and state

```{r}

model.poverty.state <- lm(price2013~poverty+state, data = train.data)
summary(model.poverty.state)



poverty.state.prediction <- predict(model.poverty.state, test.data)
result.poverty.state <- as.data.frame(poverty.state.prediction)
names(result.poverty.state) <- c('predicted2013.wrt.poverty.and.state')
head(result.poverty.state,5)
```


# price of houses 2013 with price of 2007, state and poverty

```{r}

model.price2007.state.poverty <- lm(price2013~price2007+state+poverty, data = train.data)
summary(model.price2007.state.poverty)



price2007.state.poverty.prediction <- predict(model.price2007.state.poverty, test.data)
result.price2007.state.poverty <- as.data.frame(cbind(test.data$id,price2007.state.poverty.prediction))
names(result.price2007.state.poverty) <- c('id','predicted2013.wrt.2007price, state and poverty')
head(result.price2007.state.poverty,5)



finalpred4 <- result.price2007.state.poverty
names(finalpred4) <- c('id','prediction')
write.csv(finalpred4,file = 'price2013 prediction based on price2007, state and poverty.csv',row.names = F)
```

### county data handling
```{r}

id <- which(!(test.data$county %in% levels(train.data$county)))
test.data$county[id] <- NA




model.county <- lm(price2013~county, data = train.data)
summary(model.county)
p <- predict(model.county,test.data)
```


### Data handling with county, price and state
```{r}
model.price2007.state.county <- lm(price2013~price2007+state+county, data = train.data)
summary(model.price2007.state.county)
```

### Data handling with county, price, state and poverty
```{r}
model.price2007.state.county.poverty <- lm(price2013~price2007+state+county+poverty, data = train.data)
summary(model.price2007.state.county.poverty)
```
```{r}
### Question 4
house_pred<-read.csv("house_train.csv", header = TRUE )
house_test<-read.csv("house_test.csv", header = TRUE )

pred <-lm(price2013~state, data = house_pred)

summary(pred)

min(pred[[2]][1:46])

length(pred[[1]])

   c <- data.frame(coef(pred))

Nointercept <- c[-1,,drop=FALSE]
   val <- Nointercept[7,,drop=FALSE]
   
   pin <-  Nointercept[44,,drop=FALSE]
   
   l<- data.frame(state="DC")
  Ave_1 <- predict(pred,l)
h<-data.frame(state="WV")

Ave_2 <- predict(pred,h)



County <- lm(price2013~state+county, data = house_pred)

f <- data.frame(coef(County))

  Greatest <- f[486,,drop=FALSE]
  
  least <- f[117,,drop=FALSE]

  
vo <- lm(price2013~price2007, data = house_pred)

summary (vo)

coef(vo)

y_val <- predict(vo)

er <- unname((house_pred$price2013 - y_val)^2)

mse <- sum(er)/length(house_pred$price2013)

gradientDesc <- function(x, y, learning, thres, n, max_iter) {
  plot(x, y, col = "blue", pch = 20)
  m <- runif(1, 0, 1)
  c <- runif(1, 0, 1)
  yhat <- m * x + c
  MSE <- sum((y - yhat) ^ 2) / n
  converged = F
  iterations = 0
  while(converged == F) {
    
    m_new <- m - learning * ((1 / n) * (sum((yhat - y) * x)))
    c_new <- c - learning * ((1 / n) * (sum(yhat - y)))
    m <- m_new
    c <- c_new
    yhat <- m * x + c
    MSE_new <- sum((y - yhat) ^ 2) / n
    if(MSE - MSE_new <= thres) {
      abline(c, m) 
      converged = T
      return(paste("Optimal intercept:", c, "Optimal slope:", m))
    }
    iterations = iterations + 1
    if(iterations > max_iter) { 
      abline(c, m) 
      converged = T
      return(paste("Optimal intercept:", c, "Optimal slope:", m))
    }
  }
}
1.23425067701092
0.016138802471596

gradientDesc(house_pred$price2007,house_pred$price2013,144e-12,10000,8973,25000000)



prediction <- (house_test$price2007*1.23425067701092-0.016138802471596)
     ID <-house_test$id
Final <- cbind(ID,prediction)


Final_csv <- data.frame(Final)

write.csv(Final_csv,"Predictions.csv",row.names = FALSE)
```
  
```
