# Assignment 1- Analysis of new york dataset  
## By: Abhiram Dapke

**Parta**  
nyt15 data imported used in this file.
Comments: Age group categorized as per the question by using cut command. 

```{r}
nyt_data= read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt15.csv")
library(ggplot2)
head(nyt_data)
nyt_data$age_group = cut(nyt_data$Age,c(0,18,24,34,44,54,64,Inf),labels = c("<18","18-24","25-34","35-44","45-54","55-64","65+"),include.lowest = T)
```

**Partb 1 & 2**  
Comments:    
1. A histogram of impressions is plotted with respect to the age groups. This histogram depicts the number of impressions made by males and females of different age groups. It is clear that as the age increases, the proportion of impressions have reduced.Also, "<18" age group has maximum number of impressions ie. 5 impressions.  
2. The next one is the box plot between number of impressions and age group.
Here, impressions are segregated into three parts by ysing cut command.  
3. This plot pictures the density of Clicks/Impressions with Impressions>0 with respect to the age groups. It is visible that the graph reduces drastically as it crosses the ratio "0.05".   
4. This is a similar plot but with clicks>0 and in this plot, the the ratio first increases and then plummets.  
5. The next one is a box plot showing number of clicks with respect ot age group and is evident that age group 18-64 doesn't go beyond 3 impressions.  
6. It is the density of number of clicks with respect to the age group.   
7. First, ctr variable is used in which Clicks/Impressions is feeded and then it is put into CTR.  
8.A graph is plotted which shows CTR distribution within age groups.  
9.A variable 'Clickscat' is defined to segment users based on their click behavior.  
```{r}
ggplot(nyt_data,aes(x=Impressions,fill = age_group))+ggtitle("Density of Impressions with respect to age groups") +geom_histogram(binwidth=1)
ggplot(nyt_data,aes(x=age_group,y=Impressions,fill = age_group))+ggtitle("Boxplot of Impressions with respect to age groups") +geom_boxplot()
nyt_data$impsyes <-cut(nyt_data$Impressions,c(-Inf,0,Inf))
ggplot(subset(nyt_data, Impressions>0), aes(x=Clicks/Impressions,colour=age_group))+ggtitle("Analysis of Clicks/Impressions for Impressions>0") + geom_density()
ggplot(subset(nyt_data, Clicks>0), aes(x=Clicks/Impressions,colour=age_group))+ggtitle("Analysis of Clicks/Impressions for Clicks>0 wrt density") + geom_density()
ggplot(subset(nyt_data, Clicks>0), aes(x=age_group, y=Clicks,fill=age_group))+ggtitle("Boxplot of Clicks>0") + geom_boxplot()
ggplot(subset(nyt_data, Clicks>0), aes(x=Clicks, colour=age_group))+ggtitle("Density plot of Clicks")+ geom_density()
ctr = ifelse(nyt_data$Impressions!=0,nyt_data$Clicks/nyt_data$Impressions,NA)
CTR = ctr
ggplot(data = nyt_data,aes(CTR, fill = age_group,color = age_group))+geom_histogram(position = "dodge", binwidth = 0.1)+labs(title = "CTR Distribution")
nyt_data$clickscat = cut(nyt_data$Clicks,c(-Inf,0,1,2,3,4,5,Inf),include.lowest = T)
```

**Partb 3**  
Visualization and analysis of data   
Creating two dataframes - maledata and female data.  
1.Plotted two line graphs of maledata and female data with x axis=Age and y axis=Clicks. These graphs depict the number of clicks done by males sorted by their age and with female data. It is clear that most of the male and female data is concentrated in between 1 and 2 clicks and very few are above 3.  
2.These two plots summarize the data of Clicks and the number of people Signed_in with respect to age groups and around about 200000 obsverations for not signed are under 18 years old whereas all the other age categories are signed in. This is in complete contrast if we compare it wih the number of clicks as most of the observations have in between  and 1 clicks.  
```{r}
attach(nyt_data)
maledata = nyt_data[Gender == 1, ]
femaledata = nyt_data[Gender == 0, ]
ggplot(maledata,aes(x=Age,y=Clicks))+ggtitle("Clicks with respect to age(Males)") + geom_line()
ggplot(femaledata,aes(x=Age,y=Clicks))+ggtitle("Clicks with respect to age(Females)") + geom_line()
ggplot(nyt_data,aes(x=Clicks,fill = age_group), legend = T)+ggtitle("Clicks with respect to age groups") + geom_histogram(bins = 3)
ggplot(nyt_data,aes(x=Signed_In,fill = age_group), legend = T) +ggtitle("Signed in With respect to age groups") + geom_histogram(bins = 3)
```

**Part c and d**  
Comments:    
1. Reading data from nyt16-nyt22.csv files i.e. 7 files.  
2. Assigning data to the given variables nyt1,nyt2,nyt3 etc.  
3. Assigning key to each variable to differenciate between days.  
4. By using library tidyverse, first encoding '0' and '1' to 'Male' and 'Female' respectively.  
5. then, calculating CTR.  
6. After that, Comparing mean number of clicks/person for both genders across 7 days.  
7. From table:-  
a. Average number of clicks/ person for males and females separately, is almost same from day1 to day7.    
b. Overall Males have a higher average number of clicks/ person.  
8. Plotting distribution of CTR for Males and Females for Day1 to Day7.  
9. From plot :-  
a.Total Clicks remains almost same for all days except On day5. On day 5, it is the highest. This behaviour is common for both males and females, however males have a lot more clicks than females.   



```{r}
nyt1 <- read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt16.csv")
nyt2 <- read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt17.csv")
nyt3 <- read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt18.csv")
nyt4 <- read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt19.csv")
nyt5 <- read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt20.csv")
nyt6 <- read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt21.csv")
nyt7 <- read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\dataset\\nyt22.csv")
nyt1$day <- "day1"
nyt2$day <- "day2"
nyt3$day <- "day3"
nyt4$day <- "day4"
nyt5$day <- "day5"
nyt6$day <- "day6"
nyt7$day <- "day7"
nyt <- rbind(nyt1, nyt2, nyt3, nyt4, nyt5, nyt6, nyt7)
levels(nyt$day) <- list('day1' = 'day1','day2'='day2','day3' = 'day3','day4' = 'day4','day5' = 'day5', 'day6' = 'day6','day7' = 'day7')
library(tidyverse)
nyt$Gender <- ifelse(nyt$Gender == 0, 'Male','Female')
nyt$CTR <- ifelse(nyt$Impressions != 0, nyt$Clicks/nyt$Impressions,NA)
clicks.plot2 <- aggregate(nyt$Clicks,by = list(nyt$day, nyt$Gender), FUN = mean,na.rm = TRUE)
clicks.plot2
agg <- ggplot(nyt, aes(day, Clicks))
agg + geom_bar(stat = "identity") + facet_grid(.~ Gender)
```


### Question 2 - Analyse your own dataset - Heart disease uci dataset used from kaggle.  
Heart disease prediction dataset used.          
**Explanation of the dataset:-**    
This dataset contains 14 variables and a total of 303 patients tested to check for heart disease. In this dataset, the meaning of columns is not clear, so I have explained it here.  
1. age: The person's age in years  
2. sex: The person's sex (1 = male, 0 = female)  
3. cp: The chest pain experienced (Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)  
4. trestbps: The person's resting blood pressure (mm Hg)  
5. chol: The person's cholesterol measurement in mg/dl  
6. fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)  
7. restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)  
8. thalach: The person's maximum heart rate achieved  
9. exang: Exercise induced angina (1 = yes; 0 = no)  
10. oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.  
11. slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)  
12. ca: The number of major vessels (0-3)  
13. thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)  
14. target: Heart disease (0 = no, 1 = yes)  

**Diagnosis:**  
The diagnosis of heart disease is done on a combination of clinical signs and test results. The types of tests run will be chosen on the basis of what the physician thinks is going on:-  
1 ranging from electrocardiograms and cardiac computerized tomography (CT) scans, to blood tests and exercise stress tests   
2. Looking at information of heart disease risk factors led me to the following: high cholesterol, high blood pressure, diabetes, weight, family history and smoking   
3. According to another source   
4. the major factors that can't be changed are: increasing age, male gender and heredity. Note that thalassemia, one of the variables in this dataset, is heredity. Major factors that can be modified are: Smoking, high cholesterol, high blood pressure, physical inactivity, and being overweight and having diabetes. Other factors include stress, alcohol and poor diet/nutrition.  
Although this dataset has a variety of variables and a lot of combinations can be done keeping the aim as the target and plotting graphs of almost all the variables with the target and in-between themselves, I have not selected each and every one. My approach in this is to be selective and plot the important ones.  

**Comments:**     
1.Imported the csv file by using read.csv command. Imported libraries dplyr and ggplot2.  
2.To make the data usable, I have changed the target from 0 and 1 to "Yes" and "No" i.e. made it a factor instead of a numeric value. similarly I have done this to sex as well. Regarding the chest pain experienced in the patients, I have classified them into three categories:- Atypical angina, non-anginal pain and asymptomatic pain.    
3.summary of the dataset is displayed.  
4.Plotted a graph to ckeck the presence and absense of heart disease in males and females. From the graph and from the data, we can say that out of 303 poeple, around about 160 are having a heart disease and 143 don't.  
5. The graph between Chest pain and Age Count depicts the type of chest pain in patients and whether they are likely to have a heart disease or not. It is clear form the graph that atypical and non-anginal pain patients are more likely to have a heart disease than the asymptomatic pain patients.  
6. The next plot displays the variation in the rest blood pressures of male and female experiencing chest pain. It is evident that for asymptomatic and atypical anginal pain, the females have higher rest bps than males which is not the case in the non-anginal pain.  
7. On a similar note, a box plot of people having these pains and with cholesterol is plotted. surprisingly, inall the three cases, it is seen that the females have comparatively higher cholesterol as compared to males.  
```{r}
heartdata = read.csv("C:\\Users\\abhir\\Desktop\\ENPM-808W\\Hw1\\My dataset\\heart.csv")
library(ggplot2)
library(dplyr)
head(heartdata)
heartdata$target = as.factor(heartdata$target)
heartdata$target = if_else(heartdata$target == 1, "YES", "NO")
summary(heartdata)
ggplot(heartdata, aes(x=target, fill=target)) + geom_bar() + xlab("Heart Disease") + ylab("Count") + ggtitle("Analysis of Presence and Absence of Heart Disease") + scale_fill_discrete(name = "Heart Disease", labels = c("Absence", "Presence"))
ggplot(heartdata, aes(cp, fill = target))+geom_bar(position = "fill")+ggtitle("Chest Pain Analysis") + xlab("Chest Pain") + ylab("Age Count")
heartdata$sex=as.factor(heartdata$sex)
heartdata$sex = if_else(heartdata$sex == 1, "MALE", "FEMALE")
heartdata$cp = if_else(heartdata$cp == 1, "ATYPICAL ANGINA",
                       if_else(heartdata$cp == 2, "NON-ANGINAL PAIN", "ASYMPTOMATIC"))
ggplot(heartdata,aes(x=sex,y=trestbps,main= "Analysis of BP across pain type",xlab="Sex", ylab="BP")) + geom_boxplot(fill="Orange") + facet_grid(~heartdata$cp)
ggplot(heartdata,aes(x=sex,y=chol,main= "Analysis of chol aross pain type",xlab="Sex", ylab="BP")) + geom_boxplot(fill="red") + facet_grid(~heartdata$cp)
```
