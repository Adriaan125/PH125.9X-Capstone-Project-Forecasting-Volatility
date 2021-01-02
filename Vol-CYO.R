##########################################################
# Create train set and test set
##########################################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubdridate", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(neuralnet)) install.packages("neuralnet", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(dplyr)
library(readxl)
library(lubridate)
library(rpart)
library(neuralnet)
library(GGally)
library(readr)
#library(knitr)
#library(rmarkdown)
#library(latexpdf)
#library(latex2exp)
#library(tinytex)


#data import from my MAC

url <- "https://raw.githubusercontent.com/Adriaan125/PH125.9X-Capstone-Project-Forecasting-Volatility/main/USDZAR_Curncy_2.csv"

currency <- read_csv2(url)


#fullpath <- file.path("/users/Adriaan/desktop/r/edx/9. capstone/project 2/USDZAR Curncy_Jaco.xlsx")
#currency <- read_xlsx(fullpath)
#view(currency)

#add column names
names(currency) <- make.names(c("Date", "USDZAR Curncy", "LOG RETURNS", "ROLLING 30 DAY VOL", "ROLLING 7 DAY VOL", "DAILY VOL", "NEXT 7 DAY VOL"), unique=TRUE)

#remove rows with N/A
currency <- na.omit(currency)

#create train and test set from the currency datatset with a 70/30 split
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = currency$NEXT.7.DAY.VOL, times = 1, p = 0.3, list = FALSE)
train_set <- currency[-test_index,]
test_set <- currency[test_index,]


#visualisation
head(currency)   #tibble of currency dataset

dim(currency)    #total lines in currency dataset

dim(train_set)   #total lines in train dataset

dim(test_set)    #total lines in test dataset

currency %>% ggplot(aes(Date, ROLLING.30.DAY.VOL)) + 
  ggtitle("Rolling 30 day VOL") +
  geom_line()

currency %>% ggplot(aes(Date, NEXT.7.DAY.VOL)) + 
  ggtitle("Next 7 day VOL") +
  geom_line()

#guessing
guess <- mean(train_set$NEXT.7.DAY.VOL)
guess

mean((guess - test_set$NEXT.7.DAY.VOL)^2)   #squared loss of the guess

RMSE(test_set$NEXT.7.DAY.VOL, guess)        #RMSE 8.83%

#linear regression

fit <- lm(NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, data=train_set)
fit$coef

y_hat <- fit$coef[1] + fit$coef[2]*test_set$ROLLING.30.DAY.VOL
mean((y_hat - test_set$NEXT.7.DAY.VOL)^2)                 #squared loss of linear regression is lower than the squared loss of the guess


y_hat1 <- predict(fit, test_set)
mean((y_hat1 - test_set$NEXT.7.DAY.VOL)^2)  #using predict to calculate outcome
LinearReg_RMSE <- RMSE(test_set$NEXT.7.DAY.VOL, y_hat)        #RMSE 7.63%
LinearReg_RMSE


cor(train_set$NEXT.7.DAY.VOL , train_set$ROLLING.30.DAY.VOL)    #correlation between the predictor and outcome

data <- data.frame(train_set)

ggplot(data, aes(ROLLING.30.DAY.VOL, NEXT.7.DAY.VOL)) +
  geom_point() +
  geom_smooth(method='lm') +
  ggtitle("Linear regression") +
  xlab("ROLLING.30.DAY.VOL") +
  ylab("NEXT.7.DAY.VOL")

####kNN

train_knn <- train(  NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, method = "knn", data = train_set)  #by default it does cross validation 

ggplot(train_knn, highlight = TRUE)        #use ggplot to see results of cross validation and use highlight to show parameter that was optimized

train_knn <- train( NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, method = "knn", 
                   data = train_set,
                   tuneGrid = data.frame(k = seq(9, 71, 2)))   #use tunegrid to try more options for k
ggplot(train_knn, highlight = TRUE)                            #we run 30 versions of k to 25 bootstrap samples
train_knn$bestTune          #this is the best parameter that maximizes accuracy
train_knn$finalModel        #best performing model,  all up to now was on training set, not test set.


knn_fit <- knn3( NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, data = train_set, k=47)  #number of neighbours to include

y_hat_knn <- predict(knn_fit, test_set)

KNN_RMSE <- RMSE(test_set$NEXT.7.DAY.VOL, y_hat_knn)    #RMSE 17.49%
KNN_RMSE
#####Regression tree

# load data for regression tree

fit <- rpart(NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, data=train_set)   

# visualize the splits 
plot(fit, margin = 0.1)  #first split at 15.81%, we end up with 8 partitions
text(fit, cex = 0.3)   #visialize where splits were done

train_set %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(ROLLING.30.DAY.VOL, NEXT.7.DAY.VOL)) +
  geom_step(aes(ROLLING.30.DAY.VOL, y_hat), col="red") +    #final estimate of f hat of x
  ggtitle("Visual of partitions")

# change parameters                                                                          #cp is minimum the RSS must improve for another partition to be added, this avoids overtraining
fit <- rpart(NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, data=train_set, control = rpart.control(cp = 0, minsplit = 2))   #minsplit is minimum observations per partition
train_set %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(ROLLING.30.DAY.VOL, NEXT.7.DAY.VOL)) +
  geom_step(aes(ROLLING.30.DAY.VOL, y_hat), col="red") +
  ggtitle("Including minsplit")

# use cross validation to choose cp
library(caret)
train_rpart <- train( NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, method = "rpart", tuneGrid = data.frame(cp = seq(0, 0.1, len = 25)), data=train_set)
ggplot(train_rpart)    #cp = 0.004166


# access the final model and plot it
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.3)

train_set %>% 
  mutate(y_hat = predict(train_rpart)) %>% 
  ggplot() +
  geom_point(aes(ROLLING.30.DAY.VOL, NEXT.7.DAY.VOL)) +
  geom_step(aes(ROLLING.30.DAY.VOL, y_hat), col="red") +
  ggtitle("Plot of f hat of x")


#####artificial neural network
currency <- select(currency, NEXT.7.DAY.VOL, ROLLING.30.DAY.VOL, ROLLING.7.DAY.VOL, DAILY.VOL)
currency1 <- select(currency, NEXT.7.DAY.VOL, ROLLING.30.DAY.VOL)

#data visual
ggpairs(currency1, title = "Scatterplot Matrix of the Features of the currency Data Set (1 predictor)")
ggpairs(currency, title = "Scatterplot Matrix of the Features of the currency Data Set (3 predictors)")

# Split currency dataset into test and train sets with a 70/30 split. I use set.seed for every model to compensate for randomness.
set.seed(12345)
train_set <- sample_frac(tbl = currency, replace = FALSE, size = 0.70)
test_set <- anti_join(currency, train_set)


#1st Regression ANN
#To begin we construct a 1-hidden layer ANN with 1 neuron, the simplest of all neural networks.

set.seed(12321)
train_NN1 <- neuralnet( NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, data = train_set)

plot(train_NN1, rep = 'best')   #all parameters of regression and results of neural network on train set

#plot shows weights learned by neural network and number of iterations before convergence and SSE on train set

NN1_Train_SSE <- sum((train_NN1$net.result - train_set[, 1])^2)/2
paste("SSE: ", round(NN1_Train_SSE, 4))

#we use SSE to compare on train and test data
Test_NN1_Output <- compute(train_NN1, test_set[, 2])$net.result    #[ ,2] column has predictor
NN1_Test_SSE <- sum((Test_NN1_Output - test_set[, 1])^2)/2         #[ ,1] column has outcome
paste("SSE: ", round(NN1_Test_SSE, 4))                  #SSE is lower on test vs train set

NN1_Test_RMSE <- RMSE(test_set$NEXT.7.DAY.VOL, Test_NN1_Output)

##Regression Hyperparameters

# 3-Hidden Layers, Layer-1 16-neurons, Layer-2, 4-neurons, Layer-3, 4 neurons, logistic activation
# function
set.seed(12321)
train_NN2 <- neuralnet(NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, 
                       data = train_set, 
                       hidden = c(16, 4, 4), 
                       act.fct = "logistic")     #logistic smooths the results

plot(train_NN2, rep = "best")

## Training Error
NN2_Train_SSE <- sum((train_NN2$net.result - train_set[, 1])^2)/2

## Test Error
Test_NN2_Output <- compute(train_NN2, test_set[, 2])$net.result
NN2_Test_SSE <- sum((Test_NN2_Output - test_set[, 1])^2)/2

NN2_Test_RMSE <- RMSE(test_set$NEXT.7.DAY.VOL, Test_NN2_Output)

# 2-Hidden Layers, Layer-1 4-neurons, Layer-2, 1-neuron, tanh activation
# function
set.seed(12321)
train_NN3 <- neuralnet(NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, 
                       data = train_set, 
                       hidden = c(4, 1), 
                       act.fct = "tanh")      #tanh is another way of smoothing the results

plot(train_NN3, rep = "best")

## Training Error
NN3_Train_SSE <- sum((train_NN3$net.result - train_set[, 1])^2)/2

## Test Error
Test_NN3_Output <- compute(train_NN3, test_set[, 2])$net.result
NN3_Test_SSE <- sum((Test_NN3_Output - test_set[, 1])^2)/2

NN3_Test_RMSE <- RMSE(test_set$NEXT.7.DAY.VOL, Test_NN3_Output)

#1-Hidden Layer, 1-neuron, tanh activation function
set.seed(12321)
train_NN4 <- neuralnet(NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL, 
                       data = train_set, 
                       act.fct = "tanh",
                       stepmax = 1e7)           #added stepmax because I reached the limit before the algorithm could converge to 1

plot(train_NN4, rep = "best")

## Training Error
NN4_Train_SSE <- sum((train_NN4$net.result - train_set[, 1])^2)/2

## Test Error
Test_NN4_Output <- compute(train_NN4, test_set[, 2])$net.result
NN4_Test_SSE <- sum((Test_NN4_Output - test_set[, 1])^2)/2

NN4_Test_RMSE <- RMSE(test_set$NEXT.7.DAY.VOL, Test_NN4_Output)



# 3-Hidden Layers, Layer-1 16-neurons, Layer-2, 4-neurons, Layer-3, 4 neurons, logistic activation
# function
set.seed(12321)
train_NN5 <- neuralnet(NEXT.7.DAY.VOL ~ ROLLING.30.DAY.VOL + ROLLING.7.DAY.VOL + DAILY.VOL, 
                       data = train_set, 
                       hidden = c(16, 4, 4), 
                       act.fct = "logistic",
                       stepmax = 1e7)     #logistic smooths the results

plot(train_NN5, rep = "best")

## Training Error
NN5_Train_SSE <- sum((train_NN5$net.result - train_set[, 1])^2)/2

## Test Error
Test_NN5_Output <- compute(train_NN5, test_set[, 2:4])$net.result        #predictors are column 2 to 4
NN5_Test_SSE <- sum((Test_NN5_Output - test_set[, 1])^2)/2       

NN5_Test_RMSE <- RMSE(test_set$NEXT.7.DAY.VOL, Test_NN5_Output)

# Bar plot of results
Regression_NN_Errors <- tibble(Network = rep(c("NN1", "NN2", "NN3", "NN4", "NN5"), each = 2), 
                               DataSet = rep(c("Train", "Test"), time = 5), 
                               SSE = c(NN1_Train_SSE, NN1_Test_SSE, 
                                       NN2_Train_SSE, NN2_Test_SSE, 
                                       NN3_Train_SSE, NN3_Test_SSE, 
                                       NN4_Train_SSE, NN4_Test_SSE,
                                       NN5_Train_SSE, NN5_Test_SSE))

Regression_NN_Errors %>% 
  ggplot(aes(Network, SSE, fill = DataSet)) + 
  geom_col(position = "dodge") + 
  ggtitle("SSE's of neural network regressions")      #SSE results of train and test set

#Summary of all RMSE and SSE from the 4 neural networks attempted
NN1_Train_SSE
NN1_Test_SSE 
NN2_Train_SSE
NN2_Test_SSE
NN3_Train_SSE
NN3_Test_SSE 
NN4_Train_SSE
NN4_Test_SSE
NN5_Train_SSE
NN5_Test_SSE

NN1_Test_RMSE
NN2_Test_RMSE
NN3_Test_RMSE
NN4_Test_RMSE
NN5_Test_RMSE

#Summary of RMSE from all machine learning methods attempted
summary <- tibble( Model = c("Linear Reg", "KNN", "Neural Network"),
                   RMSE = c(LinearReg_RMSE*100, KNN_RMSE*100, NN5_Test_RMSE*100))

summary
ggplot(summary, aes(Model, RMSE)) + geom_col()


#options(max.print=100000)
#View(as_data_frame(...))



