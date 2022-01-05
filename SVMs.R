library(e1071)
library(keras)

mnist <- dataset_mnist()

library(kernlab)

#----------------------------------------------------------------------------------------------------------
mnist_train <- mnist$train
mnist_test <- mnist$test


names(mnist_test)[-1] <- "label"
names(mnist_train)[-1] <- "label"




# The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.

#-------------------------------------------- Data preparation --------------------------------------------#


mnist_train$label <- factor(mnist_train$label)
summary(mnist_train$label)

mnist_test$label <- factor(mnist_test$label)
summary(mnist_test$label)

mnist_train$x<-  array_reshape(mnist_train$x, c(nrow(x_train), 784))
mnist_test$x<- array_reshape(mnist_test$x, c(nrow(x_test), 784))
# Sampling training dataset


mnist_train<-as.data.frame(mnist_train)
mnist_test<- as.data.frame(mnist_test)

set.seed(10)
sample_indices <- sample(1: nrow(mnist_train), 10000) # extracting subset of 5000 samples for modelling
train <- mnist_train[sample_indices, ]

# Scaling data 

max(train[ ,1:784]) # max pixel value is 255, lets use this to scale data
train[ , 1:784] <- train[ , 1:784]/255

test <- cbind(label = mnist_test[ 785], mnist_test[ , 1:784]/255)


plot(train[,"label"])
plot(test[,"label"])



######################################### Model Building & Evaluation ######################################

#--------------------------------------------- Linear Kernel ----------------------------------------------#
#In the first two examples library kernlab was used (not e1071)
## Linear kernel using default parameters

model1_linear <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "vanilladot", C = 1)
print(model1_linear) 

eval1_linear <- predict(model1_linear, newdata = test , type = "response")
#confusion matrix
table(eval1_linear, test$label)
c=table(eval1_linear, test$label)
c[1,1]
d=0
s=for(i in 1:10){
  d=d + c[i,i]
}  
d
accuracy=d/10000
accuracy



#--------------------------------------------- Radial Kernel ----------------------------------------------#

## Radial kernel using default parameters

model1_rbf <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "rbfdot", C = 1, kpar = "automatic")
print(model1_rbf) 

eval1_rbf <- predict(model1_rbf, newdata = test , type = "response")
#confusion matrix
table(eval1_rbf, test$label)
c=table(eval1_rbf, test$label)
c[1,1]
d=0
s=for(i in 1:10){
  d=d + c[i,i]
}  
d
accuracy=d/10000
accuracy



#---------------------------------------------------------------------
#Grid search for and C
#find optimal parameters
#10 fold CV
tune_out <- tune.svm(label ~ ., data = train ,cost=10^seq(-2,2,1),kernel="linear")

#print best values of cost and gamma
tune_out$best.parameters$cost
summary(tune_out)

#build model
svm_model <- svm(label ~ ., data = train , method="C-classification", kernel="linear",cost=tune_out$best.parameters$cost)
summary(svm_model)
eval1_linear <- predict(svm_model, newdata = test , type = "response")
#confusion matrix
table(eval1_linear, test$label)
c=table(eval1_linear, test$label)
d=0
s=for(i in 1:10){
  d=d + c[i,i]
}  
d
accuracy=d/10000
accuracy


#grid search for c and gamma 5 fold CV

tune_out <- tune.svm(label ~ ., data = train ,cost=c(0.1,10), gamma=c(0.01,0.1),kernel="radial", tunecontrol= tune.control(cross=5))

#print best values of cost and gamma
tune_out$best.parameters$cost
tune_out$best.parameters$gamma
summary(tune_out)

tune_out <- tune.svm(label ~ ., data = train ,cost=c(5,20), gamma=c(0.001,0.05), kernel="radial", tunecontrol= tune.control(cross=5))

#print best values of cost and gamma
tune_out$best.parameters$cost
tune_out$best.parameters$gamma
summary(tune_out)



#best rbfs
svm_model <- svm(label ~ ., data = train , method="C-classification", kernel="radial",cost=10, gamma=0.01)
summary(svm_model)
eval1_rbf <- predict(svm_model, newdata = test , type = "response")
#confusion matrix
table(eval1_rbf, test$label)
c=table(eval1_rbf, test$label)
d=0
s=for(i in 1:10){
  d=d + c[i,i]
}  
d
accuracy=d/10000
accuracy



svm__model <- svm(label ~ ., data = train , method="C-classification", kernel="radial",cost=20, gamma=0.05)
summary(svm__model)
eval1__rbf <- predict(svm__model, newdata = test , type = "response")
#confusion matrix
table(eval1__rbf, test$label)
c=table(eval1__rbf, test$label)
d=0
s=for(i in 1:10){
  d=d + c[i,i]
}  
d
accuracy=d/10000
accuracy





#Binary

train$label<-as.factor(ifelse(as.numeric(train$label) %% 2 ==0, "even", "odd"))
test$label<- as.factor(ifelse(as.numeric(test$label) %% 2 ==0, "even", "odd"))



svm__model <- svm(label ~ ., data = train , method="C-classification", kernel="radial",cost=20, gamma=0.05)
summary(svm__model)
eval1__rbf <- predict(svm__model, newdata = test , type = "response")
#confusion matrix
table(eval1__rbf, test$label)
c=table(eval1__rbf, test$label)
d=0
s=for(i in 1:10){
  d=d + c[i,i]
}  
d
accuracy=d/10000
accuracy                       


