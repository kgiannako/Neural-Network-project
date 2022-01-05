library(keras)

mnist <- dataset_mnist()

#----------------------------------------------------------------------------------------------------------
mnist_train <- mnist$train
mnist_test <- mnist$test


names(mnist_test)[-1] <- "label"
names(mnist_train)[-1] <- "label"

#-------------------------------------------- Data preparation --------------------------------------------


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
sample_indices <- sample(1: nrow(mnist_train), 5000) # extracting subset of 5000 samples for training
train <- mnist_train[sample_indices, ]

# Scaling data 

max(train[ ,1:784]) # max pixel value is 255, min=0
train[ , 1:784] <- train[ , 1:784]/255

test <- cbind(label = mnist_test[ 785], mnist_test[ , 1:784]/255)
plot(train[,785])

#-------------------------------------------------------------------------------------------------------------

#k means and Phi matrix
#30 centers
k<- kmeans(train[,1:784], centers = 30, iter.max = 50 )
cent<-k$centers

Phi <- matrix(rep(NA,(30+1)*5000), ncol=30+1)

#gamma=dmax/sqrt(2k)
max(dist(k$centers))/sqrt(60)
#gamma should have been 1.21

for (lin in 1:5000) {
  Phi[lin,1] <- 1    # bias column
  for (col in 1:30) {
    Phi[lin,col+1] <- exp( -0.24 * norm(as.matrix(train[lin,1:784]-k$centers[col,]),"F")^2 ) #euclidean norm squared
  }
}
exp( -0.24* norm(as.matrix(train[1,1:784]-k$centers[2,]),"F")^2 )


Phi1<- Phi[,2: ncol(Phi)]
max(Phi1)<1
min(Phi1)>0
#perceptron 
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 10 , activation = 'sigmoid', input_shape = c(30))
  
summary(model)

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_rmsprop(lr=0.4),
  metrics = 'accuracy'
)
categ<- to_categorical(train$label, 10)

# Training & Evaluation ----------------------------------------------------
# Fit model to data
history <- model %>% fit(
  Phi1 , categ,
  batch_size = 64,
  epochs = 500,
  verbose = 1,
  validation_split = 0.1
)

plot(history)

#----------------------------------------------------------------------------------------------
#k means and Phi matrix
#100 centers
k<- kmeans(train[,1:784], centers = 100, iter.max = 50 )
cent<-k$centers

#gamma=dmax/sqrt(2k)

max(dist(k$centers))/sqrt(200)
#gamma should have been ~0.6

Phi <- matrix(rep(NA,(100+1)*5000), ncol=100+1)
for (lin in 1:5000) {
  Phi[lin,1] <- 1    # bias column
  for (col in 1:100) {
    Phi[lin,col+1] <- exp( -0.24 * norm(as.matrix(train[lin,1:784]-k$centers[col,]),"F")^2 ) #euclidean norm squared
  }
}
exp( -0.24* norm(as.matrix(train[1,1:784]-k$centers[2,]),"F")^2 )
max(k$centers)/sqrt(200)
min(k$centers)

Phi3<- Phi[,2: ncol(Phi)]
max(Phi3)<1
min(Phi3)>0
#perceptron 
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 10 , activation = 'sigmoid', input_shape = c(100))

summary(model)

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_rmsprop(lr=1),
  metrics = 'accuracy'
)
categ<- to_categorical(train$label, 10)

# Training & Evaluation ----------------------------------------------------
# Fit model to data
history <- model %>% fit(
  Phi3 , categ,
  batch_size = 90,
  epochs = 350,
  verbose = 1,
  validation_split = 0.1
)

plot(history)


#---------------------------------------------------------------
# k means 70 centers
k<- kmeans(train[,1:784], centers = 70, iter.max = 100 )
cent<-k$centers

#gamma=dmax/sqrt(2k)
max(dist(k$centers))/sqrt(140)
#gamma should have been ~0.7

Phi <- matrix(rep(NA,(70+1)*5000), ncol=70+1)
for (lin in 1:5000) {
  Phi[lin,1] <- 1    # bias column
  for (col in 1:70) {
    Phi[lin,col+1] <- exp( -0.24 * norm(as.matrix(train[lin,1:784]-k$centers[col,]),"F")^2 ) #euclidean norm squared
  }
}
exp( -0.24* norm(as.matrix(train[1,1:784]-k$centers[2,]),"F")^2 )
max(k$centers)/sqrt(12)
min(k$centers)

Phi2<- Phi[,2: ncol(Phi)]
max(Phi2)<1
min(Phi2)>0

#perceptron
model2 <- keras_model_sequential()
model2 %>% 
  layer_dense(units = 10 , activation = 'sigmoid', input_shape = c(70)) 
  

summary(model2)

model2 %>% compile(
  loss = 'mse',
  optimizer = optimizer_rmsprop(lr=0.5),
  metrics = 'accuracy'
)
categ<- to_categorical(train$label, 10)

# Training & Evaluation ----------------------------------------------------
# Fit model to data
history <- model2 %>% fit(
  Phi2 , categ,
  batch_size = 90,
  epochs = 500,
  verbose = 1,
  validation_split = 0.1
)

plot(history)

#--------------------------------------------------------------------------------------------------
#TEST
set.seed(10)
sample_indices <- sample(1: nrow(mnist_test), 1000) # extracting subset of 1000 samples for testing
test <- mnist_test[sample_indices, ]



testcateg<-to_categorical(test$label, 10)

Phitest <- matrix(rep(NA,(70)*1000), ncol=70)
for(lin in 1:1000) {
  for (col in 1:70) {
      Phitest[lin,col] <- exp( -0.24 * norm(as.matrix(test[lin,1:784]-k$centers[col,]),"F")^2 ) #euclidean norm squared
  }
}


score <- model %>% evaluate(
  Phitest, testcateg ,
  verbose = 0
)

# Output metrics
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')