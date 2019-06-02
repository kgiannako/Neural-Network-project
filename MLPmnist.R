# Trains a deep NN on the MNIST dataset.
# train.R + tuning runs.R for tuning hyperparameters

library(keras)
library(tfruns)

# Data Preparation ---------------------------------------------------


mnist <- dataset_mnist()
x_train <- mnist$train$x  #matrix 60,000 x 28 x 28
y_train <- mnist$train$y  #array 60,000
x_test <- mnist$test$x    #matrix 60,000 x 28 x 28
y_test <- mnist$test$y    #array 60,000


#matrices into arrays dim 784
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

#normalization
# Transform values into [0,1] range 0:white 255:Black-> 0:white 1:black
x_train <- x_train / 255
x_test <- x_test / 255


cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

#for kNN and nearest Centroid integer values [0,9] were used as labels
# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)



batch_size <- 128
num_classes <- 10
epochs <- 30

##PCA --------------------------------------------------------------

Xcov <- cov(x_train)
pcaX <- prcomp(Xcov)
# Creating a datatable to store and plot the
# No of Principal Components vs Cumulative Variance Explained
vexplained <- as.data.frame(pcaX$sdev^2/sum(pcaX$sdev^2))
vexplained <- cbind(c(1:784),vexplained,cumsum(vexplained[,1]))
colnames(vexplained) <- c("No_of_Principal_Components","Individual_Variance_Explained","Cumulative_Variance_Explained")
#Plotting the curve using the datatable obtained
plot(vexplained$No_of_Principal_Components,vexplained$Cumulative_Variance_Explained, xlim = c(0,100),type='b',pch=16,xlab = "Principal Componets",ylab = "Cumulative Variance Explained",main = 'Principal Components vs Cumulative Variance Explained')
#Datatable to store the summary of the datatable obtained
vexplainedsummary <- vexplained[seq(0,100,5),]
vexplainedsummary
Xfinal <- as.matrix(x_train) %*% pcaX$rotation[,1:60]
testfinal <- as.matrix(x_test) %*%  pcaX$rotation[,1:60]

# Define Model --------------------------------------------------------------

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(60)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 512, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 512, activation = 'sigmoid') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'tanh')

summary(model)

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_rmsprop(lr=0.001),
  metrics = c('accuracy')
)

# Training & Evaluation ----------------------------------------------------
# Fit model to data
history <- model %>% fit(
  Xfinal, y_train,
  batch_size = 128,
  epochs = 30,
  verbose = 1,
  validation_split = 0.1
)

plot(history)

score <- model %>% evaluate(
  testfinal, y_test,
  verbose = 0
)

# Output metrics
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')

#visualisation
#devtools::install_github("andrie/deepviz")
#library(deepviz)
#library(magrittr)
#plot_model(model, to_file = "model.png", show_shapes = FALSE,show_layer_names = TRUE)
