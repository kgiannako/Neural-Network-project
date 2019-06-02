FLAGS <- flags(
  flag_numeric("dropout1", 0.1, "Dropout in first layer"),
  flag_numeric("dropout2", 0.1, "Dropout in second layer")
)

runs <- tuning_run("train.R",confirm = FALSE, flags = list(
  dropout1 = c(0.1, 0.15, 0.2,0.25, 0.3),
  dropout2 = c(0.1, 0.15, 0.2,0.25, 0.3)
  ))
runs[order(runs$metric_val_acc, decreasing=TRUE),c("metric_loss","metric_acc", "metric_val_loss", "metric_val_acc", "flag_dropout1", "flag_dropout2")]
a=runs[order(runs$metric_val_acc, decreasing=TRUE),]


FLAGS <- flags(
  flag_numeric("unit1", 256, "Units in first layer"),
  flag_numeric("unit2", 128, "Units in second layer")
)

runs <- tuning_run("train.R",confirm = FALSE, flags = list(
  unit1 = c(32,64,128,256,512),
  unit2 = c(32,64,128,256,512)
))
runs[order(runs$metric_val_acc, decreasing=TRUE),c("metric_loss","metric_acc", "metric_val_loss", "metric_val_acc", "flag_unit1", "flag_unit2")]
a=runs[order(runs$metric_val_acc, decreasing=TRUE),]


FLAGS <- flags(
  flag_integer("batch", 128 , "Batch size per epoch")
)

runs <- tuning_run("train.R",confirm = FALSE, flags = list(
  batch = c(32,64,128,256,512)
))
runs[order(runs$metric_val_acc, decreasing=TRUE),c("metric_loss","metric_acc", "metric_val_loss", "metric_val_acc", "flag_batch")]
a=runs[order(runs$metric_val_acc, decreasing=TRUE),]



FLAGS <- flags(
  flag_integer("epoch", 128 , "No. of epochs")
)
runs <- tuning_run("train.R",confirm = FALSE, flags = list(
  epoch = c(20,30,40,50,60)
))
runs[order(runs$metric_val_acc, decreasing=TRUE),c("metric_loss","metric_acc", "metric_val_loss", "metric_val_acc", "flag_epoch")]
a=runs[order(runs$metric_val_acc, decreasing=TRUE),]


FLAGS <- flags(
  flag_string("activ1", "relu" , "firstlayer activation"),
  flag_string("activ2","relu","second layer activation"),
  flag_string("activ3","relu", "output activation")  
)
runs <- tuning_run("train.R",confirm = FALSE, flags = list(
  activ1 = c("relu","sigmoid", "tanh"),
  activ2 = c("relu","sigmoid", "tanh"),
  activ3 = c("relu","sigmoid", "tanh","softmax")
  ))
runs[order(runs$metric_val_acc, decreasing=TRUE),c("metric_loss","metric_acc", "metric_val_loss", "metric_val_acc", "flag_activ1", "flag_activ2", "flag_activ3")]
a=runs[order(runs$metric_val_acc, decreasing=TRUE),]


FLAGS <- flags(
  flag_numeric("lr", 0.1 , "learning rate")
)

runs <- tuning_run("train.R",confirm = FALSE, flags = list(
  lr = c(0.1, 0.01, 0.001, 0.0001, 0.00001)
))
runs[order(runs$metric_val_acc, decreasing=TRUE),c("metric_loss","metric_acc", "metric_val_loss", "metric_val_acc", "flag_lr")]
a=runs[order(runs$metric_val_acc, decreasing=TRUE),]

