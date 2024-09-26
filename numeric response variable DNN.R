library(neuralnet)
library(keras)
library(dplyr)
library(mlbench)
library(magrittr)



##data (mlbench package)
data("BostonHousing")
data<-BostonHousing
str(data)


data %>% mutate_if(is.factor,as.numeric)

str(data)

data$chas=as.numeric(data$chas)


###visualize neural network

n<-neuralnet(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,
             data = data,
             hidden = c(10,5),
             linear.output = F,
             lifesign = 'full',
             rep = 1)


plot(n)

plot(n,
     col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill='lightblue')


##matrix

data<-as.matrix(data)
dimnames(data)<-NULL


##partition


set.seed(07)

ind<-sample(2,nrow(data),replace = T,prob = c(0.7,0.3))
training<-data[ind==1,1:13]
test<-data[ind==2,1:13]
trainingtarget<-data[ind==1,14]
testtarget<-data[ind==2,14]


##normalize


m<-colMeans(training)
s<-apply(training,2,sd)


training<-scale(training,center = m,scale = s)
test<-scale(test,center = m,scale = s)


##create model

model<-keras_model_sequential()

model %>% 
  layer_dense(units = 5,activation = 'relu',input_shape = c(13)) %>% 
  layer_dense(units = 1)

##compile

model %>% 
  compile(loss='mse'
          optimizer='rmsprop',
          metrics='mae')


##fit the model

mymodel<-model %>% 
  fit(training,
      trainingtarget,
      epochs=100,
      batch_size=32,
      validation_split=0.2)


##evaluate

model %>% evaluate(test,testtarget)
pred<-model %>% predict(test)

mean((testtarget-pred)^2)

plot(testtarget,pred)




###FINE TUNE MODEL


##create model

model<-keras_model_sequential()

model %>% 
  layer_dense(units =100,activation = 'relu',input_shape = c(13)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 50,activation = 'relu',input_shape = c(13)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 20,activation = 'relu',input_shape = c(13)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)

##compile

model %>% 
  compile(loss='mse'
          optimizer='rmsprop',
          metrics='mae')


##fit the model

mymodel<-model %>% 
  fit(training,
      trainingtarget,
      epochs=100,
      batch_size=32,
      validation_split=0.2)


##evaluate

model %>% evaluate(test,testtarget)
pred<-model %>% predict(test)

mean((testtarget-pred)^2)

plot(testtarget,pred)




  