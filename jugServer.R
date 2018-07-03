library(caret)
library(jug)

iris_fit<-  readRDS("./model/iris_rf.Rdata")

predict_species<-function(Sepal.Length, Sepal.Width, Petal.Length,Petal.Width){
  new_data<-data.frame(Sepal.Length=as.numeric(Sepal.Length), 
        Sepal.Width=as.numeric(Sepal.Width), 
        Petal.Length=as.numeric(Petal.Length),
        Petal.Width=as.numeric(Petal.Width))
  predict(iris_fit, newdata = new_data)
}

jug() %>%
  post("/iris_api", decorate(predict_species)) %>%
  simple_error_handler() %>%
  serve_it(host="0.0.0.0",port=5002)

