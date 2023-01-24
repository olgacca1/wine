library(tidyverse)
library(tidytext)
library(tm)
library(corpus)

#dataset ripulito
#=================
vino <- readr::read_csv("winemag-data-130k-v2.csv") %>%
  mutate(qp=points/price) %>% #aggiunge variabile qualità/prezzo
  rename(ID=X1)%>%
  select(-c(designation, province, region_1, region_2, taster_twitter_handle,
            title, winery)) %>%
  mutate(variety=fct_lump(variety, 20))%>%
  na.omit

#trasformazione dei testi delle recensioni in corpus
#===================================================
corpus <- VectorSource (vino$description) %>% VCorpus

#pulizia dei testi
corpus_clean <- corpus %>%
  tm_map (tolower) %>% #riduce tutto a minuscole
  tm_map (removeWords, c(stopwords("SMART"), #rimuove stopword
                         "wine","flavors","flavor","aromas","palate",
                         "finish","drink","notes","nose","offers",
                         "texture","shows","character","structure",
                         "made", "tannins"))%>% 
  tm_map (stemDocument) %>% #fa stemming
  tm_map (removePunctuation) %>% #rimuove punteggiatura
  tm_map (removeNumbers) %>% #rimuove numeri
  tm_map (stripWhitespace) %>%#toglie gli spazi bianchi in più
  tm_map (PlainTextDocument)
corpus_clean
dtm <- DocumentTermMatrix(corpus_clean) %>% removeSparseTerms(0.95)

term_matrix <- data.frame(as.matrix(dtm))
head(term_matrix)
dim(term_matrix)

row.names(term_matrix) <- vino$ID
head(term_matrix)







#Analisi esplorative (facoltative)
#=================================
vino %>% ggplot(aes(points)) +  geom_histogram()
vino %>% ggplot(aes(price)) +  geom_histogram() + scale_x_log10()

vino %>% ggplot(aes(points, price)) +  geom_point() + geom_smooth()

#Analizziamo il punteggio rispetto all'origine per i paesi più frequenti
vino %>% 
  mutate(country = fct_lump(country, 7)) %>% 
  ggplot(aes(country, points)) +  geom_boxplot()

#Analizziamo la tendenza dei degustatori nell'attribuire i punteggi
vino %>% 
  mutate(taster_name = fct_lump(taster_name, 10)) %>% 
  filter(!is.na(taster_name)) %>%
  ggplot(aes(taster_name, points)) + 
  geom_boxplot() +
  coord_flip()

#Si è analizzato il rapporto qualità prezzo in base ai paese e ai degustatori
vino %>%
  mutate(qp = points/price) %>%
  mutate(country = fct_lump(country, 7)) %>%
  filter(!is.na(country)) %>%
  ggplot(aes(x=country, y=qp)) +
  geom_boxplot()+  coord_flip()  

vino %>%
  mutate(qp = points/price) %>%
  mutate(taster_name = fct_lump(taster_name, 10)) %>%
  filter(!is.na(taster_name)) %>%
  ggplot(aes(x=taster_name, y=qp)) +
  geom_boxplot() +  coord_flip() 

#Calcoliamo le frequenze assolute delle parole
tidy.description <- corpus_clean$ch unnest_tokens(word, text)

#Analisi parole delle descrizioni
# carichiamo le librerie rJava e qdap
suppressPackageStartupMessages({library (rJava);library (qdap)})

df_corpus <- as.data.frame(corpus_clean)

vino.description <- tibble(text= df_corpus$text,
                           ID=vino$ID)
tidy.vino.description <- vino.description %>% unnest_tokens(word, text)
tidy.vino.description %>%
  count(word, sort = TRUE) %>%
  filter(n > 5000) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) + geom_col() + coord_flip()

wordcloud::wordcloud(tidy.vino.description %>% count(word) %>% pull(word),
                     tidy.vino.description %>% count(word) %>% pull(n), 
                     min.freq = 4000, random.color = T, colors = "blue" )




#SELEZIONE DELLE VARIABILI
#==========================
#regressione su points
#0
vino.0 <- vino %>% mutate(country = fct_lump(country, 7),
                            taster_name = fct_lump(taster_name, 7))
set.seed(1234)
split0 <- initial_split(vino.0)
train0 <- training(split0)
test0 <- testing(split0)

#lm
lm0 <- lm(points ~country + taster_name + variety , data = train0)
pred.lm0 <- predict(lm0, newdata = test0)
summary(lm0)

#Regressione stepwise
step0 <- lm0 %>% stats:::step()
pred.step0 <- predict(step0, newdata=test0)

library(glmnet)
#ridge
ridge0 <- glmnet(y = train0 %>% pull(points),
                 x = train0 %>% select(!points), alpha = 0)
plot(ridge0, xvar="lambda", label=T)

previsione <- predict(ridge0, test0 %>% select(!points) %>% data.matrix())
errori<- (test %>% pull(points) - previsione)^2 %>% apply(2,mean)
lambda_opt <- errori %>% which.min()
predict(ridge0, type="coefficients", s=ridge0$lambda[lambda_opt])
pred.ridge0 <- previsione[, lambda_opt]

#lasso
lasso0 <- glmnet(y = train0 %>% pull(points),
                 x = train0 %>% select(!points) , alpha = 1)
plot(lasso0, xvar="lambda", label=T)

previsione <- predict(lasso0, newx= test0 %>% select(!points) %>% data.matrix())
errori <- (test0 %>% pull(points) - previsione)^2 %>% apply(2,mean)
lambda_opt <- errori %>% which.min()
predict(lasso0, type="coefficients", s=lasso0$lambda[lambda_opt])
pred.lasso0 <- previsione[, lambda_opt]


err0 <- (cbind(pred.lm0,pred.step0, pred.ridge0, pred.lasso0) - test$points)^2 %>% colMeans()
tibble(c("Minimi Quadrati","Step", "Ridge", "Lasso"),err0)


#1---------------------
vino.1 <- vino %>% 
  select(-c(description, qp)) %>%
  mutate(price=log(price))
vino.1 <- cbind(vino.1, term_matrix)



library(tidymodels)
library(forcats)

set.seed(1234)
split <- initial_split(vino.1)
train <- training(split)
test <- testing(split)


#lm
lm1 <- lm(points ~ ., data = train)
pred.lm <- predict(lm1, newdata = test)
summary(lm1)

#Regressione stepwise
step1 <- lm1 %>% stats:::step()
pred.step <- predict(step1, newdata=test)

library(glmnet)
#ridge
ridge1 <- glmnet(y = train %>% pull(points),
                 x = train %>% select(!points), alpha = 0)
plot(ridge1, xvar="lambda", label=T)

previsione <- predict(ridge1, test %>% select(!points) %>% data.matrix())
errori <- (test %>% pull(points) - previsione)^2 %>% apply(2,mean)
lambda_opt <- errori %>% which.min()
predict(ridge1, type="coefficients", s=ridge1$lambda[lambda_opt])
pred.ridge <- previsione[, lambda_opt]

#lasso
lasso1 <- glmnet(y = train %>% pull(points),
                 x = train %>% select(!points) , alpha = 1)
plot(lasso1, xvar="lambda", label=T)

previsione <- predict(lasso1, newx= test %>% select(!points) %>% data.matrix())
errori <- (test %>% pull(points) - previsione)^2 %>% apply(2,mean)
lambda_opt <- errori %>% which.min()
predict(lasso1, type="coefficients", s=lasso1$lambda[lambda_opt])
pred.lasso <- previsione[, lambda_opt]


err <- (cbind(pred.lm,pred.step, pred.ridge, pred.lasso) - test$points)^2 %>% colMeans()
tibble(c("Minimi Quadrati","Step", "Ridge", "Lasso"),err)


##mod2-------------------------------------------------------------------------
vino.2 <- vino.1 %>% mutate(country = fct_lump(country, 7),
                            taster_name = fct_lump(taster_name, 7))
set.seed(1234)
split2 <- initial_split(vino.2)
train2 <- training(split2)
test2 <- testing(split2)

#lm
lm2 <- lm(points ~ ., data = train2)
pred.lm2 <- predict(lm2, newdata = test2)
summary(lm2)

#Regressione stepwise
step2 <- lm2 %>% stats:::step()
pred.step2 <- predict(step2, newdata=test2)

library(glmnet)
#ridge
ridge2 <- glmnet(y = train2 %>% pull(points),
                 x = train2 %>% select(!points), alpha = 0)
plot(ridge2, xvar="lambda", label=T)

previsione <- predict(ridge2, test2 %>% select(!points) %>% data.matrix())
errori <- (test2 %>% pull(points) - previsione)^2 %>% apply(2,mean)
lambda_opt <- errori %>% which.min()
predict(ridge2, type="coefficients", s=ridge2$lambda[lambda_opt])
pred.ridge2 <- previsione[, lambda_opt]

#lasso
lasso2 <- glmnet(y = train2 %>% pull(points),
                 x = train2 %>% select(!points) , alpha = 1)
plot(lasso2, xvar="lambda", label=T)

previsione <- predict(lasso2, newx= test %>% select(!points) %>% data.matrix())
errori <- (test %>% pull(points) - previsione)^2 %>% apply(2,mean)
lambda_opt <- errori %>% which.min()
predict(lasso2, type="coefficients", s=lasso2$lambda[lambda_opt])
pred.lasso2 <- previsione[, lambda_opt]


err2 <- (cbind(pred.lm2,pred.step2, pred.ridge2, pred.lasso2) - test$points)^2 %>% colMeans()
tibble(c("Minimi Quadrati","Step", "Ridge", "Lasso"),err2)

#ALBERI
#=====================================
#regressione su points
matrice1 <- term_matrix %>% mutate(points = vino$points)
head(matrice1)

library(rpart)
library(tidymodels)
set.seed(1234)
split <- initial_split(matrice1)
train <- training(split)
test <- testing(split)
t0 <- rpart(points~., data=train) #primo albero
plot(t0)
text(t0)
t1 <- rpart(points ~., data=train, 
            method="anova", #albero regressione 
            control=rpart.control(minbucket = 20, cp=0.002, xval=10))
plot(t1)
text(t1)
printcp(t1)
plotcp(t1)
t2 <- prune(t1, cp=t1$cptable[13,1])
plot(t2)
text(t2)
plotcp(t2)

pred1 <- predict(t1, newdata=test)
pred2 <- predict(t2, newdata=test)
err1 <- (pred1-test$points)^2 %>% mean()
err2 <- (pred2-test$points)^2 %>% mean()
c(err1, err2)

#---------------------------------------
#Alberi di classificazione su variety
matrice2 <- term_matrix %>% mutate(variety_lump = vino$variety)
head(matrice2)

library(rpart)
set.seed(1234)
split2 <- initial_split(matrice2)
train2 <- training(split2)
test2 <- testing(split2)
ct1 <- rpart(variety_lump ~ . , data=train2, method="class",
             control=rpart.control(minbucket=200, cp=0.003, xval=10))
plot(ct1)
text(ct1)
printcp(ct1)
plotcp(ct1)
ct2 <- prune(ct1, cp=ct1$cptable[22,1])
plot(ct2)
printcp(ct2)
plotcp(ct2)

#per utilizzare l'entropia al posto dell'indice di Gini
ct1b <- rpart(variety_lump ~ . , data=train2, method="class", 
              parms=list(split="information"),
              control=rpart.control(minbucket=200, cp=0.003, xval=10))
plot(ct1b)
printcp(ct1b)
plotcp(ct1b)
#tagliamo a size=4
ct2b <- prune(ct1b, cp=ct1b$cptable[30,1])
plot(ct2b)
text(ct2b)






